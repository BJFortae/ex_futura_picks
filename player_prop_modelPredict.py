# ===== Predict next week's QB pass attempts =====
import pandas as pd
import numpy as np
import json
import arviz as az
import bambi as bmb
import player_prop_data as ppd
from xarray import DataArray
from datetime import datetime, date
import nfl_defensive_stats as ds

# defining date variables
nfl_start = '2025-09-04'
today = date.today().strftime("%Y%m%d")
days_since_start = (datetime.strptime(today, '%Y%m%d') - datetime.strptime(nfl_start, '%Y-%m-%d')).days
weeks_since_start = days_since_start // 7
next_week = weeks_since_start + 1

# importing dfs from other files
holdout_df = ppd.holdout_df.copy()
defense_ranked_2025 = ds.def_ranked_2025.copy()

# ------------------------------
# LOADING ARTIFACTS
# ______________________________

# import model
idata = az.from_netcdf("models/qb_pass_attempts_idata.nc")

# import player_id collapse mapping used for training
with open("artifacts/qb_pass_attempts_collapse.json") as f:
    collapse = json.load(f)
rare_ids = set(map(str, collapse['rare_player_ids']))

# import z-score scaling params used for training
with open("artifacts/qb_pass_attempts_z_params.json") as f:
    z_params = json.load(f)

# import levels used for training
with open("artifacts/qb_pass_attempts_levels.json") as f:
    levels = json.load(f)

# ------------------------------
# normalizing df
# ______________________________

# identifying last week played by each team
target_year = holdout_df['year'].max()  # or set explicitly, e.g., 2025
cur = holdout_df.loc[holdout_df['year'].eq(target_year)]

idx = cur.groupby('pro_team_id')['period'].idxmax()
df_next = cur.loc[idx].reset_index(drop=True)
df_next = df_next[['pro_team_id', 'player_id', 'ewma_total_team_plays','ewma_pass_rate']]

pd.set_option('display.max_rows', None)

# building next_df with future data (e.g. odds from next weeks games)
df_next['period'] = f'Week {next_week:02d}'

print(df_next)


# Drop rows with any missing target/predictors used below
training_cols = [
    "ewma_total_team_plays","ewma_pass_rate","spread","total_ou","pass_yds_perGame_rank","rush_def_rank"
]
cat_cols = ["is_home", "player_id","pro_team_id"]
target_col = "pass_attempts"

# Numeric predictors (coerce if needed)
for c in training_cols + [target_col] + ['is_home']:
    if c in df_next.columns:
        df_next[c] = pd.to_numeric(df_next[c], errors="coerce")

# ------------------------------
# creating clean holdout data set
# ______________________________

# creating clean data set that mirrors my training data set
df_nextm = df_next[training_cols+cat_cols+[target_col]].dropna(subset=training_cols+cat_cols+[target_col]).copy()

# preparing player ids for loading collapse from training
df_nextm["player_id"] = df_nextm["player_id"].astype(str)

# collapse using the saved mapping (no re-detection) from training
df_nextm["player_id_collapsed"] = np.where(
    df_nextm["player_id"].isin(rare_ids),
    "OTHER",
    df_nextm["player_id"]
)

# ensuring categories and levels are ordered the exact same way as training
pro_team_levels = levels["pro_team_id_levels"]
player_levels   = levels["player_id_collapsed_levels"]

# optional sanity: verify shape matches model expectations
print(f"Training levels: {len(pro_team_levels)} teams, {len(player_levels)} players")

# enforce the exact same categories and order
df_nextm["pro_team_id"] = pd.Categorical(
    df_nextm["pro_team_id"].astype(str),
    categories=pro_team_levels,
    ordered=True
)

df_nextm["player_id_collapsed"] = pd.Categorical(
    df_nextm["player_id_collapsed"].astype(str),
    categories=player_levels,
    ordered=True
)

# keep is_home numeric, not categorical
df_nextm["is_home"] = df_nextm["is_home"].astype(int)


# ------------------------------
# REBUILD Z_Scores
# ______________________________

for col, stats in z_params.items():
    if col in df_nextm.columns and stats.get('std', 0):
        df_nextm[f'{col}_z'] = (df_nextm[col] - stats["mean"]) / stats["std"]

# Use the _z columns in the formula if created
plays_term  = "ewma_total_team_plays_z"     if "ewma_total_team_plays_z"        in df_nextm.columns else "ewma_total_team_plays"
prate_term  = "ewma_pass_rate_z"            if "ewma_pass_rate_z"               in df_nextm.columns else "ewma_pass_rate"
spread_term = "spread_z"                    if "spread_z"                      in df_nextm.columns else "spread"
total_term  = "total_ou_z"                  if "total_ou_z"                    in df_nextm.columns else "total_ou"
pass_def_term = "pass_yds_perGame_rank_z"   if "pass_yds_perGame_rank_z"      in df_nextm.columns else "pass_yds_perGame_rank"
rush_def_term = "rush_def_rank_z"           if "rush_def_rank_z"                in df_nextm.columns else "rush_def_rank"

# -----------------------------
# 5) Rebuild the model shell (no refit) and predict with saved idata
# -----------------------------

formula = (
    f"{target_col} ~ 1 + is_home + "
    f"{plays_term} + {prate_term} + {spread_term} + {total_term} + "
    f"{pass_def_term} + {rush_def_term} + "
    f"(1|pro_team_id) + (1|player_id_collapsed)"
)

print(df_nextm)

# Likelihood: same as training (Negative Binomial typical for counts)
model = bmb.Model(formula, data=df_nextm, family="negativebinomial")

# Posterior predictive samples for the holdout rows
# --- Posterior predictive samples for the holdout rows (robust to Bambi versions) ---
# Posterior predictive samples for the holdout rows
ppc = model.predict(idata=idata, data=df_nextm, kind="pps", inplace=False)

# ppc is an InferenceData → (chain, draw, obs)
da = ppc.posterior_predictive[target_col]
pps = da.stack(sample=("chain","draw")).transpose("sample",f"{target_col}_obs").values
print("pps shape (samples, n_obs):", pps.shape)

pred_mean = pps.mean(axis=0)
lower, upper = np.percentile(pps, [5, 95], axis=0)

# -----------------------------
# 6) Attach predictions and score
# -----------------------------
out = df_nextm.copy()
out["predicted"] = pred_mean
out["lower_90"] = lower
out["upper_90"] = upper
out["error"] = out[target_col] - out["predicted"]
out["abs_error"] = out["error"].abs()

MAE = float(out["abs_error"].mean())
RMSE = float(np.sqrt((out["error"] ** 2).mean()))

print(f"Holdout size: {len(out)}  |  MAE: {MAE:.2f}  |  RMSE: {RMSE:.2f}")

print(df_next[['pro_team_id','pass_attempts']])

# Optional: quick team-level aggregation (useful for sanity checks)
by_team = (out[out['pass_attempts'] != 0]
    .groupby(['period','pro_team_id'], observed=True)[["pass_attempts","predicted","error"]]
    .mean()
    .sort_values("error"))

print(by_team)

# Coverage of 90% credible intervals
coverage = np.mean((out[target_col] >= out["lower_90"]) & (out[target_col] <= out["upper_90"]))
print(f"Empirical 90% coverage: {coverage:.1%}")

# Quick plot: predicted vs. actual
import matplotlib.pyplot as plt
plt.scatter(out["predicted"], out[target_col], alpha=0.6)
plt.xlabel("Predicted pass attempts")
plt.ylabel("Actual pass attempts")
plt.plot([0, 60], [0, 60], "r--")  # 1:1 line
plt.show()


