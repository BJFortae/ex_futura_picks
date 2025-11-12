# ===== Predict next week's QB pass attempts =====
import pandas as pd
import numpy as np
import json
import arviz as az
import bambi as bmb
import create_passAttempts_training_df as tdf
from sklearn.metrics import r2_score
from datetime import date
from xarray import DataArray

# import holdout df
holdout_df = tdf.holdout_df.copy()

# ------------------------------
# LOADING ARTIFACTS
# ______________________________

# file paths for model and artifacts
run_id = "673c2a"
model_path = f"/Users/brendenforte/documents/personal/ex_futura_picks/models/qb_pass_attempts_idata_{run_id}.nc"
artifact_dir = f"/Users/brendenforte/documents/personal/ex_futura_picks/artifacts/{run_id}"

# import model
idata = az.from_netcdf(model_path)

# import player_id collapse mapping used for training
with open(f'{artifact_dir}/qb_pass_attempts_collapse.json') as f:
    collapse = json.load(f)
rare_ids = set(map(str, collapse['rare_player_ids']))

# import z-score scaling params used for training
with open(f'{artifact_dir}/qb_pass_attempts_z_params.json') as f:
    z_params = json.load(f)

# import levels used for training
with open(f'{artifact_dir}/qb_pass_attempts_levels.json') as f:
    levels = json.load(f)

# import manifest used for training
with open(f'{artifact_dir}/manifest.json') as f:
    manifest = json.load(f)

print(manifest)

# ------------------------------
# normalizing df
# ______________________________

# Drop rows with any missing target/predictors used below
training_cols = [
    "ewma_total_team_plays","ewma_pass_rate","spread","total_ou","pass_yds_perGame_rank","rush_def_rank"
]
cat_cols = ["is_home", "player_id","pro_team_id"]
target_col = "pass_attempts"  # add this near the top

# Numeric predictors (coerce if needed)
for c in training_cols + ['is_home']:
    if c in holdout_df.columns:
        holdout_df[c] = pd.to_numeric(holdout_df[c], errors="coerce")

# extra eval columns
extra_eval_cols = [c for c in ["period"] if c in holdout_df.columns]
need_cols = training_cols + cat_cols + [target_col] + extra_eval_cols

# ------------------------------
# creating clean holdout data set
# ______________________________

# creating clean data set that mirrors my training data set
holdout_dfm = holdout_df[need_cols].dropna(subset=training_cols+cat_cols+[target_col]).copy()

# preparing player ids for loading collapse from training
holdout_dfm["player_id"] = holdout_dfm["player_id"].astype(str)

# collapse using the saved mapping (no re-detection) from training
holdout_dfm["player_id_collapsed"] = np.where(
    holdout_dfm["player_id"].isin(rare_ids),
    "OTHER",
    holdout_dfm["player_id"]
)

# ensuring categories and levels are ordered the exact same way as training
pro_team_levels = levels["pro_team_id_levels"]
player_levels   = levels["player_id_collapsed_levels"]

# optional sanity: verify shape matches model expectations
print(f"Training levels: {len(pro_team_levels)} teams, {len(player_levels)} players")

# enforce the exact same categories and order
holdout_dfm["pro_team_id"] = pd.Categorical(
    holdout_dfm["pro_team_id"].astype(str),
    categories=pro_team_levels,
    ordered=True
)

holdout_dfm["player_id_collapsed"] = pd.Categorical(
    holdout_dfm["player_id_collapsed"].astype(str),
    categories=player_levels,
    ordered=True
)

# keep is_home numeric, not categorical
holdout_dfm["is_home"] = holdout_dfm["is_home"].astype(int)


# ------------------------------
# REBUILD Z_Scores
# ______________________________

for col, stats in z_params.items():
    if col in holdout_dfm.columns and stats.get('std', 0):
        holdout_dfm[f'{col}_z'] = (holdout_dfm[col] - stats["mean"]) / stats["std"]

# Use the _z columns in the formula if created
plays_term  = "ewma_total_team_plays_z"     if "ewma_total_team_plays_z"        in holdout_dfm.columns else "ewma_total_team_plays"
prate_term  = "ewma_pass_rate_z"            if "ewma_pass_rate_z"               in holdout_dfm.columns else "ewma_pass_rate"
spread_term = "spread_z"                    if "spread_z"                      in holdout_dfm.columns else "spread"
total_term  = "total_ou_z"                  if "total_ou_z"                    in holdout_dfm.columns else "total_ou"
pass_def_term = "pass_yds_perGame_rank_z"   if "pass_yds_perGame_rank_z"      in holdout_dfm.columns else "pass_yds_perGame_rank"
rush_def_term = "rush_def_rank_z"           if "rush_def_rank_z"                in holdout_dfm.columns else "rush_def_rank"

# -----------------------------
# 5) Rebuild the model shell (no refit) and predict with saved idata
# -----------------------------

formula = (
    f"{target_col} ~ 1 + is_home + "
    f"{plays_term} + {prate_term} + {spread_term} + {total_term} + "
    f"{pass_def_term} + {rush_def_term} + "
    f"(1|pro_team_id) + (1|player_id_collapsed)"
)

# dropping unknown player id from dfm to quickly test model
holdout_dfm = holdout_dfm.drop(holdout_dfm.loc[holdout_dfm["player_id"] == '4427366'].index)
holdout_df_na = holdout_dfm[holdout_dfm.isna().any(axis=1)]
print(len(holdout_dfm['player_id_collapsed'].unique()), len(holdout_dfm['player_id_collapsed'].unique()))

# Likelihood: same as training (Negative Binomial typical for counts)
model = bmb.Model(formula, data=holdout_dfm, family="negativebinomial")

# Posterior predictive samples for the holdout rows
# --- Posterior predictive samples for the holdout rows (robust to Bambi versions) ---
# Posterior predictive samples for the holdout rows
ppc = model.predict(idata=idata, data=holdout_dfm, kind="pps", inplace=False)
da = ppc.posterior_predictive[target_col]  # xarray DataArray

# stack chain/draw -> sample
stack_dims = tuple(d for d in ("chain","draw") if d in da.dims)
if stack_dims:
    da = da.stack(sample=stack_dims)

# find obs dim automatically
obs_dim = next(d for d in da.dims if d != "sample")  # handles 'obs' or 'pass_attempts_obs'
da = da.transpose("sample", obs_dim)
pps = da.values  # (samples, n_obs)
print("pps shape (samples, n_obs):", pps.shape)

pred_mean = pps.mean(axis=0)
lower, upper = np.percentile(pps, [5, 95], axis=0)

# -----------------------------
# 6) Attach predictions and score
# -----------------------------
out = holdout_dfm.copy()
out["predicted"] = pred_mean
out["lower_90"] = lower
out["upper_90"] = upper
out["error"] = out[target_col] - out["predicted"]
out["abs_error"] = out["error"].abs()

MAE = float(out["abs_error"].mean())
RMSE = float(np.sqrt((out["error"] ** 2).mean()))
r2 = r2_score(y_true=out[target_col], y_pred=out["predicted"])

print(f"Holdout size: {len(out)}  |  MAE: {MAE:.2f}  |  RMSE: {RMSE:.2f} | R2: {r2:.2f}")

print(holdout_df[['pro_team_id','pass_attempts']])

# Optional: quick team-level aggregation (useful for sanity checks)
by_team = (out[out['pass_attempts'] != 0]
    .groupby(['period','pro_team_id'], observed=True)[["pass_attempts","predicted","error"]]
    .mean()
    .sort_values("error"))

print(len(by_team),by_team)

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

today = date.today()
# by_team.to_csv(f"/Users/brendenforte/documents/personal/ex_futura_picks/predictions/qb_pass_attempts_predictions_{today}.csv")
