# qb_pass_attempts_bambi.py
import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import player_prop_data as ppd
import json

# =======================
# Example wiring
# =======================
# 1) If starting from player-level rows
# team_week = make_team_week(train_df_players)

# 2) Add EWMAs

# 3) Train set is all historical rows (exclude last if you want a true out of sample test
# res_nb = fit_team_pass_attempts_nb(team_week_ewma)

# 4) Build train_df_next_team (you proivde sched_next & vegas_next)
# train_df_next_team = build_train_df_next_team(team_week_ewma, sched_next, vegas_next)

# 5) Predict
# team_volume_pred = predict_team_volume(res_nb, train_df_next_team)
# print(team_volume_pred.head())



# ---------- 1) Load your QB-week training data ----------
# The file must have exactly these columns (you said you do):
# year, period, pro_team_id, ewma_total_team_plays, ewma_pass_rate, ewma_rush_rate,
# team, is_home, opponent, spread, total_ou, opponent_pro_team_id,
# opponent_division, opponent_conference, division, conference, week_num,
# player_id, player_name, pass_attempts
train_df = ppd.train_df.copy()
holdout_df = ppd.holdout_df.copy()

# ---------- 2) Minimal cleaning / dtypes ----------
# Ensure IDs are categorical (grouping factors)
# train_df["player_id"]   = train_df["player_id"].astype("Int64").astype("category")
# train_df["pro_team_id"] = train_df["pro_team_id"].astype("category")

# Numeric predictors (coerce if needed)
num_cols = ["ewma_total_team_plays","ewma_pass_rate","is_home","spread","total_ou","week_num"]
for c in num_cols:
    if c in train_df.columns:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

# Target
train_df["pass_attempts"] = pd.to_numeric(train_df["pass_attempts"], errors="coerce")
print(f' before drop length: {len(train_df)}')

# Drop rows with any missing target/predictors used below
model_cols = ["pass_attempts","ewma_total_team_plays","ewma_pass_rate","is_home","spread","total_ou","player_id","pro_team_id"]
train_dfm = train_df.dropna(subset=model_cols).copy()
print(f' after drop length: {len(train_df)}')

# (Optional but helpful) standardize continuous predictors for sampling stability
# ---- standardization & save params ----
z_cols = ["ewma_total_team_plays","ewma_pass_rate","spread","total_ou","week_num"]
z_params = {}

for col in z_cols:
    if col in dfm.columns:
        mu = dfm[col].mean()
        sd = dfm[col].std(ddof=0)
        z_params[col] = {"mean": float(mu), "std": float(sd)}
        if sd > 0:
            dfm[col+"_z"] = (dfm[col] - mu) / sd
        else:
            dfm[col+"_z"] = 0.0

# persist the scaling params for future inference
with open("qb_pass_attempts_z_params.json","w") as f:
    json.dump(z_params, f, indent=2)


# Use the _z columns in the formula if created
plays_term  = "ewma_total_team_plays_z" if "ewma_total_team_plays_z" in train_dfm.columns else "ewma_total_team_plays"
prate_term  = "ewma_pass_rate_z"        if "ewma_pass_rate_z"        in train_dfm.columns else "ewma_pass_rate"
spread_term = "spread_z"                 if "spread_z"                 in train_dfm.columns else "spread"
total_term  = "total_ou_z"               if "total_ou_z"               in train_dfm.columns else "total_ou"

# ---------- 3) Build & fit the Negative Binomial GLMM ----------
# Baseline fixed effects + random intercepts for QB and Team
formula = f"""
pass_attempts ~
    1 +
    {plays_term} +
    {prate_term} +
    is_home +
    {spread_term} +
    {total_term} +
    (1|player_id) +
    (1|pro_team_id)
"""

# Family: Negative Binomial (default log link)
model = bmb.Model(formula, train_dfm, family="negativebinomial")

# Fit (adjust draws/tune/target_accept to your machine)
idata = model.fit(
    draws=2000,
    tune=1000,
    target_accept=0.9,
    chains=4,
    random_seed=42,
)

# (Optional) Save the fit
az.to_netctrain_df(idata, "qb_pass_attempts_idata.nc")

# ---------- 4) Posterior predictive: training set or new holdout ----------
# Predictive samples (posterior predictive) on training data:
ppc_train = model.predict(train_dfm, kind="pps", draws=2000)  # shape: (n_rows, draws)
# Summaries:
train_dfm["pa_pred_mean"] = ppc_train.mean(axis=1)
q = np.quantile(ppc_train, [0.2, 0.5, 0.8, 0.9], axis=1).T
train_dfm["pa_p20"], train_dfm["pa_p50"], train_dfm["pa_p80"], train_dfm["pa_p90"] = q[:,0], q[:,1], q[:,2], q[:,3]

train_dfm[["year","period","pro_team_id","player_id","pass_attempts","pa_pred_mean","pa_p20","pa_p50","pa_p80","pa_p90"]].to_csv(
    "qb_pass_attempts_train_predictions.csv", index=False
)

# ---------- 5) Predict on a QB-level holdout (same schema) ----------
# Holdout must be QB-week rows with the SAME columns as training (IDs, ewma features, etc.)
# Example:
# holdout = pd.read_csv("qb_holdout_merged.csv")
# holdout["player_id"]   = holdout["player_id"].astype("Int64").astype("category")
# holdout["pro_team_id"] = holdout["pro_team_id"].astype("category")
# for c in ["ewma_total_team_plays","ewma_pass_rate","is_home","spread","total_ou","week_num"]:
#     holdout[c] = pd.to_numeric(holdout[c], errors="coerce")
# # apply same z-scales used in training (use training means/sds!)
# # quick inline re-compute using training stats captured above:
# def z(x, mu, sd): return (x - mu)/sd if (np.isfinite(sd) and sd>0) else 0.0
# holdout[plays_term]  = z(holdout["ewma_total_team_plays"], train_dfm["ewma_total_team_plays"].mean(), train_dfm["ewma_total_team_plays"].std(ddof=0)) if plays_term.endswith("_z") else holdout["ewma_total_team_plays"]
# holdout[prate_term]  = z(holdout["ewma_pass_rate"], train_dfm["ewma_pass_rate"].mean(), train_dfm["ewma_pass_rate"].std(ddof=0)) if prate_term.endswith("_z") else holdout["ewma_pass_rate"]
# holdout[spread_term] = z(holdout["spread"], train_dfm["spread"].mean(), train_dfm["spread"].std(ddof=0)) if spread_term.endswith("_z") else holdout["spread"]
# holdout[total_term]  = z(holdout["total_ou"], train_dfm["total_ou"].mean(), train_dfm["total_ou"].std(ddof=0)) if total_term.endswith("_z") else holdout["total_ou"]
# ppc_hold = model.predict(holdout, kind="pps", draws=2000)
# holdout["pa_pred_mean"] = ppc_hold.mean(axis=1)
# qh = np.quantile(ppc_hold, [0.2,0.5,0.8,0.9], axis=1).T
# holdout["pa_p20"], holdout["pa_p50"], holdout["pa_p80"], holdout["pa_p90"] = qh[:,0], qh[:,1], qh[:,2], qh[:,3]
# holdout.to_csv("qb_pass_attempts_holdout_predictions.csv", index=False)

# ---------- 6) (Optional) Quick model checks ----------
# az.loo(idata)  # quick OOS estimate
# az.plot_ppc(idata); az.plot_trace(idata);  # diagnostics
