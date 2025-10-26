# qb_pass_attempts_bambi.py
import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import player_prop_data as ppd
import json
import os
os.makedirs("artifacts", exist_ok=True)

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

# ---------- 2) Minimal cleaning / dtypes ----------
# Ensure IDs are categorical (grouping factors)
# train_df["player_id"]   = train_df["player_id"].astype("Int64").astype("category")
# train_df["pro_team_id"] = train_df["pro_team_id"].astype("category")

# Drop rows with any missing target/predictors used below
training_cols = ["ewma_total_team_plays","ewma_pass_rate","spread","total_ou","pass_yds_perGame_rank","rush_def_rank"]
cat_cols = ["is_home", "player_id","pro_team_id"]
target_cols = ["pass_attempts"]

# Numeric predictors (coerce if needed)
num_cols = training_cols
for c in num_cols:
    if c in train_df.columns:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

train_df['pass_attempts'] = pd.to_numeric(train_df['pass_attempts'], errors="coerce")
train_df["is_home"] = pd.to_numeric(train_df["is_home"], errors="coerce")

# creating clean training data set
train_dfm = train_df[training_cols+cat_cols+target_cols].dropna(subset=training_cols+cat_cols+target_cols).copy()

# group low-count players together so as to not distract the training model
cnt = train_dfm.groupby('player_id').size()
less_than = set(cnt[cnt < 3].index)
train_dfm['player_id_collapsed'] = np.where(
    train_dfm['player_id'].isin(less_than), "OTHER", train_dfm['player_id']
)

# create a persisted collapsed player mapping for training and inference
# convert to STR so the set can be compared against df_nextm later
rare_ids = set(map(str, less_than))
with open("artifacts/qb_pass_attempts_collapse.json", "w") as f:
    json.dump({"rare_player_ids": sorted(rare_ids)}, f, indent=2)

# --- lock categorical levels & order (DETERMINISTIC) ---
# Teams: alphabetical (or your preferred fixed order)
team_levels = sorted(train_dfm["pro_team_id"].astype(str).unique())

# Players: put "OTHER" first, then alphabetical of the rest
player_levels = ["OTHER"] + sorted(
    set(train_dfm["player_id_collapsed"].astype(str).unique()) - {"OTHER"}
)

# Enforce categories + order that the model will bind to
train_dfm["pro_team_id"] = pd.Categorical(
    train_dfm["pro_team_id"].astype(str),
    categories=team_levels,
    ordered=True
)
train_dfm["player_id_collapsed"] = pd.Categorical(
    train_dfm["player_id_collapsed"].astype(str),
    categories=player_levels,
    ordered=True
)

with open("artifacts/qb_pass_attempts_levels.json", "w") as f:
    json.dump(
        {
            "pro_team_id_levels": team_levels,
            "player_id_collapsed_levels": player_levels,
        },
        f,
        indent=2
    )

# Keep is_home numeric (not categorical)
train_dfm["is_home"] = train_dfm["is_home"].astype(int)



# (Optional but helpful) standardize continuous predictors for sampling stability
# ---- standardization & save params ----
z_cols = training_cols
z_params = {}

for col in z_cols:
    if col in train_dfm.columns:
        mu = train_dfm[col].mean()
        sd = train_dfm[col].std(ddof=0)
        z_params[col] = {"mean": float(mu), "std": float(sd)}
        if sd > 0:
            train_dfm[col+"_z"] = (train_dfm[col] - mu) / sd
        else:
            train_dfm[col+"_z"] = 0.0

# persist the scaling params for future inference
with open("artifacts/qb_pass_attempts_z_params.json","w") as f:
    json.dump(z_params, f, indent=2)


# Use the _z columns in the formula if created
plays_term  = "ewma_total_team_plays_z" if "ewma_total_team_plays_z"        in train_dfm.columns else "ewma_total_team_plays"
prate_term  = "ewma_pass_rate_z"        if "ewma_pass_rate_z"               in train_dfm.columns else "ewma_pass_rate"
spread_term = "spread_z"                 if "spread_z"                      in train_dfm.columns else "spread"
total_term  = "total_ou_z"               if "total_ou_z"                    in train_dfm.columns else "total_ou"
pass_def_term = "pass_yds_perGame_rank_z" if "pass_yds_perGame_rank_z"      in train_dfm.columns else "pass_yds_perGame_rank"
rush_def_term = "rush_def_rank_z"       if "rush_def_rank_z"                in train_dfm.columns else "rush_def_rank"

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
    {pass_def_term} +
    {rush_def_term} +
    (1|player_id_collapsed) +
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
az.to_netcdf(idata, "models/qb_pass_attempts_idata.nc")

import arviz as az
summ = az.summary(idata, round_to=2)
summ[["r_hat","ess_bulk","ess_tail"]].max()
pd.set_option("display.max_rows", None)
print(summ)
