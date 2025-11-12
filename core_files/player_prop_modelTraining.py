# qb_pass_attempts_bambi.py
import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import create_passAttempts_training_df as tdf
import json
import os
import uuid
from datetime import date, datetime

# checking core directories
os.makedirs("/Users/brendenforte/documents/personal/ex_futura_picks/artifacts", exist_ok=True)
os.makedirs("/Users/brendenforte/documents/personal/ex_futura_picks/models", exist_ok=True)   # <- ensure models dir exists

# creating a unique run_id so I don't overwrite previously trained models
run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"{run_date}_{uuid.uuid4().hex[:6]}"
print(run_id)

# artifact and model directories using unique run id
artifact_dir = f"/Users/brendenforte/documents/personal/ex_futura_picks/artifacts/{run_id}"
z_params_path = f"{artifact_dir}/qb_pass_attempts_z_params.json"
levels_path   = f"{artifact_dir}/qb_pass_attempts_levels.json"
collapse_path = f"{artifact_dir}/qb_pass_attempts_collapse.json"
manifest_path = f"{artifact_dir}/manifest.json"
model_path    = f"/Users/brendenforte/documents/personal/ex_futura_picks/models/qb_pass_attempts_idata_{run_id}.nc"

# create directories that don't already exist
os.makedirs(artifact_dir, exist_ok=True)


# import training data
train_df = tdf.train_df.copy()

# ---------- 2) Minimal cleaning / dtypes ----------
# Ensure IDs are categorical (grouping factors)
# train_df["player_id"]   = train_df["player_id"].astype("Int64").astype("category")
# train_df["pro_team_id"] = train_df["pro_team_id"].astype("category")

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
cnt = train_dfm.groupby('player_id', observed=True).size()
less_than = set(cnt[cnt < 3].index)
train_dfm['player_id_collapsed'] = np.where(
    train_dfm['player_id'].isin(less_than), "OTHER", train_dfm['player_id']
)

# create a persisted collapsed player mapping for training and inference
# convert to STR so the set can be compared against df_nextm later
rare_ids = set(map(str, less_than))

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

# Use the _z columns in the formula if created
plays_term  = "ewma_total_team_plays_z" if "ewma_total_team_plays_z"        in train_dfm.columns else "ewma_total_team_plays"
prate_term  = "ewma_pass_rate_z"        if "ewma_pass_rate_z"               in train_dfm.columns else "ewma_pass_rate"
spread_term = "spread_z"                 if "spread_z"                      in train_dfm.columns else "spread"
total_term  = "total_ou_z"               if "total_ou_z"                    in train_dfm.columns else "total_ou"
pass_def_term = "pass_yds_perGame_rank_z" if "pass_yds_perGame_rank_z"      in train_dfm.columns else "pass_yds_perGame_rank"
rush_def_term = "rush_def_rank_z"       if "rush_def_rank_z"                in train_dfm.columns else "rush_def_rank"

#print train_dfm
print(train_dfm.columns, train_dfm)

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

# # Fit (adjust draws/tune/target_accept to your machine)
idata = model.fit(
   draws=2000,
   tune=2000,
   target_accept=0.96,
   chains=4,
   random_seed=42,
   init = "jitter+adapt_diag"
)

# (Optional) Save the fit
az.to_netcdf(idata, model_path)

# persist the scaling params for future inference
with open(z_params_path,"w") as f:
    json.dump(z_params, f, indent=2)

with open(levels_path, "w") as f:
    json.dump(
        {
            "pro_team_id_levels": team_levels,
            "player_id_collapsed_levels": player_levels,
        },
        f,
        indent=2
    )

with open(collapse_path, "w") as f:
    json.dump({"rare_player_ids": sorted(rare_ids)}, f, indent=2)

# ---------- Save manifest for this run ----------
manifest = {
    "run_id": run_id,
    "model_path": model_path,
    "collapse_path": collapse_path,
    "levels_path": levels_path,
    "z_params_path": z_params_path,
    "formula": formula,
    "features": training_cols,
    "categoricals": cat_cols,
    "target": target_cols[0],
    "n_rows": int(len(train_dfm)),
    "random_seed": 42,
    "bambi_version": bmb.__version__
}

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

#analyze results
summary = az.summary(idata, var_names=["~Intercept"], round_to=2)
print(summary)