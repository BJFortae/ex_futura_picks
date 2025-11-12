# ===== Predict next week's QB pass attempts (manifest-aware, path-simplified) =====
import os, json
import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb
from next_week_function import df_next, clean_player_data

# ------------------ Config ------------------
run_id   =  # your simplified unique id
base_dir = "/Users/brendenforte/documents/personal/ex_futura_picks"
model_path_default   = f"{base_dir}/models/qb_pass_attempts_idata_{run_id}.nc"
artifact_dir_default = f"{base_dir}/artifacts/{run_id}"
manifest_path_default = f"{artifact_dir_default}/manifest.json"

# ------------------ Load manifest (and normalize paths) ------------------
with open(manifest_path_default) as f:
    manifest = json.load(f)

# Always prefer simplified scheme for paths (override any old timestamped ones)
manifest["run_id"]       = run_id
manifest["model_path"]   = model_path_default
manifest["collapse_path"]= f"{artifact_dir_default}/qb_pass_attempts_collapse.json"
manifest["levels_path"]  = f"{artifact_dir_default}/qb_pass_attempts_levels.json"
manifest["z_params_path"]= f"{artifact_dir_default}/qb_pass_attempts_z_params.json"

# Pull fields
model_path   = manifest["model_path"]
collapse_path= manifest["collapse_path"]
levels_path  = manifest["levels_path"]
z_params_path= manifest["z_params_path"]
formula      = manifest["formula"].strip()
features     = manifest["features"]
cats         = manifest["categoricals"]           # ["is_home","player_id","pro_team_id"]
target_col   = manifest["target"]                 # "pass_attempts"

# ------------------ Load artifacts ------------------
idata = az.from_netcdf(model_path)

with open(collapse_path) as f:
    rare_ids = set(map(str, json.load(f)["rare_player_ids"]))

with open(levels_path) as f:
    levels = json.load(f)

with open(z_params_path) as f:
    z_params = json.load(f)

# ------------------ Prepare next-week data ------------------
df_nextm = df_next.copy()

# Coerce numerics
for c in features + ["is_home"]:
    if c in df_nextm.columns:
        df_nextm[c] = pd.to_numeric(df_nextm[c], errors="coerce")

# Keep only required inputs
df_nextm = df_nextm[features + cats].dropna(subset=features + cats).copy()
df_nextm[target_col] = 0

# Collapse players as in training
df_nextm["player_id"] = df_nextm["player_id"].astype(str)

# defining player lvls
player_lvls   = levels["player_id_collapsed_levels"]
if "OTHER" not in set(player_lvls):
    player_lvls = player_lvls + ["OTHER"]

df_nextm['player_id_collapsed'] = np.where(
    df_nextm['player_id'].isin(rare_ids) | ~df_nextm['player_id'].isin(player_lvls),
    "OTHER",
    df_nextm["player_id"]
)

# setting categories exactly to training levels
df_nextm["player_id_collapsed"] = pd.Categorical(
    df_nextm["player_id_collapsed"].astype(str), categories=player_lvls, ordered=True
)

# Enforce team categorical levels from training
pro_team_lvls = levels["pro_team_id_levels"]
df_nextm["pro_team_id"] = pd.Categorical(
    df_nextm["pro_team_id"].astype(str), categories=pro_team_lvls, ordered=True
)


# Binary int
df_nextm["is_home"] = df_nextm["is_home"].fillna(0).astype(int)

print(df_nextm)

# ------------------ Rebuild z-scores (manifest formula uses *_z) ------------------
for col in features:
    if col in df_nextm.columns and col in z_params and z_params[col].get("std", 0):
        m, s = z_params[col]["mean"], z_params[col]["std"]
        df_nextm[f"{col}_z"] = (df_nextm[col] - m) / s

# Validate that any *_z terms referenced in the formula exist
needed_z = [f"{c}_z" for c in features if f"{c}_z" in formula]
missing_z = [c for c in needed_z if c not in df_nextm.columns]
if missing_z:
    raise ValueError(f"Missing z-features required by formula: {missing_z}")

# ------------------ Model shell (no refit) + predict ------------------
model = bmb.Model(formula, data=df_nextm, family="negativebinomial")
ppc = model.predict(idata=idata, data=df_nextm, kind="pps", inplace=False)

# Robust extraction across Bambi/ArviZ versions
da = ppc.posterior_predictive[target_col]
obs_dim = next(d for d in da.dims if d.endswith("_obs") or d == "obs")
pps = da.stack(sample=("chain","draw")).transpose("sample", obs_dim).values

pred_mean = pps.mean(axis=0)
lower, upper = np.percentile(pps, [5, 95], axis=0)

# ------------------ Output ------------------
out = df_nextm.copy()
out["predicted"] = pred_mean
out["lower_90"]  = lower
out["upper_90"]  = upper
out[["predicted","lower_90","upper_90"]] = out[["predicted","lower_90","upper_90"]].round(1)

#formatting
out['player_id'] = out['player_id'].astype(str)
clean_player_data = clean_player_data.copy()
clean_player_data = clean_player_data[['player_id','player_name']]
clean_player_data = clean_player_data.drop_duplicates()
clean_player_data['player_id'] = clean_player_data['player_id'].astype(str)

out = out.merge(clean_player_data[['player_name','player_id']], on = "player_id", how = "left")
cols_show = [c for c in ["period","player_id","player_id_collapsed","player_name","pro_team_id","is_home","predicted","lower_90","upper_90"] if c in out.columns]
print(out[cols_show][['player_name','predicted','lower_90','upper_90']])

# Optional: save
save_path = f"{base_dir}/predictions/qb_pass_attempts_pred_{run_id}.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
out[cols_show].to_csv(save_path, index=False); print("Saved:", save_path)
