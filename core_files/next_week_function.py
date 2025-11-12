import pandas as pd
from datetime import datetime, date
import player_prop_data as ppd
from functions import clean_player_df, team_stats
from future_odds_formatter import odds_formatter

# target training_cols = ["ewma_total_team_plays","ewma_pass_rate","spread","total_ou","pass_yds_perGame_rank","rush_def_rank"]

# defining variables
API_KEY = '063c3e01050dac6fe698c6676fb5abcf'
BASE = "https://api.the-odds-api.com/v4"

# defining date / time variables and calculations
today = date.today()
today_calc = pd.Timestamp.today(tz='America/Chicago').normalize()

# Align nfl_start type & timezone
nfl_start = pd.Timestamp('2025-09-04', tz='America/Chicago')

# Days and week number (Week 1 starts at season start)
days_since_start = (today_calc - nfl_start).days
this_week_num = days_since_start// 7 + 1
next_week_num = this_week_num +1

# ------------------------------
# CREATING CLEAN PLAYER DF FOR NEXT WEEK PROJECTIONS
# ------------------------------

# importing player props, team week summary, and dfs
player_data = ppd.df.copy()

# creating clean df similar to what was used for training
clean_player_data = clean_player_df(player_data)

# prep player data for training by filtering down to QBs
player_data_copy = clean_player_data.copy()
keep_cols_player_training = ['year', 'period', 'pro_team_id', 'player_id', 'player_name', 'pass_attempts']
player_data_copy = player_data_copy[player_data_copy['pass_attempts'] > 0]
player_data_copy = player_data_copy[keep_cols_player_training]
player_data_copy['player_id'] = player_data_copy['player_id'].astype("Int64").astype("category")
player_data_copy['pro_team_id'] = player_data_copy['pro_team_id'].astype("category")
pd.set_option('display.max_rows', None)

# ------------------------------
# CREATING CLEAN TEAM STATS DF FOR NEXT WEEK PROJECTIONS
# ------------------------------
# creating team stats df
team_stats = team_stats(clean_player_data)
team_stats = team_stats.sort_values(by = ['pro_team_id','year','period'], ascending=True)

# identifying last week played by each team
# defining current year
target_year = team_stats['year'].max()  # or set explicitly, e.g., 2025
cur = team_stats.loc[team_stats['year'].eq(target_year)]

# identifying last week played by each team
idx = cur.groupby('pro_team_id')['period'].idxmax()
df_next = cur.loc[idx].reset_index(drop=True)

# merging team level data with player level data (QB)
df_next = df_next.merge(player_data_copy, on=['year', 'period', 'pro_team_id'], how='left')

# adding "next week"
df_next['period'] = f'Week {next_week_num:02d}'
pd.set_option('display.max_rows', None)

# _____________________________________
# creating odds file for joining
#______________________________________

# weekly odds file path
file_path = f'~/documents/personal/ex_futura_picks/weekly_files/bookmaker_odds_weekly_{today}.csv'
odds_df = pd.read_csv(file_path)

odds_df = odds_formatter(odds_df)
odds_df['week_label'].rename('period', inplace=True)
odds_df['period'] = f'Week {next_week_num:02d}'

# _____________________________________
# Create Defensive DF rankings
#______________________________________
# simplify DF rankings (just using csv)
defensive_file = "~/documents/personal/ex_futura_picks/defense_ranked.csv"
def_stats = pd.read_csv(defensive_file).set_index('Unnamed: 0')


# defining latest period of defensive stats to reference
def_target_year = team_stats['year'].max()  # or set explicitly, e.g., 2025
def_cur = def_stats.loc[def_stats['year'].eq(target_year)]
def_idx = def_cur.groupby('pro_team_id')['year'].idxmax()
def_stats = def_cur.loc[def_idx].reset_index(drop=True)

# _____________________________________
# joining dfs together
#______________________________________

target_cols = ["ewma_total_team_plays","ewma_pass_rate","spread","total_ou","pass_yds_perGame_rank","rush_def_rank"]
cat_cols = ["is_home", "player_id_collapsed", "pro_team_id"]  # model needs these fields too

# creating copys of key dfs
df_next = df_next.copy().reset_index(drop=True)
odds_df = odds_df.copy()
def_stats = def_stats.copy()

# filtering to needed columns
df_next = df_next[['pro_team_id','player_id','player_name','ewma_total_team_plays','ewma_pass_rate','period','pass_attempts']]
odds_df = odds_df[['pro_team_id', 'opponent_pro_team_id', 'spread','total_ou','period','is_home']]
def_stats = def_stats[['pro_team_id','pass_yds_perGame_rank','rush_def_rank']]
def_stats.rename(columns={'pro_team_id': 'opponent_pro_team_id'}, inplace=True)
def_stats = def_stats.sort_values(by='opponent_pro_team_id')

# merging dfs together
odds_df = odds_df.merge(def_stats, on=['opponent_pro_team_id'], how='left')
df_next = df_next.merge(odds_df, on=['pro_team_id','period'], how='left')

# ---------- Build next-week inference DF ----------
## Requires:
## - team_week_ewma: one row per team with latest EWMA features (must have: pro_team_id, ewma_total_team_plays, ewma_pass_rate)
## - sched_next:     next game info (must have: pro_team_id, opponent_pro_team_id (or opponent_team_id/opponent), is_home)
## - vegas_next:     next game vegas (must have: pro_team_id, spread, total_ou)
## - def_ranks:      defensive ranks by team (must have: pro_team_id, pass_yds_perGame_rank, rush_def_rank)
##
## Optional:
## - qb_map:         mapping of starting QB per team (columns: pro_team_id, player_id). If omitted, player_id -> OTHER.
##
## Uses your artifacts produced during training:
## - artifacts/qb_pass_attempts_levels.json
## - artifacts/qb_pass_attempts_z_params.json
## - artifacts/qb_pass_attempts_collapse.json
#
#import json
#import pandas as pd
#import numpy as np
#
#TRAINING_COLS = [
#    "ewma_total_team_plays",
#    "ewma_pass_rate",
#    "spread",
#    "total_ou",
#    "pass_yds_perGame_rank",
#    "rush_def_rank",
#]
#CAT_COLS = ["is_home", "player_id_collapsed", "pro_team_id"]  # model needs these fields too
#
#def _coalesce(cols: list, df: pd.DataFrame):
#    """Return first existing column name from cols list."""
#    for c in cols:
#        if c in df.columns:
#            return c
#    raise KeyError(f"None of the columns {cols} found in dataframe.")
#
#def build_train_df_next_team(
#    team_week_ewma: pd.DataFrame,
#    sched_next: pd.DataFrame,
#    vegas_next: pd.DataFrame,
#    def_ranks: pd.DataFrame,
#    qb_map: pd.DataFrame | None = None,
#    levels_path: str = "artifacts/qb_pass_attempts_levels.json",
#    z_params_path: str = "artifacts/qb_pass_attempts_z_params.json",
#    collapse_path: str = "artifacts/qb_pass_attempts_collapse.json",
#) -> pd.DataFrame:
#    # --- Load artifacts from training ---
#    with open(levels_path) as f:
#        levels = json.load(f)
#    with open(z_params_path) as f:
#        z_params = json.load(f)
#    with open(collapse_path) as f:
#        rare = json.load(f)
#    rare_ids = set(map(str, rare.get("rare_player_ids", [])))
#
#    # --- Normalize key column names ---
#    # opponent key may be named differently across inputs
#    opp_col_sched = _coalesce(["opponent_pro_team_id", "opponent_team_id", "opponent"], sched_next)
#
#    # --- Base merge: team features + schedule + vegas ---
#    base = (team_week_ewma
#            .merge(sched_next.rename(columns={opp_col_sched: "opponent_pro_team_id"}),
#                   on="pro_team_id", how="inner")
#            .merge(vegas_next[["pro_team_id", "spread", "total_ou"]],
#                   on="pro_team_id", how="left"))
#
#    # --- Defensive ranks come from opponent ---
#    def_ranks_req = ["pro_team_id", "pass_yds_perGame_rank", "rush_def_rank"]
#    missing = [c for c in def_ranks_req if c not in def_ranks.columns]
#    if missing:
#        raise KeyError(f"def_ranks missing columns: {missing}")
#
#    base = base.merge(
#        def_ranks[def_ranks_req].rename(columns={"pro_team_id": "opponent_pro_team_id"}),
#        on="opponent_pro_team_id", how="left"
#    )
#
#    # --- Attach player_id (QB) if provided; else OTHER ---
#    if qb_map is not None and "player_id" in qb_map.columns and "pro_team_id" in qb_map.columns:
#        base = base.merge(qb_map[["pro_team_id", "player_id"]], on="pro_team_id", how="left")
#    else:
#        base["player_id"] = np.nan  # will become OTHER below
#
#    # --- Types / coercions for numeric predictors ---
#    for c in TRAINING_COLS + ["is_home"]:
#        if c in base.columns:
#            base[c] = pd.to_numeric(base[c], errors="coerce")
#    base["is_home"] = base["is_home"].fillna(0).astype(int)
#
#    # --- Collapse rare / unseen players ---
#    # convert to str for comparison with saved rare_ids
#    base["player_id_str"] = base["player_id"].astype("Int64").astype(str)
#    base["player_id_collapsed"] = np.where(
#        base["player_id_str"].isin(rare_ids) | base["player_id_str"].isna(),
#        "OTHER",
#        base["player_id_str"]
#    )
#
#    # --- Enforce deterministic categorical levels (matching training) ---
#    team_levels   = levels["pro_team_id_levels"]
#    player_levels = levels["player_id_collapsed_levels"]
#
#    base["pro_team_id"] = pd.Categorical(base["pro_team_id"].astype(str),
#                                         categories=team_levels, ordered=True)
#    base["player_id_collapsed"] = pd.Categorical(base["player_id_collapsed"].astype(str),
#                                                 categories=player_levels, ordered=True)
#
#    # --- Create standardized (_z) columns using training params ---
#    for col, params in z_params.items():
#        if col in base.columns:
#            mu = params.get("mean", 0.0)
#            sd = params.get("std", 0.0)
#            if sd and sd > 0:
#                base[col + "_z"] = (base[col] - mu) / sd
#            else:
#                base[col + "_z"] = 0.0
#
#    # --- Select & order columns for the model ---
#    # Keep raw + z so you can choose in the formula like you did during training
#    z_cols_present = [c for c in (c + "_z" for c in TRAINING_COLS) if c in base.columns]
#    out_cols = TRAINING_COLS + z_cols_present + CAT_COLS + ["opponent_pro_team_id", "player_id"]
#    out_cols = [c for c in out_cols if c in base.columns]  # safety
#
#    df_nextm = base[out_cols].dropna(subset=TRAINING_COLS)  # ensure predictors present
#
#    return df_nextm
#