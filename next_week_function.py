import pandas as pd
import numpy as np
import player_prop_data as ppd
from datetime import datetime, date

team_week = ppd.team_week.copy()

###########################
# building df next_week frame for model to use and store projections
###########################

def build_df_next_team(team_week: pd.DataFrame,
                       sched_next: pd.DataFrame | None = None,
                       vegas_next: pd.DataFrame | None = None,
                       target_week: int | None = None,
                       opp_def_ewma: pd.DataFrame | None = None  # optional
                       ) -> pd.DataFrame:
    """
    Build next-week team feature rows from historical EWMAs + known pregame info.
    Returns one row per team scheduled to play in target_week.
    """
    # defining weeks since start of NFL season so we can cleanly calculate target_week
    nfl_start_date = datetime.strptime('2025-09-04', '%Y-%m-%d').date()
    today = date.today()
    day_since_start = (today - nfl_start_date).days
    weeks_since_start = day_since_start // 7
    if target_week is None:
        target_week = weeks_since_start + 1
        target_week = f'Week {target_week:02d}'

    # 1) one latest row per team (as of last completed week)
    last = (team_week
            .sort_values(['year','period', 'pro_team_id'])
            .groupby('pro_team_id', as_index=False)
            .tail(1)
            .copy())

    # 2) set target week
    last['period'] = target_week

    # 3) keep only EWMA predictors you plan to use
    keep_cols = ['pro_team_id','period','ewma_total_team_plays','ewma_pass_rate']
    missing = [c for c in keep_cols if c not in last.columns]
    if missing:
        raise ValueError(f"Missing required EWMA cols: {missing}")
    df_next = last[keep_cols].copy()

    # 4) merge schedule (limits to teams actually playing; drops byes automatically)
    if sched_next is not None:
        need_sched = ['pro_team_id','opp_team','period']
        if not need_sched.issubset(sched_next.columns):
            raise ValueError(f"sched_next must contain {need_sched}")
        df_next = df_next.merge(sched_next[need_sched], on=['pro_team_id','period'], how='inner')

    # 5) merge Vegas (team-centric)
    if vegas_next is not None:
        need_vegas = ['pro_team_id','spread','ou']
        if not need_vegas.issubset(vegas_next.columns):
            raise ValueError(f"vegas_next must contain {need_vegas}")
        df_next = df_next.merge(vegas_next[need_vegas], on='pro_team_id', how='left')

    # 6) optional: merge opponent defensive EWMAs by opp_team
    if opp_def_ewma is not None:
        # expect columns like: ['team','ewma_opp_pass_yds_allowed','ewma_opp_rush_yds_allowed', ...]
        # rename 'team' -> 'opp_team' before merging
        opp_cols = [c for c in opp_def_ewma.columns if c != 'team']
        df_next = df_next.merge(
            opp_def_ewma.rename(columns={'pro_team_id':'opp_team'})[['opp_team'] + opp_cols],
            on='opp_team', how='left'
        )

    # 7) safety clamps & fills
    df_next['ewma_pass_rate'] = df_next['ewma_pass_rate'].clip(0, 1)
    df_next['ewma_total_team_plays'] = df_next['ewma_total_team_plays'].clip(lower=1)
    if 'spread' in df_next:
        df_next['spread'] = df_next['spread'].fillna(0.0)
    if 'ou' in df_next:
        default_ou = 44.0 if not df_next['ou'].notna().any() else df_next['ou'].median()
        df_next['ou'] = df_next['ou'].fillna(default_ou)

    # 8) dedupe & ordering
    df_next = (df_next
               .drop_duplicates(subset=['pro_team_id','period'])
               .sort_values(['pro_team_id']))

    return df_next

def train_team_pass_attempts(df: pd.DataFrame):
    df = df.copy()
    df['log_ewma_total_players'] = np.log(df['ewma_total_team_plays']).clip(lower=1)

    formula = "pass_attempts ~ log_ewma_total_plays + ewma_pass_rate + moneyline + spread + ou"
    model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial())
    res = model.fit()
    return res