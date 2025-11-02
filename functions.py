import pandas as pd

def clean_player_df(df: pd.DataFrame) -> pd.DataFrame:
    # setting index
    df.set_index('stat_id', inplace=True)

    # enforcing consistent weekly periods 'Week 01', instead of 'Week 1'
    df['period'] = df['period'].apply(
        lambda x: f"Week {int(x.split()[-1]):02d}" if str(x).startswith("Week") else x
    )

    # drop unneeded columns
    drop_columns = ['adp', 'owner_id', 'auction_value', 'position_id', 'position_rank', 'total_rank', 'season_pts', 'sacks', 'avg_yds_allowed_perPlay', 'offensive_yds_allowed', 'tds_allowed', 'pts_against', 'def_ints', 'created_at', 'updated_at', 'pp.player_id', 'pp.player_name', 'pp.created_at']

    # create a filtered df by dropping specific columns
    keep_cols = [c for c in df.columns if c not in drop_columns]
    f_df = df.loc[
        ((df['period'].ne('Total Year')) & (df['stat_type'] == 'Actual Stats'))
        , keep_cols
    ].copy()

    # sort columns by year then week
    f_df = f_df.sort_values(['year', 'period', 'pro_team_id'])

    return f_df

# defining a team stats function
def team_stats(df: pd.DataFrame) -> pd.DataFrame:
    # per-player targets within (team,year,week)
    player_targets = (
        df.groupby(['pro_team_id','year','period', 'player_id'])['targets']
          .sum()
          .rename('player_targets')
          .reset_index()
    )


    # team total pass attempts within (team,year,week)
    team_patts = (
        df[['pro_team_id','year','period', 'pass_attempts']]
          .groupby(['pro_team_id','year','period'])['pass_attempts']
          .sum()
          .rename('team_pass_attempts')
          .reset_index()
    )


    # calculating player target %
    passing = player_targets.merge(team_patts, on=['pro_team_id', 'year', 'period'], how='outer')
    passing['player_target_pct'] = passing['player_targets'].div(passing['team_pass_attempts']).fillna(0.0)


    # per-player rushes within (team,year,week)
    player_rushes = (
        df.groupby(['pro_team_id', 'year', 'period', 'player_id'])['rush_attempts']
        .sum()
        .rename('player_rushes')
        .reset_index()
    )


    # team total rush attempts within (team,year,week)
    team_ratts = (
        df[['pro_team_id','year','period','rush_attempts']]
          .groupby(['pro_team_id','year','period'])['rush_attempts']
          .sum()
          .rename('team_rush_attempts')
          .reset_index()
    )

    # calculating player backfield %
    rushing = player_rushes.merge(team_ratts, on=['pro_team_id', 'year', 'period'], how='outer')
    rushing['backfield_pct'] = rushing['player_rushes'].div(rushing['team_rush_attempts']).fillna(0.0)

    # combining passing and rushing dfs
    out = passing.merge(
        rushing,
        on=['pro_team_id', 'year', 'period','player_id'],
        how='outer'
    )

    # calculating total team plays
    out['total_team_plays'] = out[['team_rush_attempts','team_pass_attempts']].sum(axis=1)


    # calculating team level run vs. pass
    denom = (out[['team_pass_attempts','team_rush_attempts']].sum(axis=1)).replace(0, pd.NA)
    out['team_pass_pct'] = (out['team_pass_attempts']/ denom).fillna(0.0)
    out['team_rush_pct'] = (out['team_rush_attempts']/ denom).fillna(0.0)

    # adding EWMA values
    # EWMA Targets
    out['ewma_targets'] = (
        out.groupby('player_id')['player_targets']
        .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    )

    # EWMA team stats
    out = (
        out.sort_values(['year', 'period', 'pro_team_id'])
        .assign(
            ewma_total_team_plays=lambda out: out.groupby('pro_team_id')['total_team_plays']
            .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean()),
            ewma_pass_rate=lambda out: out.groupby('pro_team_id')['team_pass_pct']
            .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean()),
            ewma_rush_rate=lambda out: out.groupby('pro_team_id')['team_rush_pct']
            .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        )
    )

    # creating a team week df with ewma values; one row per team per year per week
    # creating a copy from df and filtering to desired columns
    out = out.drop(columns = ['player_id']).copy()

    # dropping duplicates created by original df containing multiple players per team
    out = out.drop_duplicates(subset=['year', 'period', 'pro_team_id'])


    return out

