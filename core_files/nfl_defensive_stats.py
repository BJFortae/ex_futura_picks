import pandas as pd
import player_prop_data as ppd

player_df = ppd.player_df.copy()

def defensive_stats(player_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    player_df = player_df.copy()
    odds_df = odds_df.copy()

    # formatting odds_df
    odds_df.drop(columns=['event_id','bookmaker','team', 'is_home','opponent', 'spread_odds','over_odds', 'under_odds'
                    ], inplace=True)

    def_df = (
        player_df
        .groupby(['pro_team_id', 'year', 'period','games_played'], as_index=False)[
            ['rush_attempts','rush_yds','rush_tds',
             'receptions','receiving_yds','receiving_tds',
             'pass_attempts','completions','pass_yds','pass_tds','ints_thrown']
        ]
        .sum()
    )


    # adding team stats
    denom = def_df['rush_attempts'] + def_df['pass_attempts']

    def_df['completions_pct'] = def_df['completions'] / def_df['pass_attempts']
    def_df['pass_pct'] = def_df['pass_attempts'] / denom
    def_df['rush_pct'] = def_df['rush_attempts'] / denom
    def_df['total_tds'] = def_df['pass_tds'] + def_df['rush_tds']
    def_df['pts_allowed'] = def_df['total_tds'] * 6
    def_df['offense_plays_against'] = def_df['pass_attempts'] + def_df['rush_attempts']


    # merging odds and team dataframes
    def_df = pd.merge(odds_df, def_df, on=['pro_team_id', 'year', 'period'], how='outer')

    # creating defensive summary stats
    def_df = (
        def_df
        .groupby(['year', 'opponent_pro_team_id'], as_index=False)[
            ['rush_attempts','rush_yds','rush_tds','pass_attempts','completions','pass_yds','pass_tds','total_tds','pts_allowed','ints_thrown','offense_plays_against','games_played']
        ]
        .sum()
    )

    # renaming columns to reflect defensive stats (allowed vs. accomplished)
    def_df.rename(columns={
        'opponent_pro_team_id': 'pro_team_id',
        'rush_attempts': 'rush_attempts_against',
        'rush_yds': 'rush_yds_against',
        'rush_tds': 'rush_tds_allowed',
        'pass_attempts': 'pass_attempts_against',
        'pass_yds': 'pass_yds_against',
        'completions': 'completions_allowed',
        'pass_tds': 'pass_tds_allowed',
        'total_tds': 'total_tds_allowed',
        'ints_thrown': 'ints'}, inplace=True
    )

    def_cols = ['year','pro_team_id','rush_attempts_against', 'rush_yds_against', 'rush_tds_allowed','pass_attempts_against', 'completions_allowed', 'pass_yds_against', 'pass_tds_allowed', 'total_tds_allowed','pts_allowed','ints','offense_plays_against','games_played']
    def_df[def_cols] = def_df[def_cols].astype(int)
    drop_cols = ['rush_attempts_against', 'rush_yds_against', 'rush_tds_allowed','pass_attempts_against', 'completions_allowed', 'pass_yds_against', 'pass_tds_allowed', 'total_tds_allowed','pts_allowed','ints','offense_plays_against',]

    # stat averages
    def_weekly_avg = def_df.copy()
    def_weekly_avg['opponent_plays_perGame'] = def_weekly_avg['offense_plays_against'] / def_weekly_avg['games_played']
    def_weekly_avg['opponent_pts_perGame'] = def_weekly_avg['pts_allowed'] / def_weekly_avg['games_played']
    def_weekly_avg['rush_yds_perGame'] = def_weekly_avg['rush_yds_against'] / def_weekly_avg['games_played']
    def_weekly_avg['rush_tds_perGame'] = def_weekly_avg['rush_tds_allowed'] / def_weekly_avg['games_played']
    def_weekly_avg['pass_yds_perGame'] = def_weekly_avg['pass_yds_against'] / def_weekly_avg['games_played']
    def_weekly_avg['pass_tds_perGame'] = def_weekly_avg['pass_tds_allowed'] / def_weekly_avg['games_played']
    pass_attempts_perGame = def_weekly_avg['pass_attempts_against'] / def_weekly_avg['games_played']
    completions_perGame = def_weekly_avg['completions_allowed'] / def_weekly_avg['games_played']
    def_weekly_avg['ints_perGame'] = def_weekly_avg['ints'] / def_weekly_avg['games_played']
    def_weekly_avg['completion_pct_allowed_perGame'] = completions_perGame / pass_attempts_perGame
    def_weekly_avg['pts_allowed_perGame'] = def_df['pts_allowed'] / def_df['games_played']
    def_weekly_avg['total_tds_perGame'] = def_df['total_tds_allowed'] / def_df['games_played']
    def_weekly_avg.drop(columns=drop_cols, inplace=True)

    ranked_cols = ['opponent_plays_perGame','opponent_pts_perGame','rush_yds_perGame','rush_tds_perGame','pass_yds_perGame','pass_tds_perGame','ints_perGame','completion_pct_allowed_perGame','pts_allowed_perGame','total_tds_perGame']

    def add_yearly_rank(df, cols):
        ranked = []
        for col in cols:
            ascending = False if col == 'ints_perGame' else True
            col_rank = f"{col}_rank"
            df[col_rank] = (
                df.groupby('year')[col]
                  .rank(method='dense', ascending=ascending)
            )
            ranked.append(col_rank)
        df['pass_def_rank'] = df[['pass_yds_perGame_rank','pass_tds_perGame_rank','ints_perGame_rank','completion_pct_allowed_perGame_rank']].sum(axis=1)/4
        df['rush_def_rank'] = df[['rush_yds_perGame_rank','rush_tds_perGame_rank']].sum(axis=1)/2
        df['total_rank'] = df[ranked].sum(axis=1)/len(ranked)
        df.sort_values(['year','total_rank'], inplace=True)
        return df

    # calling function and filtering by year
    defense_ranked = add_yearly_rank(def_weekly_avg, ranked_cols)
    def_ranked_2024 = defense_ranked[defense_ranked['year'] == 2024]
    def_ranked_2025 = defense_ranked[defense_ranked['year'] == 2025]

    return defense_ranked, def_ranked_2024, def_ranked_2025

def_ranked = defensive_stats(player_df, odds_df=)
defense_ranked.to_csv('defense_ranked.csv')
# def_ranked_2024.to_csv('def_ranked_2024.csv')
# def_ranked_2025.to_csv('def_ranked_2025.csv')