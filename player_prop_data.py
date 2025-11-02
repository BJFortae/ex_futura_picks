import math
import pandas as pd
from pandas import DataFrame
import pymysql
from functions import *
import numpy as np
from itertools import combinations

today = pd.Timestamp.today().strftime('%Y%m%d')

# formatting settings
pd.set_option("display.max_columns", None)
pd.set_option('future.no_silent_downcasting', True)

# historical odds file import
odds_file = "~/documents/personal/ex_futura_picks/training_data/PAID_1yrHistorical_odds_formatted_20251018.csv"
hist_odds = pd.read_csv(odds_file)

# defensive stats file import
def_stats_file = "~/documents/personal/ex_futura_picks/defense_ranked.csv"
def_stats = pd.read_csv(def_stats_file)
def_stats.rename(columns={'pro_team_id' : 'opponent_pro_team_id', 'total_rank':'defense_rank'}, inplace=True)

# formatting historical odds df
hist_odds.reset_index(drop=True, inplace=True)
hist_odds.drop(columns=['Unnamed: 0'], inplace=True)
hist_odds.rename(columns={'season_year': 'year', 'week_label': 'period'}, inplace=True)

#####################################################
#################### PLAYER STATS ###################
#####################################################

# Step 1: Establish a connection to the MySQL database
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='fellasleague'
)


try:
    # Step 2: Create a cursor object
    with connection.cursor() as cursor:
            # Create the correct number of %s placeholders based on the number of players
            sql = f"""
            SELECT * FROM player_stats ps
            LEFT JOIN player_by_position pp ON ps.player_id = pp.player_id
            ;
            """

            cursor.execute(sql)

            rows = cursor.fetchall()

finally:
    connection.close()


df = pd.DataFrame(rows, columns = [
'stat_id'                  ,
'player_id'                ,
'owner_id'                 ,
'pro_team_id'              ,
'year'                     ,
'period'                   ,
'stat_type'                ,
'player_name'              ,
'auction_value'            ,
'adp'                      ,
'position_id'              ,
'position'                 ,
'position_rank'            ,
'total_rank'               ,
'season_pts'               ,
'rush_attempts'            ,
'yds_perCarry'             ,
'rush_yds'                 ,
'rush_tds'                 ,
'rush_yds_perGame'         ,
'rush_firstDowns'          ,
'targets'                  ,
'receptions'               ,
'yds_perRec'               ,
'receiving_yds'            ,
'receiving_tds'            ,
'rec_yds_perGame'          ,
'receiving_firstDowns'     ,
'pass_attempts'            ,
'completions'              ,
'completion_pct'           ,
'pass_yds'                 ,
'pass_tds'                 ,
'ints_thrown'              ,
'sacks'                    ,
'avg_yds_allowed_perPlay'  ,
'offensive_yds_allowed'    ,
'tds_allowed'              ,
'pts_against'              ,
'def_ints'                 ,
'games_played'             ,
'created_at'              ,
'updated_at',
'pp.player_id',
'pp.player_name',
'pp.position',
'pp.created_at'
    ]
)

clean_player_df = clean_player_df(df)


# call team stats function
team_stats_df = team_stats(clean_player_df)
print(team_stats_df.head())


#####################################################
############ JOINING BOOKMAKER DATA #################
#####################################################
team_week = team_stats_df.merge(hist_odds, on=['year','period','pro_team_id'], how='right')

# removing empty rows to prep for training
required_cols = [
    'total_team_plays',
    'team_pass_pct',
    'team_rush_pct',
    'ewma_total_team_plays',
    'ewma_pass_rate',
    'ewma_rush_rate'
]
team_week = team_week.dropna(subset=required_cols)

####################################################
############### BUILD DF FOR TRAINING ##############
####################################################
def build_train_df (team_df: DataFrame, player_data: DataFrame) -> DataFrame:
    train_df = team_df.copy()
    player_data_copy = player_data.copy()

    # prep player data for training by filtering down to QBs
    keep_cols_player_training = ['year', 'period', 'pro_team_id', 'player_id', 'player_name', 'pass_attempts']
    player_data_copy = player_data_copy[player_data_copy['pass_attempts'] > 0]
    player_data_copy = player_data_copy[keep_cols_player_training]
    player_data_copy['player_id'] = player_data_copy['player_id'].astype("Int64").astype("category")
    player_data_copy['pro_team_id'] = player_data_copy['pro_team_id'].astype("category")

    # df formatting
    train_df.set_index('event_id', inplace=True)
    train_df["pro_team_id"] = train_df["pro_team_id"].astype("category")

    # Create a numeric week field
    train_df['week_num'] = train_df['period'].str.replace("Week ", "").astype(int).drop(columns=['period'])

    # drop columns not needed for training
    train_df.drop(columns=['total_team_plays', 'team_pass_pct', 'team_rush_pct', 'commence_date', 'bookmaker','moneyline','home_team', 'away_team', 'spread_odds','over_odds','under_odds'], inplace=True)

    # merge with QB Pass Attempts
    train_df = train_df.merge(player_data_copy, on=['year', 'period', 'pro_team_id'], how='left')

    # merge with defensive rank
    train_df = train_df.merge(def_stats,on = ['year','opponent_pro_team_id'], how='left')

    # defining train and holdout sets
    holdout_df = train_df[(train_df['year'] == 2025) & (train_df['week_num'].isin([6,7,8]))]
    train_df   = train_df[~((train_df['year'] == 2025) & (train_df['week_num'].isin([6,7,8])))]

    return train_df, holdout_df

train_df, holdout_df = build_train_df(team_week, clean_player_df)

corr_test = train_df.select_dtypes(include='number').corr()['pass_attempts'][
    ['opponent_plays_perGame_rank','opponent_pts_perGame_rank','rush_yds_perGame_rank',
     'rush_tds_perGame_rank','pass_yds_perGame_rank','pass_tds_perGame_rank',
     'ints_perGame_rank','completion_pct_allowed_perGame_rank',
     'pts_allowed_perGame_rank','total_tds_perGame_rank','pass_def_rank',
     'rush_def_rank','defense_rank']
]


# player_df.to_csv(f"player_df_{today}.csv", index=True)
# team_week.to_csv(f'team_summary_data_{today}.csv', index=True)
train_df.to_csv(f'training_data_{today}.csv', index=True)
holdout_df.to_csv(f'holdout_data_{today}.csv', index=True)

# df to csv
# f_df.to_csv("player_prop_data.csv", index=True)
# eam_week.to_csv(f'team_training_data_{today}.csv', index=True)






