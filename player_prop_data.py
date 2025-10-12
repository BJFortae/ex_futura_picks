import math
import pandas as pd
import pymysql
import numpy as np
from itertools import combinations

pd.set_option("display.max_columns", None)

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
df.set_index('stat_id', inplace=True)
df['period'] = df['period'].apply(
    lambda x: f"Week {int(x.split()[-1]):02d}" if str(x).startswith("Week") else x
)

# define columns by position
drop_columns = ['adp', 'owner_id', 'auction_value', 'position_id', 'position', 'position_rank', 'total_rank', 'season_pts', 'sacks', 'avg_yds_allowed_perPlay', 'offensive_yds_allowed', 'tds_allowed', 'pts_against', 'def_ints', 'created_at', 'updated_at', 'pp.player_id', 'pp.player_name', 'pp.created_at']

# create a df by position
keep_cols = [c for c in df.columns if c not in drop_columns]
f_df = df.loc[df['pro_team_id'].eq(16), keep_cols].copy()
f_df = f_df.loc[(f_df['stat_type'].ne('Projections')) & (f_df['period'].ne('Total Year'))]

# creating QB Pass attempts column
qb_pa = (
    f_df.loc[f_df['pp.position'] == 'QB']
    .groupby(['year','period'], as_index=False)['pass_attempts']
    .sum()
    .rename(columns={'pass_attempts': 'total_pass_attempts'})
)

backfield = (
    f_df.groupby(['year','period'], as_index=False)['rush_attempts']
    .sum()
    .rename(columns={'rush_attempts': 'total_rush_attempts'})
)

# joining total pass attempts to f_df and sorting by week
f_df = f_df.merge(qb_pa[['period','total_pass_attempts']], on=['period'], how='left')
f_df = f_df.merge(backfield[['period','total_rush_attempts']], on=['period'], how='left')
f_df.sort_values(['year','period'], inplace=True)

# creating target share column
f_df['target_share'] = f_df['targets'] / f_df['total_pass_attempts']
f_df['share_of_backfield'] = f_df['rush_attempts'] / f_df['total_rush_attempts']
f_df['ypa'] = f_df['pass_yds'].astype('int64') / f_df['pass_attempts'].astype('int64')

f_df.to_csv("player_prop_data.csv", index=True)

print(f_df)
