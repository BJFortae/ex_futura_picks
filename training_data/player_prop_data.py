import math
import pandas as pd
from pandas import DataFrame
import pymysql

# setting today variable
today = pd.Timestamp.today()

# formatting settings
pd.set_option("display.max_columns", None)
pd.set_option('future.no_silent_downcasting', True)

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

# keep only numeric columns
corr_df = df.select_dtypes(include='number').corr()

# correlations with pass_attempts
corr_pass = corr_df['pass_attempts']

# subset to the variables you care about
cols = [
    'rush_attempts','yds_perCarry','rush_yds','rush_tds',
    'rush_yds_perGame','rush_firstDowns','targets','receptions','yds_perRec',
    'receiving_yds','receiving_tds','rec_yds_perGame','receiving_firstDowns',
    'completions','completion_pct','pass_yds','pass_tds','ints_thrown','sacks',
    'avg_yds_allowed_perPlay','offensive_yds_allowed','tds_allowed','pts_against',
    'def_ints','games_played'
]

corr_test = corr_pass.loc[corr_pass.index.intersection(cols)]
# print(corr_test.sort_values(ascending=False))

# df.to_csv(f"player_df_{today}.csv", index=True)
