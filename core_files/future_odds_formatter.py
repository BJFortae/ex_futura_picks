import pandas as pd
import pymysql as pymysql
import numpy as np
from datetime import date, datetime, timedelta

# function to format weekly odds csv
today = date.today()

# root file path
root_path = "~/documents/personal/ex_futura_picks/weekly_files/"

def odds_formatter(df: pd.DataFrame) -> pd.DataFrame:
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
                SELECT * FROM pro_teams pt
                
                """

                cursor.execute(sql)

                rows = cursor.fetchall()

    finally:
        connection.close()


    pro_team_df = pd.DataFrame(rows, columns = [
        'pro_team_id',
        'team',
        'division',
        'conference'
        ]
                               )



    # Parse as UTC and convert properly (avoid subtracting 4 hours manually)
    df['commence_time'] = pd.to_datetime(df['commence_time'], utc=True)
    df['commence_time_local'] = df['commence_time'].dt.tz_convert('America/Chicago')

    # Keep as datetime (midnight) instead of .dt.date
    df['commence_date'] = df['commence_time_local'].dt.normalize()

    # Align nfl_start type & timezone
    nfl_start = pd.Timestamp('2025-09-04', tz='America/Chicago')

    # Days and week number (Week 1 starts at season start)
    df['days_since_start'] = (df['commence_date'] - nfl_start).dt.days
    df['week_label'] = (df['days_since_start'] // 7 + 1).astype(int)
    pd.set_option('display.max_columns', None)

    # 1) unique game keys (per bookmaker)
    keys_all = [c for c in ["event_id","commence_date","home_team","away_team","bookmaker", "week_label"] if c in df.columns]
    games = df[keys_all].reset_index(drop=True)

    # 2) explode into team rows
    home = games.assign(team=games["home_team"], opponent=games["away_team"], is_home=1)
    away = games.assign(team=games["away_team"], opponent=games["home_team"], is_home=0)
    base = pd.concat([home, away], ignore_index=True)

    # 3) H2H by team
    if "market" in df.columns:
        h2h = (df.loc[df["market"].eq("h2h"), keys_all + ["outcome","price"]]
                 .rename(columns={"outcome":"team","price":"moneyline"})
                 .drop_duplicates(keys_all + ["team"]))
    else:
        h2h = pd.DataFrame(columns=keys_all + ["team","moneyline"])

    # 4) Spreads by team
    spreads = (df.loc[df["market"].eq("spreads"), keys_all + ["outcome","price","line"]]
                 .rename(columns={"outcome":"team","price":"spread_odds","line":"spread"})
                 .drop_duplicates(keys_all + ["team"]))

    # 5) Totals (one per game)
    totals = (df.loc[df["market"].eq("totals"), keys_all + ["line","outcome","price"]]
                .assign(total_ou=lambda d: d["line"],
                        over_odds=lambda d: np.where(d["outcome"].eq("Over"), d["price"], np.nan),
                        under_odds=lambda d: np.where(d["outcome"].eq("Under"), d["price"], np.nan))
                .groupby(keys_all, as_index=False)
                .agg(total_ou=("total_ou","first"),
                     over_odds=("over_odds","max"),
                     under_odds=("under_odds","max")))

    # 6) Join markets onto scaffold
    out = (base
           .merge(h2h,    on=keys_all + ["team"], how="left")
           .merge(spreads,on=keys_all + ["team"], how="left")
           .merge(totals, on=keys_all,            how="left")).drop_duplicates()

    # 7) Order columns
    cols = [c for c in ["event_id","commence_date","bookmaker","team","is_home","opponent",
                        "home_team","away_team","moneyline","spread","spread_odds",
                        "total_ou","over_odds","under_odds", "week_label"] if c in out.columns]
    df = out[cols]

    # joining pro team ideas, divisions and conferences
    df = df.merge(pro_team_df, how="left", left_on="opponent", right_on="team")
    df = df.rename(columns={'pro_team_id':'opponent_pro_team_id', 'division': 'opponent_division','conference':'opponent_conference', 'team_x':'team'}).drop(columns=['team_y'])
    df = df.merge(pro_team_df, how="left", left_on="team", right_on="team")

    # sorting values by season year and week
    df = df.sort_values(["week_label"])

    #creating weekly props csv
    df.to_csv(f'{root_path}weekly_props_data_{today}_formatted.csv', index=True)

    return df
