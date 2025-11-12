import pandas as pd
import pymysql as pymysql
import numpy as np
from datetime import date

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

print(pro_team_df.head())

# defining today variable
today = date.today()

from datetime import datetime, date, time, timedelta
pd.set_option('display.max_columns', None)
df = pd.read_csv(f'bookmaker_historical_odds_{today}.csv')
df = df[df['bookmaker'] == 'draftkings']

# converting commence time to local date/ time instead of UTC
df['commence_time'] = pd.to_datetime(df['commence_time'])
df['commence_time_est'] = df['commence_time'] - pd.Timedelta(hours=4)
df['commence_date'] = df['commence_time_est'].dt.date

# adding day of week
df['commence_day'] = df['commence_time_est'].dt.day_name()

# create logic to define week of NFL season
df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
df['snapshot_date'] = df['snapshot_ts'].dt.date
df["end_week"] = df["snapshot_date"] + pd.Timedelta(days=6)

# Mask: games that fall within that row’s snapshot window
mask = (df["commence_time"] >= df["snapshot_ts"]) & (df["commence_date"] < df["end_week"])

# Order unique snapshot starts (this defines Week 1, Week 2, ...)
ordered_starts = (
    df.loc[mask, "snapshot_ts"]
      .drop_duplicates()
      .sort_values()
    # now map each start -> 1..N
)

# position 0.. N-1 week
pos = pd.Series(range(len(ordered_starts)), index=ordered_starts)
week_map = ((pos % 22) +1).astype(int)

# season year: start at 2024, advance by 1 every 22 weeks
BASE_SEASON_YEAR = 2024
season_year_map = (pos // 22 + BASE_SEASON_YEAR).astype(str)

# assign to rows (only where the game is within the window)
df.loc[mask, "week_num"]    = df.loc[mask, "snapshot_ts"].map(week_map)
df.loc[mask, "season_year"] = df.loc[mask, "snapshot_ts"].map(season_year_map)

# Optional: label like "Week 1"
df["week_label"] = df["week_num"].apply(lambda x: f"Week {int(x):02d}" if pd.notna(x) else pd.NA)
df.dropna(subset=["week_label"], inplace=True)



import pandas as pd
import numpy as np

def to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) unique game keys (per bookmaker)
    keys_all = [c for c in ["event_id","commence_date","home_team","away_team","bookmaker", "season_year", "week_label"] if c in df.columns]
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
                        "total_ou","over_odds","under_odds", "season_year", "week_label"] if c in out.columns]
    return out[cols]

# ---- usage ----
# df = ...  # your raw odds with columns shown in your sample (+ event_id/commence_time if available)
team_df = to_team_rows(df)

# joining pro team ideas, divisions and conferences
team_df = team_df.merge(pro_team_df, how="left", left_on="opponent", right_on="team")
team_df = team_df.rename(columns={'pro_team_id':'opponent_pro_team_id', 'division': 'opponent_division','conference':'opponent_conference', 'team_x':'team'}).drop(columns=['team_y'])
print(team_df.columns)
team_df = team_df.merge(pro_team_df, how="left", left_on="team", right_on="team")
print(df.head())

# sorting values by season year and week
team_df = team_df.sort_values(["season_year","week_label"])
print(team_df.columns)

team_df.to_csv(f'weekly_props_data_{today}_formatted.csv', index=True)
