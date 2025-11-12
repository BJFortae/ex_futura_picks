import pandas as pd
import pymysql as pymysql
import numpy as np
from datetime import date

today = date.today().strftime("%Y%m%d")
file = f'~/documents/personal/ex_futura_picks/weekly_files/bookmaker_odds_weekly_{today}.csv'


def to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # converting commence time to local date/ time instead of UTC
    df['commence_time'] = pd.to_datetime(df['commence_time'])
    df['commence_time_est'] = df['commence_time'] - pd.Timedelta(hours=4)
    df['commence_date'] = df['commence_time_est'].dt.date

    # adding day of week
    df['commence_day'] = df['commence_time_est'].dt.day_name()

    # Optional: label like "Week 1"
    df["week_label"] = df["week_num"].apply(lambda x: f"Week {int(x):02d}" if pd.notna(x) else pd.NA)
    df.dropna(subset=["week_label"], inplace=True)

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