import os, requests
import pandas as pd
from datetime import datetime, date

today = date.today()

API_KEY = '063c3e01050dac6fe698c6676fb5abcf'
BASE = "https://api.the-odds-api.com/v4"

def get_sports(all=False):
    r = requests.get(f"{BASE}/sports", params={"apiKey": API_KEY, "all": str(all).lower()})
    r.raise_for_status()
    return r.json(), r.headers  # headers include x-requests-*

def get_odds(
    sport="americanfootball_nfl",
    regions="us",
    markets="h2h,spreads,totals",
    odds_format="american",
    bookmakers=None,
    commence_from=None,
    commence_to=None,
    event_ids=None,
    date_format="iso",
):
    params = {
        "apiKey": API_KEY,
        "regions": regions,             # e.g. us, uk, au (comma-sep allowed)
        "markets": markets,             # h2h, spreads, totals, outrights, etc.
        "oddsFormat": odds_format,      # american | decimal
        "dateFormat": date_format,      # iso | unix
    }
    if bookmakers:      params["bookmakers"] = bookmakers          # e.g. "draftkings,fanduel,betmgm"
    if commence_from:   params["commenceTimeFrom"] = commence_from # ISO8601
    if commence_to:     params["commenceTimeTo"]   = commence_to
    if event_ids:       params["eventIds"] = event_ids             # comma-sep ids

    r = requests.get(f"{BASE}/sports/{sport}/odds", params=params, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers  # x-requests-remaining / used / last

# --- usage ---
sports, hdr = get_sports()                    # list of in-season sports
odds, hdr = get_odds(                         # NFL moneyline + spreads + totals, US books
    sport="americanfootball_nfl",
    regions="us",
    markets="h2h,spreads,totals",
    bookmakers="draftkings,fanduel"
)

def flatten_odds(odds_json, book_key: list = None, wanted=("h2h","spreads","totals")) -> list:
    rows = []
    for g in odds_json:
        for bk in g.get("bookmakers", []):
            if book_key and bk.get("key") != book_key:
                continue
            for m in bk.get("markets", []):
                if m.get("key") not in wanted:
                    continue
                for o in m.get("outcomes", []):
                    rows.append({
                        "event_id": g.get("id"),
                        "commence_time": g.get("commence_time"),
                        "home_team": g.get("home_team"),
                        "away_team": g.get("away_team"),
                        "bookmaker": bk.get("key"),
                        "market": m.get("key"),            # h2h | spreads | totals | ...
                        "outcome": o.get("name"),          # team or Over/Under
                        "price": o.get("price"),
                        "line": o.get("point"),            # None for h2h
                    })
    print(f"Rows: {len(rows)}  | Requests remaining: {hdr.get('x-requests-remaining')}")
    df = pd.DataFrame(rows,
                      columns=["event_id", "commence_time", "home_team", "away_team", "bookmaker", "market", "outcome",
                               "price", "line"])
    df.to_csv(f'~/documents/personal/ex_futura_picks/weekly_files/bookmaker_odds_weekly_{today}.csv', index=True)
    return df


df = flatten_odds(odds, book_key="draftkings")

print(df)