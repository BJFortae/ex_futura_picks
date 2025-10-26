import os, requests
import pandas as pd

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

# example: flatten DraftKings H2H prices for each game
def extract_moneyline(odds_json, book_key="draftkings"):
    rows = []
    for g in odds_json:
        bk = next((b for b in g.get("bookmakers", []) if b["key"] == book_key), None)
        if not bk:
            continue
        m = next((m for m in bk.get("markets", [])), None)
        if not m:
            continue
        for o in m["outcomes"]:
            if m["key"] == "h2h":
                rows.append({
                    "event_id": g["id"],
                    "commence_time": g["commence_time"],
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "bookmaker": bk["key"],
                    "team": o["name"],
                    "price": o["price"]
            })
            if m["key"] == "spread":
                rows.append({
                    "event_id": g["id"],
                    "commence_time": g["commence_time"],
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "bookmaker": bk["key"],
                    "team": o["name"],
                    "spread": o["point"],
                    "price" : o["[price"]
                })
    return rows

rows = extract_moneyline(odds)

df = pd.DataFrame(rows, columns=["event_id","commence_time","home_team","away_team","bookmaker","team","price", "spread"])

print(rows)
print(f"Rows: {len(rows)}  | Requests remaining: {hdr.get('x-requests-remaining')}")
print(df)
# df.to_csv('sports_book_odds.csv', index=True)
