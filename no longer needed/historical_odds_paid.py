import os, time, requests, datetime as dt
import pandas as pd
from typing import Iterable, Optional

API_KEY = '1d34bdb47e20af76835c5999c4d434af'  # set this in your env!
BASE = "https://api.the-odds-api.com/v4"

def _iso(dt_or_str) -> str:
    if isinstance(dt_or_str, (dt.datetime, dt.date)):
        # ensure UTC ISO8601 with Z suffix
        if isinstance(dt_or_str, dt.date) and not isinstance(dt_or_str, dt.datetime):
            dt_or_str = dt.datetime(dt_or_str.year, dt_or_str.month, dt_or_str.day)
        if dt_or_str.tzinfo is None:
            dt_or_str = dt_or_str.replace(tzinfo=dt.timezone.utc)
        return dt_or_str.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt_or_str  # assume already ISO8601

def iter_historical_snapshots(
    sport: str,
    start: dt.datetime,
    end: Optional[dt.datetime] = None,
    *,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
    date_format: str = "iso",
    throttle_sec: float = 0.2,
) -> Iterable[dict]:
    """
    Yields dict snapshots from the historical endpoint, walking via next_timestamp until > end (if provided).
    Each yield is one snapshot envelope which includes: timestamp, previous_timestamp, next_timestamp, data=[...games...]
    """
    if not API_KEY:
        raise RuntimeError("Set ODDS_API_KEY in your environment.")

    current = _iso(start)
    end_iso = _iso(end) if end else None

    while True:
        params = {
            "apiKey": API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "date": current,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        url = f"{BASE}/historical/sports/{sport}/odds"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        snap = r.json()

        # Expected envelope: {'timestamp': '...', 'previous_timestamp': '...', 'next_timestamp': '...', 'data': [...]}
        # Some libraries show keys slightly differently; be defensive:
        data = snap.get("data")
        ts = snap.get("timestamp")
        nxt = snap.get("next_timestamp")

        if data is None or ts is None:
            # If the API ever returns directly the list (unlikely), normalize:
            if isinstance(snap, list):
                data = snap
                ts = current
                nxt = None
            else:
                raise ValueError(f"Unexpected historical response shape: keys={list(snap.keys())}")

        yield {"timestamp": ts, "next_timestamp": nxt, "data": data, "headers": dict(r.headers)}

        # Stop conditions
        if not nxt:
            break
        if end_iso and nxt > end_iso:
            break

        current = nxt
        time.sleep(throttle_sec)  # be nice to the API / your quota

def flatten_snapshot(snapshot: dict, book_key: Optional[str] = None, wanted=("h2h","spreads","totals")) -> list:
    rows = []
    ts = snapshot["timestamp"]
    for g in snapshot["data"]:
        for bk in g.get("bookmakers", []):
            if book_key and bk.get("key") != book_key:
                continue
            for m in bk.get("markets", []):
                if m.get("key") not in wanted:
                    continue
                for o in m.get("outcomes", []):
                    rows.append({
                        "snapshot_ts": ts,
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
    return rows

def get_historical_dataframe(
    sport: str = "americanfootball_nfl",
    start: dt.datetime = dt.datetime(2025, 9, 4, 0, 0, 0, tzinfo=dt.timezone.utc),
    end: Optional[dt.datetime] = None,
    *,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
    book_key_for_flatten: Optional[str] = None,  # e.g., "draftkings"
) -> pd.DataFrame:
    all_rows = []
    for snap in iter_historical_snapshots(
        sport=sport, start=start, end=end,
        regions=regions, markets=markets, odds_format=odds_format, bookmakers=bookmakers
    ):
        all_rows.extend(flatten_snapshot(snap, book_key=book_key_for_flatten))
    df = pd.DataFrame(all_rows)
    if not df.empty:
        # dedupe identical quotes at the same snapshot
        df = df.drop_duplicates(
            subset=["snapshot_ts","event_id","bookmaker","market","outcome","line","price"]
        ).sort_values(["snapshot_ts","event_id","bookmaker","market","outcome"])
    return df

df = get_historical_dataframe(sport="americanfootball_nfl", start=dt.datetime(2025, 10, 13, 0, 0, 0, tzinfo=dt.timezone.utc), bookmakers="draftkings")
print(df)

# save to CSV
today = date.today()
df = pd.read_csv(f'bookmaker_historical_odds_{today}.csv')