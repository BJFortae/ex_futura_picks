import os, time, requests, datetime as dt
import pandas as pd
from typing import Iterable, Optional

API_KEY = os.getenv("ODDS_API_KEY") or "1d34bdb47e20af76835c5999c4d434af"
BASE = "https://api.the-odds-api.com/v4"

def _iso(dt_or_str) -> str:
    if isinstance(dt_or_str, (dt.datetime, dt.date)):
        if isinstance(dt_or_str, dt.date) and not isinstance(dt_or_str, dt.datetime):
            dt_or_str = dt.datetime(dt_or_str.year, dt_or_str.month, dt_or_str.day, tzinfo=dt.timezone.utc)
        if dt_or_str.tzinfo is None:
            dt_or_str = dt_or_str.replace(tzinfo=dt.timezone.utc)
        return dt_or_str.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt_or_str

def _first_weekday_on_or_after(d: dt.datetime, weekday: int) -> dt.datetime:
    # Monday=0 ... Sunday=6
    delta_days = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=delta_days)

def iter_weekly_historical_snapshots(
    sport: str,
    start: dt.datetime,
    end: Optional[dt.datetime] = None,
    *,
    weekday: int = 2,              # 2 = Wednesday
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
    date_format: str = "iso",
) -> Iterable[dict]:
    """
    Yields exactly one historical snapshot per week on the chosen weekday (UTC).
    """
    if not API_KEY:
        raise RuntimeError("Set ODDS_API_KEY in your environment.")

    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end is None:
        end = dt.datetime.now(dt.timezone.utc)
    elif end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)

    current = _first_weekday_on_or_after(start, weekday).replace(hour=0, minute=0, second=0, microsecond=0)

    while current <= end:
        params = {
            "apiKey": API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "date": _iso(current),
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        url = f"{BASE}/historical/sports/{sport}/odds"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        snap = r.json()

        # Normalize expected shape
        if isinstance(snap, dict) and "data" in snap and "timestamp" in snap:
            yield {"timestamp": snap["timestamp"], "data": snap["data"], "headers": dict(r.headers)}
        elif isinstance(snap, list):
            yield {"timestamp": _iso(current), "data": snap, "headers": dict(r.headers)}
        else:
            raise ValueError(f"Unexpected historical response shape: keys={list(snap.keys()) if isinstance(snap, dict) else type(snap)}")

        current += dt.timedelta(days=7)

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
                        "market": m.get("key"),
                        "outcome": o.get("name"),
                        "price": o.get("price"),
                        "line": o.get("point"),
                    })
    return rows

def get_historical_dataframe(
    sport: str = "americanfootball_nfl",
    start: dt.datetime = dt.datetime(2025, 10, 13, 0, 0, 0, tzinfo=dt.timezone.utc),  # your start
    end: Optional[dt.datetime] = None,
    *,
    weekday: int = 2,  # Wednesday
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    bookmakers: Optional[str] = None,
    book_key_for_flatten: Optional[str] = None,
) -> pd.DataFrame:
    all_rows = []
    for snap in iter_weekly_historical_snapshots(
        sport=sport, start=start, end=end, weekday=weekday,
        regions=regions, markets=markets, odds_format=odds_format, bookmakers=bookmakers
    ):
        all_rows.extend(flatten_snapshot(snap, book_key=book_key_for_flatten))
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["snapshot_ts","event_id","bookmaker","market","outcome","line","price"]
        ).sort_values(["snapshot_ts","event_id","bookmaker","market","outcome"])
    return df

# --- Run weekly pulls from 10/13/2025, but on Wednesdays (first = 10/15/2025 UTC) ---
df = get_historical_dataframe(
    sport="americanfootball_nfl",
    start=dt.datetime(2025, 10, 13, 0, 0, 0, tzinfo=dt.timezone.utc),
    bookmakers="draftkings",
    weekday=2  # Wednesday
)
print(df)

# Save to CSV (fix)
today_str = dt.date.today().strftime("%Y-%m-%d")
out_path = f"bookmaker_historical_odds_{today_str}.csv"
df.to_csv(out_path, index=False)
print("Saved:", out_path)
