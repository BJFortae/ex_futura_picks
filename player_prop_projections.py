import argparse
import numpy as np
import pandas as pd

def pick(df, choices):
    for name in choices:
        if name in df.columns:
            return name
    return None

def safe_div(num, den):
    if isinstance(den, pd.Series):
        den = den.replace(0, np.nan)
    else:
        den = (np.nan if den == 0 else den)
    return num / den

def recency_weighted_average(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    x = group[value_col]
    w = group[weight_col]
    mask = x.notna() & w.notna()
    if not mask.any():
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))

def build_projections(input_path: str, lambda_decay: float = 0.7, n_sim: int = 1000):
    # Load & normalize columns
    df = pd.read_csv(input_path)
    df.columns = [c.lower() for c in df.columns]

    # Identify likely columns by alias
    col_player = pick(df, ["player_name","player","name"])
    col_player_id = pick(df, ["player_id","id"])
    col_team = pick(df, ["pro_team_id","team","team_abbr"])
    col_pos = pick(df, ["pp.position","position","pos"])
    col_period = pick(df, ["period","wk","week"])
    col_year = pick(df, ["year","season"])

    # Passing
    col_pass_att = pick(df, ["pass_attempts","pass_att","attempts"])
    col_completions = pick(df, ["completions","comp","cmp"])
    col_pass_yds = pick(df, ["pass_yds","passing_yards","pass_yards"])
    col_ints = pick(df, ["ints_thrown","int","ints"])
    col_pass_tds = pick(df, ["pass_tds","passing_tds","pass_td"])
    col_ypa = pick(df, ["ypa"])  # your new field

    # Receiving
    col_targets = pick(df, ["targets"])
    col_receptions = pick(df, ["receptions","rec"])
    col_rec_yds = pick(df, ["receiving_yds","rec_yds","receiving_yards"])
    col_rec_tds = pick(df, ["receiving_tds","rec_tds"])

    # Rushing
    col_rush_att = pick(df, ["rush_attempts","rushing_attempts","carries"])
    col_rush_yds = pick(df, ["rush_yds","rushing_yds"])
    col_rush_tds = pick(df, ["rush_tds","rushing_tds"])

    # Basic checks
    for req in [col_player, col_year, col_period]:
        if req is None:
            raise ValueError("Your sheet needs player/year/period columns (e.g., player_name, year, period).")

    # Week number extraction
    df["week_num"] = pd.to_numeric(df[col_period].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    # Player key
    df["player_key"] = (df[col_player_id] if col_player_id else df[col_player]).astype(str)

    # Filter to the most recent season per player
    latest_year_by_player = df.groupby("player_key")[col_year].transform("max")
    dfr = df[df[col_year] == latest_year_by_player].copy()

    # Recency weights (newer weeks weigh more)
    dfr["week_num"] = dfr["week_num"].fillna(dfr.groupby("player_key")["week_num"].transform("max"))
    max_week_by_player = dfr.groupby("player_key")["week_num"].transform("max")
    dfr["recency_w"] = (lambda_decay ** (max_week_by_player - dfr["week_num"])).astype(float)

    # Efficiency columns
    if col_pass_att and col_pass_yds:
        dfr["comp_rate"] = safe_div(dfr[col_completions], dfr[col_pass_att]) if col_completions else np.nan
        dfr["ypa_col"] = dfr[col_ypa] if col_ypa else safe_div(dfr[col_pass_yds], dfr[col_pass_att])
        dfr["int_rate"] = safe_div(dfr[col_ints], dfr[col_pass_att]) if col_ints else np.nan
        dfr["pass_td_rate"] = safe_div(dfr[col_pass_tds], dfr[col_pass_att]) if col_pass_tds else np.nan

    if col_targets is not None:
        dfr["catch_rate"] = safe_div(dfr[col_receptions], dfr[col_targets]) if col_receptions else np.nan
    if col_receptions and col_rec_yds:
        dfr["ypr"] = safe_div(dfr[col_rec_yds], dfr[col_receptions])

    if col_rush_att and col_rush_yds:
        dfr["ypc"] = safe_div(dfr[col_rush_yds], dfr[col_rush_att])
        dfr["rush_td_rate"] = safe_div(dfr[col_rush_tds], dfr[col_rush_att]) if col_rush_tds else np.nan

    # Columns to aggregate with recency-weighted means
    group_cols = ["player_key", col_player]
    if col_pos: group_cols.append(col_pos)
    if col_team: group_cols.append(col_team)

    agg_targets = {
        col_pass_att: "wm_pass_attempts",
        "comp_rate": "wm_comp_rate",
        "ypa_col": "wm_ypa",
        "int_rate": "wm_int_rate",
        "pass_td_rate": "wm_pass_td_rate",
        col_targets: "wm_targets",
        "catch_rate": "wm_catch_rate",
        "ypr": "wm_ypr",
        col_rush_att: "wm_rush_attempts",
        "ypc": "wm_ypc",
        "rush_td_rate": "wm_rush_td_rate",
        col_rec_yds: "wm_rec_yds",
        col_rush_yds: "wm_rush_yds",
        col_pass_yds: "wm_pass_yds",
        col_receptions: "wm_receptions",
        col_completions: "wm_completions",
        col_ints: "wm_ints",
        col_pass_tds: "wm_pass_tds",
    }
    # Remove Nones
    agg_targets = {k: v for k, v in agg_targets.items() if (k is not None) and (k in dfr.columns)}

    def agg_func(g: pd.DataFrame):
        return pd.Series({out_name: recency_weighted_average(g, in_name, "recency_w")
                          for in_name, out_name in agg_targets.items()})

    wmeans = dfr.groupby(group_cols).apply(agg_func).reset_index()

    # --------------------------------------------------
    # Simple first-cut (deterministic) projections
    # --------------------------------------------------
    proj = wmeans.copy()
    # Passing block
    if "wm_pass_attempts" in proj.columns:
        proj["proj_pass_attempts"] = proj["wm_pass_attempts"]
        if "wm_comp_rate" in proj.columns:
            proj["proj_completions"] = proj["proj_pass_attempts"] * proj["wm_comp_rate"]
        if "wm_ypa" in proj.columns:
            proj["proj_pass_yds"] = proj["proj_pass_attempts"] * proj["wm_ypa"]
        if "wm_int_rate" in proj.columns:
            proj["proj_ints"] = proj["proj_pass_attempts"] * proj["wm_int_rate"]
        if "wm_pass_td_rate" in proj.columns:
            proj["proj_pass_tds"] = proj["proj_pass_attempts"] * proj["wm_pass_td_rate"]

    # Receiving block
    if "wm_targets" in proj.columns:
        proj["proj_targets"] = proj["wm_targets"]
        if "wm_catch_rate" in proj.columns:
            proj["proj_receptions"] = proj["proj_targets"] * proj["wm_catch_rate"]
        if "wm_ypr" in proj.columns and "proj_receptions" in proj.columns:
            proj["proj_receiving_yds"] = proj["proj_receptions"] * proj["wm_ypr"]

    # Rushing block
    if "wm_rush_attempts" in proj.columns:
        proj["proj_rush_attempts"] = proj["wm_rush_attempts"]
        if "wm_ypc" in proj.columns:
            proj["proj_rush_yds"] = proj["proj_rush_attempts"] * proj["wm_ypc"]
        if "wm_rush_td_rate" in proj.columns:
            proj["proj_rush_tds"] = proj["proj_rush_attempts"] * proj["wm_rush_td_rate"]

    # Save first cut (legacy/reference)
    cols_out = [col for col in [
        col_player, col_pos, col_team,
        "proj_pass_attempts","proj_completions","proj_pass_yds","proj_ints","proj_pass_tds",
        "proj_targets","proj_receptions","proj_receiving_yds",
        "proj_rush_attempts","proj_rush_yds","proj_rush_tds"
    ] if (col in proj.columns) or (col in [col_player, col_pos, col_team])]
    first_cut = proj[cols_out].copy()
    first_cut = first_cut.round({c:2 for c in first_cut.columns if c not in [col_player, col_pos, col_team]})
    first_cut.to_csv("first_cut_projections.csv", index=False)

    # --------------------------------------------------
    # Passing Yards uncertainty using your YPA (log-normal) + Attempts (normal)
    # --------------------------------------------------
    # League priors if player-level sd isn't reliable
    if col_pass_att and (col_pass_att in dfr.columns):
        league_att_sd = float(dfr[col_pass_att].dropna().std()) if dfr[col_pass_att].dropna().size else 6.0
        if not np.isfinite(league_att_sd) or league_att_sd <= 0:
            league_att_sd = 6.0
    else:
        league_att_sd = 6.0

    if "ypa_col" in dfr.columns:
        log_ypa = np.log(dfr["ypa_col"].replace({0: np.nan})).dropna()
        league_ypa_log_sd = float(log_ypa.std()) if log_ypa.size else 0.15
        if not np.isfinite(league_ypa_log_sd) or league_ypa_log_sd <= 0:
            league_ypa_log_sd = 0.15
    else:
        league_ypa_log_sd = 0.15

    rng = np.random.default_rng(42)
    sim_rows = []

    for _, row in wmeans.iterrows():
        player_key = row["player_key"]
        player = row[col_player]
        pos = row.get(col_pos, "")
        team = row.get(col_team, "")

        att_mu = row.get("wm_pass_attempts", np.nan)
        ypa_mu = row.get("wm_ypa", np.nan)

        if pd.notna(att_mu) and pd.notna(ypa_mu) and att_mu > 0 and ypa_mu > 0:
            g = dfr[dfr["player_key"] == player_key]

            # attempts sd
            if col_pass_att and (col_pass_att in g.columns):
                att_sd = float(g[col_pass_att].std()) if g[col_pass_att].dropna().size else np.nan
            else:
                att_sd = np.nan
            if not np.isfinite(att_sd) or att_sd <= 0:
                att_sd = league_att_sd

            # ypa log sd
            if "ypa_col" in g.columns:
                log_vals = np.log(g["ypa_col"].replace({0: np.nan})).dropna()
                ypa_log_sd = float(log_vals.std()) if log_vals.size >= 2 else np.nan
            else:
                ypa_log_sd = np.nan
            if not np.isfinite(ypa_log_sd) or ypa_log_sd <= 0:
                ypa_log_sd = league_ypa_log_sd

            log_mean = np.log(ypa_mu) - 0.5 * (ypa_log_sd ** 2)

            att_draws = rng.normal(loc=att_mu, scale=att_sd, size=n_sim)
            att_draws = np.clip(att_draws, 0, None)
            ypa_draws = rng.lognormal(mean=log_mean, sigma=ypa_log_sd, size=n_sim)
            pass_yards_draws = att_draws * ypa_draws

            py_mean = float(np.mean(pass_yards_draws))
            p20, p50, p80, p90 = np.percentile(pass_yards_draws, [20,50,80,90]).tolist()
        else:
            py_mean = p20 = p50 = p80 = p90 = np.nan

        sim_rows.append({
            col_player: player,
            "position": pos,
            "team": team,
            "proj_pass_yards_mean": py_mean,
            "proj_pass_yards_p20": p20,
            "proj_pass_yards_p50": p50,
            "proj_pass_yards_p80": p80,
            "proj_pass_yards_p90": p90
        })

    sim_df = pd.DataFrame(sim_rows)

    # Merge with wmeans to create final output
    out = wmeans.merge(sim_df, on=col_player, how="left")

    # Add deterministic projections (reuse from wmeans-based block)
    out["proj_pass_attempts"] = out.get("wm_pass_attempts")
    out["proj_completion_rate"] = out.get("wm_comp_rate")
    out["proj_completions"] = out["proj_pass_attempts"] * out["proj_completion_rate"]
    out["proj_ints"] = out["proj_pass_attempts"] * out.get("wm_int_rate")
    out["proj_pass_tds"] = out["proj_pass_attempts"] * out.get("wm_pass_td_rate")

    out["proj_targets"] = out.get("wm_targets")
    out["proj_catch_rate"] = out.get("wm_catch_rate")
    out["proj_receptions"] = out["proj_targets"] * out["proj_catch_rate"]
    out["proj_receiving_yds"] = out["proj_receptions"] * out.get("wm_ypr")

    out["proj_rush_attempts"] = out.get("wm_rush_attempts")
    out["proj_rush_yds"] = out["proj_rush_attempts"] * out.get("wm_ypc")
    out["proj_rush_tds"] = out["proj_rush_attempts"] * out.get("wm_rush_td_rate")

    keep_cols = [c for c in [
        col_player, col_pos, col_team,
        "proj_pass_attempts","proj_completions",
        "proj_pass_yards_mean","proj_pass_yards_p20","proj_pass_yards_p50","proj_pass_yards_p80","proj_pass_yards_p90",
        "proj_ints","proj_pass_tds",
        "proj_targets","proj_receptions","proj_receiving_yds",
        "proj_rush_attempts","proj_rush_yds","proj_rush_tds"
    ] if (c in out.columns)]
    proj_out = out[keep_cols].copy()

    round_map = {c:2 for c in proj_out.columns if c not in [col_player, col_pos, col_team]}
    proj_out = proj_out.round(round_map)
    proj_out.to_csv("projections_with_ypa_and_uncertainty.csv", index=False)

    return "first_cut_projections.csv", "projections_with_ypa_and_uncertainty.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to fellas_league_player_projections.csv")
    ap.add_argument("--lambda_decay", type=float, default=0.7, help="Exponential recency decay per week (0-1)")
    ap.add_argument("--sims", type=int, default=1000, help="Number of simulations for passing yards uncertainty")
    args = ap.parse_args()

    first_cut, with_unc = build_projections(args.input, args.lambda_decay, args.sims)
    print("Saved:", first_cut)
    print("Saved:", with_unc)

if __name__ == "__main__":
    main()
