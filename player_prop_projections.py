import argparse
import numpy as np
import pandas as pd
import player_prop_data as ppd
import next_week_function as nwf
import bambi as bmb

# import player prop data
df = ppd.df.copy()
team_week = ppd.team_week.copy()

# =======================
# Example wiring
# =======================
# 1) If starting from player-level rows
# team_week = make_team_week(df_players)

# 2) Add EWMAs

# 3) Train set is all historical rows (exclude last if you want a true out of sample test
# res_nb = fit_team_pass_attempts_nb(team_week_ewma)

# 4) Build df_next_team (you proivde sched_next & vegas_next)
# df_next_team = build_df_next_team(team_week_ewma, sched_next, vegas_next)

# 5) Predict
# team_volume_pred = predict_team_volume(res_nb, df_next_team)
# print(team_volume_pred.head())



# Example features; adjust to your columns
formula = "targets ~ completion_pct + team_pass_pct + ewma_targets + (1|player_id)"
m_tar = bmb.Model(formula, data=df, family="negativebinomial")
idata_tar = m_tar.fit(draws=1000, tune=1000, target_accept=0.9, chains=4, cores=4)

# Predict expected targets for next week (mean on the posterior predictive)
targets_pred = m_tar.predict(df_next, kind="mean")  # mean of posterior predictive



