import argparse
import numpy as np
import pandas as pd
import player_prop_data as ppd

df = ppd.f_df.copy

import pandas as pd

def team_stats(df):
    stats_list = []

    # group by team and week
    grouped = df.groupby(['team', 'week'])

    for (team, week), group in grouped:
        # total pass attempts by the team that week
        total_pass_attempts = group['pass_attempts'].sum()

        # loop through players on that team/week
        for player_id, player_data in group.groupby('player_id'):
            player_receptions = player_data['receptions'].sum()

            # avoid divide by zero
            player_target_pct = (
                player_receptions / total_pass_attempts
                if total_pass_attempts > 0
                else 0
            )

            stats_list.append({
                'team': team,
                'week': week,
                'player_id': player_id,
                'player_target_pct': player_target_pct
            })

    stats_df = pd.DataFrame(stats_list)
    return stats_df

team_stats_df = team_stats(df)

print(team_stats_df)

