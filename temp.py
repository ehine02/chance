import datetime
import pandas as pd
import numpy as np
from statsbombpy import sb


def main():
    match = 18245
    raw = sb.events(match_id=match)
    game = pd.DataFrame()
    for period in set(raw.period.values):
        events = raw[raw['period'] == period]
        events = events.join(pd.json_normalize(events['pass']), rsuffix='_pass')
        events = events.join(pd.json_normalize(events['shot']), rsuffix='_shot')
        events = events.set_index(pd.DatetimeIndex(events['timestamp']))
        events = events.sort_index()
        events = events[['timestamp', 'carry', 'counterpress', 'dribble', 'duration', 'index', 'location', 'pass',
                         'period', 'play_pattern', 'player', 'possession', 'possession_team', 'shot', 'team', 'type',
                         'under_pressure', 'length', 'angle', 'end_location', 'recipient.name', 'height.id',
                         'height.name',
                         'type.id', 'type.name', 'body_part.id', 'body_part.name', 'switch', 'through_ball',
                         'shot_assist']]
                         #'inswinging', 'cut_back', 'miscommunication']]
        events['chance'] = events['shot']
        events.loc[events['chance'].isna() == False, 'chance'] = True

        def count_unique(x):
            return len(set(x))

        def changed(x):
            return count_unique(x) > 1

        def ends_in_chance(x):
            return bool(x[-1]) is True

        frames = events.rolling('30S').agg({'length': np.var,
                                            'length': np.sum,
                                            'duration': np.sum,
                                            'angle': np.max,
                                            'chance': ends_in_chance,
                                            'index': np.count_nonzero,
                                            'possession': count_unique})

        frames['speed'] = frames['length'] / frames['duration']
        game = pd.concat([game, frames])
    game.to_csv(str(match) + '.csv')
    return game
