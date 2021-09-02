import pandas as pd
import numpy as np

from statsbombpy import sb
from utils import euclidean_distance, names_v2
from xg_utils import XgMap


def get_matches():
    comps = sb.competitions()
    matches = []
    for comp, seas in zip(comps['competition_id'], comps['season_id']):
        if seas in [76]:
            continue
        print(f'Processing competition {comp} and season {seas}'.format())
        [matches.append(match) for match in sb.matches(competition_id=comp, season_id=seas)['match_id']]
    return matches


def store_events(match_ids='ALL', str_file_name='all_events.csv'):
    games = pd.DataFrame()
    xg = XgMap()
    matches = get_matches() if match_ids == 'ALL' else match_ids
    count = 0
    for match in matches:
        raw = sb.events(match_id=match)
        for period in set(raw.period.values):
            events = raw[raw['period'] == period]
            events = events.set_index(pd.DatetimeIndex(events['timestamp']))
            events = events.sort_index()
            try:
                events = events[names_v2()]
                events = events.loc[~events['type'].isin(['Block', 'Goal Keeper', 'Starting XI', 'Half Start',
                                                          'Injury Stoppage', 'Substitution', 'Tactical Shift',
                                                          'Half End', 'Pressure', 'Bad Behaviour', 'Player On',
                                                          'Player Off', 'Camera off', 'Camera On'])]
                events['location_x'] = events['location'].apply(lambda x: x[0] if type(x) == list else np.nan)
                events['location_y'] = events['location'].apply(lambda x: x[1] if type(x) == list else np.nan)
                events['chance'] = events['shot_type']
                events.loc[events['chance'].isna() == False, 'chance'] = 1
                events.loc[events['chance'].isna() == True, 'chance'] = 0

                def calc_carry_length(event):
                    if event.carry_end_location is np.nan:
                        return 0.0
                    return euclidean_distance(event.location, event.carry_end_location)

                def calc_to_goal(event):
                    if event.location_x is np.nan or event.location_y is np.nan:
                        return 0.0
                    return euclidean_distance(event.location, [120, 40])

                def calc_xg(event):
                    return xg.value(event.location_x, event.location_y)

                events['carry_length'] = events.apply(func=calc_carry_length, axis=1)
                events['to_goal'] = events.apply(func=calc_to_goal, axis=1)
                events['xg'] = events.apply(func=calc_xg, axis=1)
                games = pd.concat([games, events])
                count += events.shape[0]
            except:
                print('Error, skipping')
            print(f'Processed {count} events...')
        print(f'Processed match id {match}')
    print(f'Saving data to {str_file_name}')
    games.to_csv(str_file_name)
    return games
