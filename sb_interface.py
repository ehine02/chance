import pandas as pd
import numpy as np

from statsbombpy import sb
from utils import euclidean_distance, names_v2, list_if_not_nan, split_location
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


def load_events(sample_size=50000):
    e = pd.read_csv('all_events_orig_bak.csv', nrows=sample_size)
    e = e.loc[~e['shot_type'].isin(['Penalty'])]
    e = e.loc[~e['location'].isin([np.nan])]

    e['location'] = e.location.apply(list_if_not_nan)
    e['pass_end_location'] = e.pass_end_location.apply(list_if_not_nan)
    e['carry_end_location'] = e.carry_end_location.apply(list_if_not_nan)
    e['location_x'], e['location_y'] = zip(*e.location.map(split_location))
    e['pass_end_x'], e['pass_end_y'] = zip(*e.pass_end_location.map(split_location))
    e['carry_end_x'], e['carry_end_y'] = zip(*e.carry_end_location.map(split_location))

    e.loc[e.type == 'Carry', 'carry_length'] = round(euclidean_distance((e.location_x, e.location_y),
                                                                        (e.carry_end_x, e.carry_end_y)), 0)

    e.loc[e.type == 'Carry', 'carry_speed'] = e.carry_length / e.duration
    e.loc[e.type == 'Pass', 'pass_speed'] = e.pass_length / e.duration

    e['to_goal_start'] = round(euclidean_distance((120, 40), (e.location_x, e.location_y)), 0)
    e.loc[e.type == 'Pass', 'to_goal_end'] = round(euclidean_distance((120, 40), (e.pass_end_x, e.pass_end_y)), 0)
    e.loc[e.type == 'Carry', 'to_goal_end'] = round(euclidean_distance((120, 40), (e.carry_end_x, e.carry_end_y)), 0)

    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100 * (e.to_goal_start - e.to_goal_end)
                                                              / e.to_goal_start, 0)

    e['delta_y'] = e.location_y.diff().abs()
    e['delta_x'] = e.location_x.diff().abs()

    e['chance'] = ~e['shot_type'].isna()

    xg = XgMap()
    e['xg'] = e.apply(func=lambda event: xg.value(event.location_x, event.location_y), axis=1)
    #e.loc[e.type == 'Shot', 'xg'] = e.shot_statsbomb_xg

    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e