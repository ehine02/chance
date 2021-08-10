import datetime
import pandas as pd
import numpy as np
from statsbombpy import sb
from attributes import names_v2
from mplsoccer import Pitch
from xg_utils import xg_model
import tensorflow as tf


def euclidean_distance(start, end):
    return np.sqrt(np.power(end[0] - start[0], 2) + np.power(end[1] - start[1], 2))


def get_matches():
    comps = sb.competitions()
    matches = []
    for comp, seas in zip(comps['competition_id'], comps['season_id']):
        if seas in [76]:
            continue
        print(f'Processing competition {comp} and season {seas}'.format())
        [matches.append(match) for match in sb.matches(competition_id=comp, season_id=seas)['match_id']]
    return matches


def csv_events(match_id):
    sb.events(match_id=match_id).to_csv(str(match_id) + '_raw.csv')


def plot_events(events):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white',
                  stripe_color='#c2d59d', stripe=True, axis=True, label=True, tick=True)  # optional stripes
    fig, ax = pitch.draw()
    events = events[events['chance'] == 1]
    for _, event in events.iterrows():
        ax.plot(event.location_x, event.location_y)


def main(match_ids='ALL'):
    games = pd.DataFrame()
    xg = xg_model()
    matches = get_matches() if match_ids == 'ALL' else match_ids
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
                                                          'Half End', 'Pressure'])]
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
                    return xg.predict_proba([[event.location_x, event.location_y]])[0][1]

                events['carry_length'] = events.apply(func=calc_carry_length, axis=1)
                events['to_goal'] = events.apply(func=calc_to_goal, axis=1)
                events['xg'] = events.apply(func=calc_xg, axis=1)
                # frames = label_frames(events)
                games = pd.concat([games, events])
            except:
                print('Error, skipping')
    # games.to_csv('games.csv')
    return games


def aggregate_events(events):
    def count_unique(x):
        return len(set(x))

    def changed(x):
        return count_unique(x) > 1

    def ends_in_chance(x):
        return x[-1] == 1

    frames = events.rolling('20S').agg({  # 'pass_length': np.var,
        'period': lambda x: x[-1],
        'pass_length': np.sum,
        'location_x': np.var,
        'location_y': np.var,
        'duration': np.sum,
        'chance': ends_in_chance,
        'index': lambda x: len(x),
        'carry_length': np.sum,
        'possession': count_unique
        # 'pass_angle': np.max,
        # 'index': np.count_nonzero,

    })

    frames['speed'] = (frames['pass_length'] + frames['carry_length']) / frames['duration']
    frames.replace([np.inf, -np.inf], np.nan, inplace=True)
    frames = frames.fillna(0.0)
    return frames


def label_frames(events):
    frames = []
    events = events[events['chance'] == 1]
    events = events[['to_goal', 'duration', 'chance']]
    for frame in events.rolling('30S'):
        if frame['chance'][-1] == 1:
            ed = 0
        frame['chance_seq'] = frame['chance'][-1] == 1
        # pd.concat([frames, frame])
        frames.append(frame)
    return frames









