from time import sleep

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt     # Plotting data
from mplsoccer.pitch import Pitch
import io


def list_if_not_nan(x):
    if x is np.nan:
        return np.nan
    else:
        return ast.literal_eval(x)


def split_location(x):
    if x is not np.nan:
        return round(x[0], 0), round(x[1], 0)
    return np.nan, np.nan


def load_events():
    e = pd.read_csv('all_events_orig.csv', nrows=100000)
    e = e.loc[~e['shot_type'].isin(['Penalty'])]
    e = e.loc[~e['location'].isin([np.nan])]
    e['location'] = e.location.apply(list_if_not_nan)
    e['pass_end_location'] = e.pass_end_location.apply(list_if_not_nan)
    e['carry_end_location'] = e.carry_end_location.apply(list_if_not_nan)
    e['location_x'], e['location_y'] = zip(*e.location.map(split_location))
    e['pass_end_x'], e['pass_end_y'] = zip(*e.pass_end_location.map(split_location))
    e['carry_end_x'], e['carry_end_y'] = zip(*e.carry_end_location.map(split_location))
    e.loc[e.type == 'Carry', 'carry_length'] = \
        np.sqrt((e.carry_end_x - e.location_x) ** 2 + (e.carry_end_y - e.location_y) ** 2)
    e.loc[e.type == 'Carry', 'carry_speed'] = e.carry_length / e.duration
    e.loc[e.type == 'Pass', 'pass_speed'] = e.pass_length / e.duration
    e.loc[e.location != np.nan, 'to_goal_start'] = \
        round(np.sqrt((120 - e.location_x) ** 2 + (40 - e.location_y) ** 2), 0)
    e.loc[e.type == 'Pass', 'to_goal_end'] = \
        round(np.sqrt((120 - e.pass_end_x) ** 2 + (40 - e.pass_end_y) ** 2), 0)
    e.loc[e.type == 'Carry', 'to_goal_end'] = \
        round(np.sqrt((120 - e.carry_end_x) ** 2 + (40 - e.carry_end_y) ** 2), 0)
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100*(e.to_goal_start - e.to_goal_end) / e.to_goal_start, 0)
    e['chance'] = ~e['shot_type'].isna()
    # e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def draw_pitch():
    pitch = Pitch(figsize=(16, 8), tight_layout=False, goal_type='box', pitch_color='green', line_color='white')
    fig, ax = pitch.draw()
    return pitch


import cv2


def main():
    e = load_events()
    e = e.loc[e['type'].isin(['Pass', 'Carry'])]
    e = e.loc[e['possession_team'] == e['team']]
    e.pass_height = e.pass_height.str.split().str[0]
    g = e.groupby(by=['match_id', 'possession'])
    count = 0
    pitch = Pitch(figsize=(16, 8), tight_layout=False, goal_type='box', pitch_color='green', line_color='white')
    e.loc[e.pass_speed != np.nan, 'pass_speed_alpha'] = pd.cut(e.pass_speed, bins=[-1, 5, 10, 20, 60, 150],
                                                              labels=[0.1, 0.3, 0.5, 0.7, 1])
    e.loc[e.carry_speed != np.nan, 'carry_speed_alpha'] = pd.cut(e.carry_speed, bins=[-1, 2, 4, 6, 10, 1500],
                                                              labels=[0.1, 0.3, 0.5, 0.7, 1])
    for ((match_id, possession), events) in g:
        events = events.set_index(events['index'])
        events = events.sort_index()
        chance = False
        fig, ax = pitch.draw()
        for _, row in events.iterrows():
            chance = chance or row.chance
            event_type = str(row['type']).lower()
            if event_type == 'pass' and row.pass_speed_alpha != np.nan:
                lt1 = pitch.lines(row.location_x, row.location_y, row.pass_end_x, row.pass_end_y, ax=ax,
                                  alpha=row.pass_speed_alpha, color="red", comet=True, label=str(row.index))
            if event_type == 'carry' and row.carry_speed_alpha != np.nan:
                lt2 = pitch.lines(row.location_x, row.location_y, row.carry_end_x, row.carry_end_y, ax=ax,
                                  alpha=row.carry_speed_alpha, color="blue", comet=True, label=str(row.index))

        plt.savefig('./poss_maps/' + str(match_id) + '_' + str(possession) + '_' + str(int(chance)) + '.png' )
        plt.clf()
        plt.close()
        fig.clf()
        count += 1
        if count % 100 == 0:
            sleep(1)
            print(count)
            #break
    return e

def fig2rgb_array(fig):
    fig.canvas.draw()
    arr = np.array(fig.canvas.renderer.buffer_rgba())
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
