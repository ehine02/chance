import ast
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mplsoccer import Pitch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_xg_events():
    e = pd.read_csv('all_events_orig_bak.csv')
    e = e.loc[~e['shot_type'].isin([np.nan, 'Penalty'])]
    e = e[['location', 'shot_type', 'shot_outcome']]
    e['location'] = e.location.apply(ast.literal_eval)
    e['location_x'] = e['location'].apply(lambda x: round(x[0], 0))
    e['location_y'] = e['location'].apply(lambda x: abs(40 - round(x[1], 0)))
    e['goal'] = e['shot_outcome'] == 'Goal'
    e = e.drop(columns=['location', 'shot_type', 'shot_outcome'])
    return e


def do_xg_regression(x, target):
    y = x.pop(target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    print(f'Trained xG model test accuracy score: {model.score(x_test, y_test)}')
    return model


def xg_model():
    return do_xg_regression(load_xg_events(), target='goal')


def xg_map():
    try:
        return pd.read_csv('xg_dist.csv', skiprows=1, names=np.arange(0, 120).tolist(), skip_blank_lines=True)
    except FileNotFoundError:
        m = xg_model()
        p = pd.DataFrame(np.zeros([80, 120]) * np.nan)
        for y in range(40):
            for x in range(120):
                xg = m.predict_proba([[x, y]])[0][1]
                p.at[40 + y, x] = xg
                p.at[39 - y, x] = xg
        p.to_csv('xg_dist.csv')
        return p


# Xg Map cache helper utility
class XgMap(object):
    def __init__(self):
        self.xg_map = xg_map()

    def value(self, x, y):
        x = min(x, 119.99)
        y = min(y, 79.99)
        return self.xg_map[math.floor(x)][math.floor(y)]

    def plot(self):
        pitch = Pitch(figsize=(16, 8), tight_layout=False, goal_type='box', pitch_color='green', line_color='white')
        fig, ax = pitch.draw()
        plt.pcolor(self.xg_map)
        plt.yticks(np.arange(0.5, len(self.xg_map.index), 1), self.xg_map.index)
        plt.xticks(np.arange(0.5, len(self.xg_map.columns), 1), self.xg_map.columns)
        plt.show()
