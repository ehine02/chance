import ast
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mplsoccer import Pitch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def log_reg(x, target='chance'):
    y = x[target]
    x = x.drop(columns=[target])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    lr_probs = model.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    print(model.score(x_test, y_test))
    return model


def xg_events():
    e = pd.read_csv('all_events_orig.csv')
    e = e.loc[~e['shot_type'].isin([np.nan, 'Penalty'])]
    e = e[['location', 'shot_type', 'shot_outcome']]
    e['location'] = e.location.apply(ast.literal_eval)
    e['location_x'] = e['location'].apply(lambda x: round(x[0], 0))
    e['location_y'] = e['location'].apply(lambda x: abs(40 - round(x[1], 0)))
    e['goal'] = e['shot_outcome'] == 'Goal'
    e = e.drop(columns=['location', 'shot_type', 'shot_outcome'])
    return e


def xg_model():
    return log_reg(xg_events(), 'goal')


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

    def pretty(self):
        pitch = Pitch(figsize=(16, 8), tight_layout=False, goal_type='box', pitch_color='green', line_color='white')
        fig, ax = pitch.draw()
        plt.pcolor(self.xg_map)
        plt.yticks(np.arange(0.5, len(self.xg_map.index), 1), self.xg_map.index)
        plt.xticks(np.arange(0.5, len(self.xg_map.columns), 1), self.xg_map.columns)
        plt.show()

    def pretty2(self):
        pitch = Pitch(line_zorder=2, pitch_color='black')
        fig, ax = pitch.draw()
        x = np.random.uniform(low=0, high=120, size=100)
        y = np.random.uniform(low=0, high=80, size=100)
        stats = pitch.bin_statistic(x, y)
        pitch.heatmap(stats, edgecolors='black', cmap='hot', ax=ax)


def pretty3():
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    from mplsoccer import Pitch, VerticalPitch, FontManager
    from mplsoccer.statsbomb import read_event, EVENT_SLUG

    # get data
    match_files = ['19789.json', '19794.json', '19805.json']
    kwargs = {'related_event_df': False, 'shot_freeze_frame_df': False,
              'tactics_lineup_df': False, 'warn': False}
    df = pd.concat([read_event(f'{EVENT_SLUG}/{file}', **kwargs)['event'] for file in match_files])
    # filter chelsea pressure events
    mask_chelsea_pressure = (df.team_name == 'Chelsea FCW') & (df.type_name == 'Pressure')
    df = df.loc[mask_chelsea_pressure, ['x', 'y']]
    ed = 0