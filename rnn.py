import pandas as pd
import numpy as np
import keras
import ast
import tensorflow as tf
from xg_utils import xg_events


def load_events():
    e = pd.read_csv('all_events.csv')
    e = e.loc[~e['shot_type'].isin(['Penalty'])]
    e = e.loc[~e['location'].isin([np.nan])]
    e = e[['index', 'location', 'duration', 'type', 'shot_type']]
    e['type'] = pd.Categorical(e['type'])
    e['type'] = e.type.cat.codes
    e['location'] = e.location.apply(ast.literal_eval)
    e['location_x'] = e['location'].apply(lambda x: round(x[0], 0))
    e['location_y'] = e['location'].apply(lambda x: round(x[1], 0))
    e['chance'] = ~e['shot_type'].isna()
    e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['shot_type', 'location'])
    return e


def main():
    e = load_events()
    train_size = len(e) * 90 // 100
    target = e.pop('chance')
    dataset = tf.data.Dataset.from_tensor_slices((e.values, target.values))
    n_steps = 30
    window_length = n_steps + 1
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    def batch_window(window):
        window.batch(window_length)
    dataset = dataset.flat_map(batch_window)
    batch_size = 64
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))


