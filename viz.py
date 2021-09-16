import io
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mplsoccer import Pitch

from lr_agg import build_aggregates
from utils import list_if_not_nan, split_location


def plot_history(history, what):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1
    plt.plot(epochs, x, 'b', label="Training " + what)
    plt.plot(epochs, val_x, 'g', label="Validation " + what)
    plt.grid()
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()


def plot_events(events):
    pitch = Pitch(figsize=(32, 16), tight_layout=False, goal_type='box', pitch_color='green', line_color='white')
    fig, ax = pitch.draw()
    for _, event in events.iterrows():
        event_type = str(event['type']).lower()
        if event_type == 'pass':
            lt1 = pitch.arrows(event.location_x, event.location_y, event.pass_end_x, event.pass_end_y, ax=ax,
                              color="red", width=2, headwidth=3, alpha=event.pass_speed_alpha)
        if event_type == 'carry':
            lt2 = pitch.arrows(event.location_x, event.location_y, event.carry_end_x, event.carry_end_y, ax=ax,
                              color="blue", width=2, headwidth=3, label=event.carry_speed_alpha)


def save_embeddings(weights, vocab):
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


def attribute_chance_plot(df, x_value, y_value, label_col):
    df = df[np.abs(df[x_value] - df[x_value].mean()) <= (1.5 * df[x_value].std())]
    df = df[np.abs(df[y_value] - df[y_value].mean()) <= (1.5 * df[y_value].std())]
    x = df[x_value].array
    y = df[y_value].array
    labels = df[label_col].array
    colour_map = {True: 'red', False: 'blue'}

    fig, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x[ix], y[ix], c=colour_map[g], label=g, s=75, marker='2')
    ax.legend(labels=['No Chance', 'Chance'])
    plt.title(f'{x_value} vs. {y_value}')
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.show()


def do_interesting_plots():
    aggs = build_aggregates(sample_size=20000, read_rows=None)
    attribute_chance_plot(aggs, 'location_x', 'location_y', 'chance')
    attribute_chance_plot(aggs, 'sum_progression_pct', 'avg_pass_length', 'chance')
    attribute_chance_plot(aggs, 'sum_progression_pct', 'sum_delta_y', 'chance')


from sklearn import preprocessing



def plot_possession(match_id, possession, events=None):
    e = events
    if e is None:
        e = pd.read_csv('raw_events.csv')
    e = e.loc[e.match_id == match_id].loc[e.possession == possession]
    e = e.sort_index()
    e['location'] = e.location.apply(list_if_not_nan)
    e['pass_end_location'] = e.pass_end_location.apply(list_if_not_nan)
    e['carry_end_location'] = e.carry_end_location.apply(list_if_not_nan)
    e['location_x'], e['location_y'] = zip(*e.location.map(split_location))
    e['pass_end_x'], e['pass_end_y'] = zip(*e.pass_end_location.map(split_location))
    e['carry_end_x'], e['carry_end_y'] = zip(*e.carry_end_location.map(split_location))
    e.loc[e.possession_team != e.team, 'location_x'] = 120 - e.location_x
    e.loc[e.possession_team != e.team, 'location_y'] = 80 - e.location_y
    e.loc[e.possession_team != e.team, 'pass_end_x'] = 120 - e.pass_end_x
    e.loc[e.possession_team != e.team, 'pass_end_y'] = 80 - e.pass_end_y
    e.loc[e.possession_team != e.team, 'carry_end_x'] = 120 - e.carry_end_x
    e.loc[e.possession_team != e.team, 'carry_end_y'] = 80 - e.carry_end_y
    e.loc[e.type == 'Carry', 'carry_speed'] = e.carry_length / e.duration
    e.loc[e.type == 'Pass', 'pass_speed'] = e.pass_length / e.duration
    e['pass_speed_alpha'] = (e.pass_speed - e.pass_speed.min())/(e.pass_speed.max()-e.pass_speed.min())
    e['carry_speed_alpha'] = (e.carry_speed - e.carry_speed.min())/(e.carry_speed.max()-e.carry_speed.min())

    print(e.shape)
    plot_events(e)
    return e


def inspect_false_positive(predicts, events=None, match_pos=None):
    examples = predicts.loc[predicts.predicted == True]
    examples = examples.loc[examples.actual == False]
    if match_pos is None:
        row = random.randint(0, examples.shape[0])
    example = examples.iloc[row]
    match_id, possession = map(int, example.match_pos.split('_'))
    return plot_possession(match_id, possession, events)


def inspect_positive_positive(predicts, events=None, match_pos=None):
    examples = predicts.loc[predicts.predicted == True]
    examples = examples.loc[examples.actual == True]
    if match_pos is None:
        row = random.randint(0, examples.shape[0])
    example = examples.iloc[row]
    match_id, possession = map(int, example.match_pos.split('_'))
    return plot_possession(match_id, possession, events)
