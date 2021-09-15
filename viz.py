import io
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mplsoccer import Pitch

from lr_agg import build_aggregates


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
            lt1 = pitch.lines(event.location_x, event.location_y, event.pass_end_x, event.pass_end_y, ax=ax,
                              color="red", comet=True, label=str(event.index))
        if event_type == 'carry':
            lt2 = pitch.lines(event.location_x, event.location_y, event.carry_end_x, event.carry_end_y, ax=ax,
                              color="blue", comet=True, label=str(event.index))


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


def plot_possession(match_id, possession, events=None):
    if events is None:
        events = pd.read_csv('all_events_orig_bak.csv')
    events = events.loc[events.match_id == match_id].loc[events.possession == possession]
    events = events.sort_index()
    print(events.shape)
    plot_events(events)
    return events


def inspect_false_positive(predicts, events, match_pos=None):
    examples = predicts.loc[predicts.predicted != predicts.actual]
    if match_pos is None:
        row = random.randint(0, examples.shape[0])
    example = examples.iloc[row]
    match_id, possession = map(int, example.match_pos.split('_'))
    return plot_possession(match_id, possession, events)
