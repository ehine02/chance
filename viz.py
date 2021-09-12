import io

import numpy as np
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
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white',
                  stripe_color='#c2d59d', stripe=True, axis=True, label=True, tick=True)  # optional stripes
    fig, ax = pitch.draw()
    events = events[events['chance'] == 1]
    for _, event in events.iterrows():
        ax.plot(event.location_x, event.location_y)


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
    df = df[np.abs(df[y_value] - df[y_value].mean()) <= (2 * df[y_value].std())]
    x = df[x_value].array
    y = df[y_value].array
    labels = df[label_col].array
    colour_map = {True: 'red', False: 'blue'}

    fig, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x[ix], y[ix], c=colour_map[g], label=g, s=50, marker='2')
    ax.legend(labels=['No Chance', 'Chance'])
    plt.title(f'{x_value} vs. {y_value}')
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.show()


def do_aggregate_plots():
    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'var_pass_length', 'chance')

    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'sum_pass_length', 'chance')

    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'max_pass_speed', 'chance')

    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'var_pass_speed', 'chance')

    attribute_chance_plot(build_aggregates(), 'location_x', 'location_y', 'chance')
