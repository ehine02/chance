import numpy as np
from matplotlib import pyplot as plt
from mplsoccer import Pitch


def plot_history(history, what):
    x = history.history[what]
    val_x = history.history['val_' + what]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
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