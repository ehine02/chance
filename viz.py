import numpy as np
from matplotlib import pyplot as plt


def history_plot(history, what):
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