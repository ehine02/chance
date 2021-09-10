import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Masking, Dropout
from keras.metrics import Precision, Recall, FalsePositives, FalseNegatives
from keras.losses import BinaryCrossentropy, MeanSquaredError

from sb_interface import load_events
from utils import perform_oversampling
from viz import plot_history
from wame_opt import WAME


def build_numeric_sequences(target):
    e = load_events()
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    e = e.loc[e['type'].isin(['shot', 'pass', 'carry', 'dribble'])]
    e = e.loc[e.team == e.possession_team]

    e.xg = e.groupby(by=['match_id', 'possession'])['xg'].transform(lambda x: x.iloc[-1])
    e = e.loc[~e['type'].isin(['shot'])]
    e.pass_height = e.pass_height.str.split().str[0]
    g = e.groupby(by=['match_id', 'possession'])
    sequences = []
    target_chance = []
    target_xg = []
    for ((match_id, possession), events) in g:
        events = events.set_index(events['index'])
        events = events.sort_index()
        events.fillna(value=0.0, inplace=True)

        seq_events = events[['pass_speed', 'pass_length', 'carry_speed', 'carry_length']]
        sequences.append(seq_events.values)
        target_chance.append(events.chance.any())
        target_xg.append(events.xg.iloc[-1])

    seq_df = perform_oversampling(pd.DataFrame({'seqs': sequences,
                                                'chance': target_chance,
                                                'xg': target_xg}))
    print(seq_df.chance.value_counts())
    return seq_df.seqs, seq_df[target], g.index.count().max()


def pad_sequences(sequences, padding_shape, value):
    # padded_shape is (longest, width)
    padded = np.full((len(sequences), padding_shape[0], padding_shape[1]), fill_value=value)
    for index, sequence in enumerate(sequences):
        padded[index, 0:sequence.shape[0], :] = sequence
    return padded


def assemble_model(input_layer, loss_function, metrics):
    model = Sequential()
    model.add(input_layer)
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_function,
                  optimizer=WAME(learning_rate=0.0001),
                  metrics=metrics)
    print(model.summary())
    return model


def classy():
    sequences, targets, longest_sequence = build_numeric_sequences(target='chance')

    padding_shape = (longest_sequence, len(sequences[0][0]))  # (longest, width)
    padding_value = -10.0
    sequences_padded = pad_sequences(sequences, padding_shape, padding_value)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    masking_layer = Masking(mask_value=padding_value, input_shape=padding_shape)
    metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives()]
    model = assemble_model(masking_layer, 'binary_crossentropy', metrics)

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=1024)

    scores = model.evaluate(x_test, y_test, verbose=True)
    print(f'Accuracy: {round(scores[1]*100, 1)}%')

    y_prob = [i[0] for i in model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    plot_history(h, 'accuracy')

    return pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'prob': y_prob})


def chancy():
    sequences, targets, longest_sequence = build_numeric_sequences(target='xg')

    padding_shape = (longest_sequence, len(sequences[0][0]))  # (longest, width)
    padding_value = -10.0
    sequences_padded = pad_sequences(sequences, padding_shape, padding_value)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    masking_layer = Masking(mask_value=padding_value, input_shape=padding_shape)
    model = assemble_model(masking_layer, MeanSquaredError(), [MeanSquaredError()])

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=1024)

    scores = model.evaluate(x_test, y_test, verbose=True)
    print(f'MSE: {round(scores[1])}')

    y_prob = [i[0] for i in model.predict(x_test)]
    print(f'R2 Score: {r2_score(y_test, y_prob)}')

    plot_history(h, 'mean_squared_error')

    return pd.DataFrame({'actual': y_test, 'prob': y_prob})
