import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Masking
from tensorflow.python.keras.layers import Dropout

from utils import list_if_not_nan, split_location, perform_oversampling
from viz import plot_history
from wame_opt import WAME
from xg_utils import XgMap


def load_events():
    e = pd.read_csv('all_events.csv', nrows=100000)
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
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100 * (e.to_goal_start - e.to_goal_end) /
                                                              e.to_goal_start, 0)
    e['delta_y'] = e.location_y.diff().abs()
    e['delta_x'] = e.location_x.diff().abs()

    e['chance'] = ~e['shot_type'].isna()
    xg = XgMap()

    def calc_xg(event):
        return xg.value(event.location_x, event.location_y)

    e['xg'] = e.apply(func=calc_xg, axis=1)
    e.loc[e.type == 'Shot', 'xg'] = e.shot_statsbomb_xg
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def build_numeric_sequences(target):
    # Parameters
    dimensions = ['pass_speed', 'pass_length', 'carry_speed', 'carry_length', 'delta_y']

    e = load_events()
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')

    # e = e.loc[~e['type'].isin(['shot', 'block', 'goal keeper', 'pressure', 'clearance'])]
    e = e.loc[e['type'].isin(['shot', 'pass', 'carry', 'dribble', 'dribbled past'])]

    e.xg = e.groupby(by=['match_id', 'possession'])['xg'].transform(lambda x: x.iloc[-1])
    e = e.loc[~e['type'].isin(['shot'])]
    e.pass_height = e.pass_height.str.split().str[0]
    g = e.groupby(by=['match_id', 'possession'])
    sequences = []
    target_chance = []
    target_xg = []
    longest_seq = 0
    for ((match_id, possession), events) in g:
        if len(events.index) > longest_seq:
            longest_seq = len(events.index)
        events = events.set_index(events['index'])
        events = events.sort_index()
        events.fillna(value=0.0, inplace=True)

        seq_events = events[dimensions]
        sequences.append(seq_events.values)
        target_chance.append(events.chance.any())
        target_xg.append(events.xg.iloc[-1])

    seq_df = perform_oversampling(pd.DataFrame({'seqs': sequences,
                                                'chance': target_chance,
                                                'xg': target_xg}))
    print(seq_df.chance.value_counts())
    return seq_df.seqs, seq_df[target], longest_seq


def pad_sequences(sequences, padding_shape, value):
    # padded_shape is (longest, width)
    padded = np.full((len(sequences), padding_shape[0], padding_shape[1]), fill_value=value)
    for index, sequence in enumerate(sequences):
        padded[index, 0:sequence.shape[0], :] = sequence
    return padded


def classification_model(mask_value, input_shape):
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=WAME(learning_rate=0.0001),
                  metrics=['accuracy'])
    print(model.summary())
    return model


def regression_model(mask_value, input_shape):
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse',
                  optimizer=WAME(learning_rate=0.0001),
                  metrics=['mean_squared_error'])
    print(model.summary())
    return model


def classy():
    sequences, targets, longest_sequence = build_numeric_sequences(target='chance')
    padding_shape = (longest_sequence, len(sequences[0][0]))  # (longest, width)
    padding_value = -10.0
    sequences_padded = pad_sequences(sequences, padding_shape, padding_value)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    model = classification_model(mask_value=padding_value,
                                 input_shape=padding_shape)

    h = model.fit(x_train,
                  y_train,
                  validation_data=(x_val, y_val),
                  epochs=10,
                  batch_size=1024)

    scores = model.evaluate(x_test, y_test, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_history(h, 'accuracy')
    return scores, model, pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'prob': y_prob})


def chancy():
    sequences, targets, longest_sequence = build_numeric_sequences(target='xg')
    padding_shape = (longest_sequence, len(sequences[0][0]))  # (longest, width)
    padding_value = -10.0
    sequences_padded = pad_sequences(sequences, padding_shape, padding_value)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    model = regression_model(mask_value=padding_value,
                             input_shape=padding_shape)

    h = model.fit(x_train,
                  y_train,
                  validation_data=(x_val, y_val),
                  epochs=10,
                  batch_size=2048)

    scores = model.evaluate(x_test, y_test, verbose=True)
    print("MSE: %.2f%%" % scores[1])
    y_prob = [i[0] for i in model.predict(x_test)]

    print('R2 Score:', r2_score(y_test, y_prob))
    plot_history(h, 'mean_squared_error')
    return pd.DataFrame({'actual': y_test, 'prob': y_prob})
