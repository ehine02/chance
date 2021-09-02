import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Masking
from tensorflow.python.keras.layers import Dropout

from utils import list_if_not_nan, split_location
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
    e['delta_y'] = e.location_y.diff()
    e['delta_x'] = e.location_x.diff()

    e['chance'] = ~e['shot_type'].isna()
    xg = XgMap()

    def calc_xg(event):
        return xg.value(event.location_x, event.location_y)

    e['xg'] = e.apply(func=calc_xg, axis=1)
    e.loc[e.type == 'Shot', 'xg'] = e.shot_statsbomb_xg
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def build_numseq():
    # Parameters
    dimensions = ['pass_speed', 'pass_length', 'carry_speed', 'carry_length', 'location_x']#, 'progression_pct']
    # , 'pass_angle']#, 'progression_pct', 'to_goal']

    # patterns

    # bool switch - y goes from >60 to <20 in one pass or vice versa
    # bool width - y goes from >30 and < 50 to >70 or < 10 in one getpass
    # bool layoff - long to centre and then short



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
    max_seq = 0
    for ((match_id, possession), events) in g:
        if len(events.index) > max_seq:
            max_seq = len(events.index)
        events = events.set_index(events['index'])
        events = events.sort_index()
        events.fillna(value=0.0, inplace=True)

        seq_events = events[dimensions]
        sequences.append(seq_events.values)
        target_chance.append(events.chance.any())
        target_xg.append(events.xg.iloc[-1])

    df = pd.DataFrame({'seqs': sequences, 'chance': target_chance, 'xg': target_xg})
    # Oversampling performed here
    # first count the records of the majority
    majority_count = df.chance.value_counts().max()
    working = [df]
    # group by each salary band
    for _, chance in df.groupby('chance'):
        # append N samples to working list where N is the difference between majority and this band
        working.append(chance.sample(majority_count - len(chance), replace=True))
    # add the working list contents to the overall dataframe
    df = pd.concat(working)

    print(df.chance.value_counts())
    return df, max_seq


def classy():
    df, longest_seq = build_numseq()

    x = np.array(df.seqs.to_list())
    y = np.array([[int(t)] for t in df.chance.to_list()])
    print(x.shape)
    print(y.shape)

    seq_count = x.shape[0]
    seq_width = len(x[0][0])

    special_value = -10.0
    x_pad = np.full((seq_count, longest_seq, seq_width), fill_value=special_value)
    for s, x in enumerate(x):
        seq_len = x.shape[0]
        x_pad[s, 0:seq_len, :] = x

    x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(longest_seq, seq_width)))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=WAME(learning_rate=0.0001),#keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy',
                           keras.metrics.Precision(),
                           keras.metrics.Recall(),
                           keras.metrics.FalsePositives(),
                           keras.metrics.FalseNegatives()])
    print(model.summary())

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=1024)
    scores = model.evaluate(x_test, y_test, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_history(h, 'accuracy')
    return scores, model, pd.DataFrame({'actual': [i[0] for i in y_test], 'predicted': y_pred, 'prob': y_prob})


def chancy():
    df, longest_seq = build_numseq()

    x = np.array(df.seqs.to_list())
    y = np.array([[t] for t in df.xg.to_list()])
    print(x.shape)
    print(y.shape)

    seq_count = x.shape[0]
    seq_width = len(x[0][0])

    special_value = -10.0
    x_pad = np.full((seq_count, longest_seq, seq_width), fill_value=special_value)
    for s, x in enumerate(x):
        seq_len = x.shape[0]
        x_pad[s, 0:seq_len, :] = x

    x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(longest_seq, seq_width)))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse',
                  optimizer=WAME(learning_rate=0.0001),#keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['mean_squared_error'])
    print(model.summary())

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=1024)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=True)
    print("MSE: %.2f%%" % scores[1])
    y_prob = [i[0] for i in model.predict(x_test)]

    print('R2 Score:', r2_score(y_test, y_prob))
    plot_history(h, 'mean_squared_error')
    return pd.DataFrame({'actual': [i[0] for i in y_test], 'prob': y_prob})