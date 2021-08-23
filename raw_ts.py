import pandas as pd
import numpy as np
import ast

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense, Masking


def list_if_not_nan(x):
    if x is np.nan:
        return np.nan
    else:
        return ast.literal_eval(x)


def split_location(x):
    if x is not np.nan:
        return round(x[0], 0), round(x[1], 0)
    return np.nan, np.nan


def load_events():
    e = pd.read_csv('all_events_orig.csv', nrows=100000)
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
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100 * (e.to_goal_start - e.to_goal_end) / e.to_goal_start,
                                                              0)
    e['chance'] = ~e['shot_type'].isna()
    # e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def classy():
    # Parameters
    dimensions = ['pass_speed', 'pass_length', 'carry_speed', 'carry_length']#, 'pass_angle']#, 'progression_pct', 'to_goal']

    e = load_events()
    e = e.loc[~e['type'].isin(['Ball Receipt*'])]
    e.pass_height = e.pass_height.str.split().str[0]
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    #e.xg = e.groupby(by=['match_id', 'possession'])['xg'].transform(lambda xg: xg[-1])
    e = e.loc[~e['type'].isin(['shot', 'block', 'goal keeper', 'pressure', 'clearance'])]
    g = e.groupby(by=['match_id', 'possession'])
    sequences = []
    target = []
    max_seq = 0
    for ((match_id, possession), events) in g:
        if len(events.index) > max_seq:
            max_seq = len(events.index)
        events = events.set_index(events['index'])
        events = events.sort_index()
        seq_events = events[dimensions]
        seq_events.fillna(value=0.0, inplace=True)
        sequences.append(seq_events.values)
        target.append(events.chance.any())

    df = pd.DataFrame({'seqs': sequences, 'chance': target})
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

    x = np.array(df.seqs.to_list())
    y = np.array([[int(t)] for t in df.chance.to_list()])
    print(x.shape)
    print(y.shape)

    special_value = -10.0
    x_pad = np.full((x.shape[0], max_seq, len(dimensions)), fill_value=special_value)
    for s, x in enumerate(x):
        seq_len = x.shape[0]
        x_pad[s, 0:seq_len, :] = x

    x, x_test, y, y_test = train_test_split(x_pad, y, test_size=0.1, random_state=0)

    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(max_seq, len(dimensions))))
    model.add(LSTM(16))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    h = model.fit(x, y, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    history_plot(h, 'accuracy')
    return scores, model, pd.DataFrame({'actual': [i[0] for i in y_test], 'predicted': y_pred, 'prob': y_prob})


def history_plot(history, what):
    x = history.history[what]
    val_x = history.history['val_' + what]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = np.asarray(history.epoch) + 1

    plt.subplot(1, 2, 1)
    plt.plot(epochs, x, 'b', label="Training " + what)
    plt.plot(epochs, val_x, 'r', label="Validation " + what)
    plt.grid()
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label="Training loss")
    plt.plot(epochs, val_loss, 'r', label="Validation loss")
    plt.grid()
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
