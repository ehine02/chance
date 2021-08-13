import keras.preprocessing.text
import pandas as pd
import numpy as np
import ast

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D


# numpy.random.seed(7)

def euclidean_distance(start, end):
    return np.sqrt(np.power(end[0] - start[0], 2) + np.power(end[1] - start[1], 2))


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
    e.loc[e.pass_speed != np.nan, 'pass_speed_text'] = pd.cut(e.pass_speed, bins=[-1, 5, 20, 150],
                                                              labels=['tapped', 'solid', 'pinged'])
    e.loc[e.carry_speed != np.nan, 'carry_speed_text'] = pd.cut(e.carry_speed, bins=[-1, 2.5, 6, 1500],
                                                                labels=['drifted', 'glided', 'surged'])
    e.loc[e.location != np.nan, 'to_goal_start'] = \
        round(np.sqrt((120 - e.location_x) ** 2 + (40 - e.location_y) ** 2), 0)
    e.loc[e.type == 'Pass', 'to_goal_end'] = \
        round(np.sqrt((120 - e.pass_end_x) ** 2 + (40 - e.pass_end_y) ** 2), 0)
    e.loc[e.type == 'Carry', 'to_goal_end'] = \
        round(np.sqrt((120 - e.carry_end_x) ** 2 + (40 - e.carry_end_y) ** 2), 0)
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100*(e.to_goal_start - e.to_goal_end) / e.to_goal_start, 0)
    e.loc[e.progression_pct != np.nan, 'progression_text'] = pd.cut(e.progression_pct, bins=[-1000, 0, 10, 50, 100],
                                                                    labels=['backwards', 'sideways', 'forwards', 'dangerous'])
    e['location_text'] = e.apply(location_to_text, axis=1)
    e['chance'] = ~e['shot_type'].isna()
    # e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def location_to_text(row):
    if row.location_x < 60:
        return 'ownhalf'
    if row.location_x < 80:
        return 'midfield'
    if row.location_y >= 62:
        return 'rightwing'
    if row.location_y <= 18:
        return 'leftwing'
    if row.location_x > 102:
        return 'box'
    return 'boxedge'


class EventString(list):
    def add(self, event_str):
        if type(event_str) == str:
            self.append(event_str.lower())

    def to_str(self):
        return ' '.join(self)


def main():
    text = pd.DataFrame(columns=['text', 'chance'])
    e = load_events()
    e = e.loc[~e['type'].isin(['Ball Receipt*'])]
    e.pass_height = e.pass_height.str.split().str[0]
    g = e.groupby(by=['match_id', 'possession'])
    max_events = 0
    for ((match_id, possession), events) in g:
        if len(events.index) > max_events:
            max_events = len(events.index)
        events = events.set_index(events['index'])
        events = events.sort_index()
        chance = False
        events_list = EventString()
        for _, row in events.iterrows():
            chance = chance or row.chance
            event_type = str(row['type']).lower()
            if event_type in ['shot', 'block', 'goal keeper', 'pressure', 'clearance']:
                continue
            events_list.add(row.location_text)
            events_list.add(row.progression_text)
            if event_type == 'pass':
                events_list.add(row.pass_outcome)
                events_list.add(row.pass_speed_text)
                events_list.add(row.pass_height)
                events_list.add(row.pass_type)
            if event_type == 'carry':
                events_list.add(row.carry_speed_text)
            if row.under_pressure == True:
                events_list.add('pressured')
            events_list.add(event_type)
            events_list.add('|')
        match_pos = '_'.join([str(match_id), str(possession)])
        if len(events_list):
            text = text.append({'match_pos': match_pos, 'text': events_list.to_str(), 'chance': chance}, ignore_index=True)
    print('MAX EVENTS:', max_events)
    df = text.copy()
    # Oversampling performed here
    # first count the records of the majority
    majority_count = df['chance'].value_counts().max()
    working = [df]
    # group by each salary band
    for _, salary_band in df.groupby('chance'):
        # append N samples to working list where N is the difference between majority and this band
        working.append(salary_band.sample(majority_count - len(salary_band), replace=True))
    # add the working list contents to the overall dataframe
    df = pd.concat(working)
    print(df['chance'].value_counts())
    return df


from keras.datasets import imdb


def test_imdb():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
    ed = 0


def classy(text):
    y = text.pop('chance').astype(int)
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(text.text.tolist())
    print('VOCAB SIZE: ', str(len(t.word_counts)))
    x_train, x_test, y_train, y_test = train_test_split(text, y, test_size=0.1, random_state=0)
    x_train_encoded = t.texts_to_sequences(x_train.text.tolist())
    x_test_encoded = t.texts_to_sequences(x_test.text.tolist())
    # truncate and pad input sequences
    max_possession_events = 210
    x_train_encoded = sequence.pad_sequences(x_train_encoded, maxlen=max_possession_events)
    x_test_encoded = sequence.pad_sequences(x_test_encoded, maxlen=max_possession_events)
    # create the model
    embedding_vector_length = 64
    model = Sequential()
    model.add(Embedding(60, embedding_vector_length, input_length=max_possession_events))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.BinaryAccuracy(),
                                                                         keras.metrics.Precision(),
                                                                         keras.metrics.Recall(),
                                                                         keras.metrics.FalsePositives(),
                                                                         keras.metrics.FalseNegatives()])
    print(model.summary())
    model.fit(x_train_encoded, y_train, epochs=10, batch_size=128)
    # Final evaluation of the model
    scores = model.evaluate(x_test_encoded, y_test, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict(x_test_encoded)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))

    return pd.DataFrame({'id': x_test.match_pos, 'seq': x_test.text,
                         'actual': y_test, 'predicted': y_pred, 'prob': y_prob})
