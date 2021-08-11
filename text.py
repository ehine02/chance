import keras.preprocessing.text
import pandas as pd
import numpy as np
import ast
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D

#numpy.random.seed(7)

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
    e = pd.read_csv('all_events.csv', nrows=10000)
    e = e.loc[~e['shot_type'].isin(['Penalty'])]
    e = e.loc[~e['location'].isin([np.nan])]
    e['location'] = e.location.apply(list_if_not_nan)
    e['pass_end_location'] = e.pass_end_location.apply(list_if_not_nan)
    e['carry_end_location'] = e.carry_end_location.apply(list_if_not_nan)
    e['location_x'], e['location_y'] = zip(*e.location.map(split_location))
    e['pass_end_x'], e['pass_end_y'] = zip(*e.pass_end_location.map(split_location))
    e['carry_end_x'], e['carry_end_y'] = zip(*e.carry_end_location.map(split_location))
    e.loc[e.type == 'Carry', 'carry_length'] = \
        np.sqrt((e.carry_end_x-e.location_x)**2 + (e.carry_end_y-e.location_y)**2)
    e.loc[e.type == 'Carry', 'carry_speed'] = e.carry_length / e.duration
    e.loc[e.type == 'Pass', 'pass_speed'] = e.pass_length / e.duration
    e['chance'] = ~e['shot_type'].isna()
    e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def location_to_text(x, y):
    if x < 60:
        return 'ownhalf'
    if x < 80:
        return 'midfield'
    if y >= 62:
        return 'rightwing'
    if y <= 18:
        return 'leftwing'
    if x > 102:
        return 'box'
    return 'boxedge'


def pass_speed(duration, length):
    if duration in [np.nan, 0.0] or length in [np.nan, 0.0]:
        return ''
    speed = length / duration
    if speed < 10:
        return 'tap'
    if speed > 18:
        return 'ping'
    return 'safe'


def carry_length():
    pass


def carry_speed():
    pass


def main():
    text = pd.DataFrame(columns=['text', 'chance'])
    e = load_events()
    e = e.loc[~e['type'].isin(['Ball Receipt*'])]
    e.loc[e['type'] == 'Pass', 'type'] = e['pass_height']
    g = e.groupby(by=['match_id', 'possession'])
    max_events = 0
    for ((match_id, possession), events) in g:
        if len(events.index) > max_events:
            max_events = len(events.index)
        events = events.set_index(events['index'])
        events = events.sort_index()
        chance = False
        events_list = []
        for _, row in events.iterrows():
            event_type = str(row['type']).lower()
            chance = chance or row['chance']
            events_list.append(location_to_text(row['location_x'], row['location_y']))
            if event_type in ['shot', 'goal keeper']:
                continue
            events_list.append(pass_speed(row.duration, row.pass_length))

            events_list.append(event_type)
        text = text.append({'text': ' '.join(events_list), 'chance': chance}, ignore_index=True)
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


def classy(text):
    y = text.pop('chance').astype(int)
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(text.text.tolist())
    x_train, x_test, y_train, y_test = train_test_split(text, y, test_size=0.1, random_state=0)
    x_train_encoded = t.texts_to_matrix(x_train.text.tolist())
    x_test_encoded = t.texts_to_matrix(x_test.text.tolist())
    # truncate and pad input sequences
    max_possession_events = 120
    x_train_encoded = sequence.pad_sequences(x_train_encoded, maxlen=max_possession_events)
    x_test_encoded = sequence.pad_sequences(x_test_encoded, maxlen=max_possession_events)
    # create the model
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(50, embedding_vector_length, input_length=max_possession_events))
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
    model.fit(x_train_encoded, y_train, epochs=10, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(x_test_encoded, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_pred = [round(i[0], 0) for i in model.predict(x_test_encoded)]
    print(confusion_matrix(y_test, y_pred))

    return pd.DataFrame({'seq': x_test.text, 'actual': y_test, 'predicted': y_pred})
