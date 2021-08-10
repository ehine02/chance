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


def load_events():
    e = pd.read_csv('all_events.csv', nrows=50000)
    e = e.loc[~e['shot_type'].isin(['Penalty'])]
    e = e.loc[~e['location'].isin([np.nan])]
    e['location'] = e.location.apply(ast.literal_eval)
    e['location_x'] = e.location.apply(lambda x: round(x[0], 0))
    e['location_y'] = e.location.apply(lambda x: round(x[1], 0))
    e['pass_end_x'] = e.pass_end_location.apply(lambda x: round(x[0], 0))
    e['pass_end_y'] = e.pass_end_location.apply(lambda x: round(x[1], 0))
    e['carry_end_x'] = e.carry_end_location.apply(lambda x: round(x[0], 0))
    e['carry_end_y'] = e.carry_end_location.apply(lambda x: round(x[1], 0))
    e['chance'] = ~e['shot_type'].isna()
    e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['location'])
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


def pass_speed_to_text(duration, length):
    if duration in [np.nan, 0.0] or length in [np.nan, 0.0]:
        return ''
    speed = length / duration
    if speed < 10:
        return 'slow'
    if speed > 18:
        return 'fast'
    return 'medium'


def progressive_action(start, end):
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
            events_list.append(location_to_text(row['location_x'], row['location_y']))
            events_list.append(pass_speed_to_text(row['duration'], row['pass_length']))
            events_list.append(row['type'])
            chance = chance or row['chance']
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
    encoded = t.texts_to_matrix(text.text.tolist())
    x_train, x_test, y_train, y_test = train_test_split(encoded, y, test_size=0.1, random_state=0)
    # truncate and pad input sequences
    max_possession_events = 120
    x_train = sequence.pad_sequences(x_train, maxlen=max_possession_events)
    x_test = sequence.pad_sequences(x_test, maxlen=max_possession_events)
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
    model.fit(x_train, y_train, epochs=10, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_pred = [round(i[0], 0) for i in model.predict(x_test)]
    print(confusion_matrix(y_test, y_pred))
    #'possession': {},
    return pd.DataFrame({'actual': y_test, 'predicted': y_pred}), x_test
