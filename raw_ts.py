import pandas as pd
import numpy as np
import ast

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import CuDNNGRU, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
import tensorflow as tf


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
    e = pd.read_csv('all_events_orig.csv', nrows=50000)
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
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100*(e.to_goal_start - e.to_goal_end) / e.to_goal_start, 0)
    e['chance'] = ~e['shot_type'].isna()
    # e.fillna(value=0.0, inplace=True)
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def main():
    e = load_events()
    e = e.loc[e['possession_team'] == e['team']]
    e.pass_height = e.pass_height.str.split().str[0]
    g = e.groupby(by=['match_id', 'possession'])
    ts_list = []
    ts_class = []
    for ((match_id, possession), events) in g:
        ts_class.append([events.chance.any().astype(int)])
        events = events.set_index(events['index'])
        events = events.sort_index()
        events = events.loc[events['type'].isin(['Pass', 'Carry'])]
        events = events.transpose()
        ts_list.append([events.loc['pass_speed'].combine_first(events.loc['carry_speed']).to_list()])
    return np.asarray(ts_list), np.asarray(1*ts_class)


def classy(ts_data):
    x = ts_data[0]
    y = ts_data[1]
    max_possession_events = 0
    for seq in x:
        if len(seq) > max_possession_events:
            max_possession_events = len(seq)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    r_train_x = tf.ragged.constant(x_train)
    r_test_x = tf.ragged.constant(x_test)
    print(r_train_x.shape)
    max_seq = r_train_x.bounding_shape()[-1]
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[None, None], dtype=tf.float64, ragged=True),
        #tf.keras.layers.Embedding(10, 128),
        #tf.keras.layers.LSTM(32),
        #tf.keras.layers.Dense(32),
        #tf.keras.layers.Activation(tf.nn.relu),
        #tf.keras.layers.Dense(1)
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    NumEpochs = 10
    BatchSize = 32

    keras_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    keras_model.summary()

    history = keras_model.fit(r_train_x, y_train, epochs=NumEpochs, batch_size=BatchSize)
                              #validation_data=(r_test_x, y_test))
    scores = keras_model.evaluate(x_test, y_test, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in keras_model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))


def test():
    xx = tf.ragged.constant([
        [[0.1, 0.2]],
        [[0.4, 0.7, 0.5, 0.6]]
    ])

    """
    Labels represented as OneHotEncoding so you 
    should use CategoricalCrossentropy instade of SparseCategoricalCrossentropy
    """

    yy = np.array([[0, 0, 1], [1, 0, 0]])

    # For ragged tensor , get maximum sequence length
    max_seq = xx.bounding_shape()[-1]

    mdl = tf.keras.Sequential([
        # Input Layer with shape = [Any,  maximum sequence length]
        tf.keras.layers.Input(shape=[None, max_seq], batch_size=2, dtype=tf.float32, ragged=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # CategoricalCrossentropy
    mdl.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])

    mdl.summary()
    history = mdl.fit(xx, yy, epochs=10)


def test2():
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed
    from tensorflow.keras.utils import to_categorical
    import numpy as np

    model = Sequential()

    model.add(LSTM(32, return_sequences=True, input_shape=(None, 5)))
    model.add(LSTM(8, return_sequences=True))
    model.add(TimeDistributed(Dense(2, activation='sigmoid')))

    print(model.summary(90))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')

    def train_generator():
        # e = load_events()
        # e = e.loc[~e['type'].isin(['Ball Receipt*'])]
        # e.pass_height = e.pass_height.str.split().str[0]
        # e.type = e.type.str.lower()
        # e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
        # e = e.loc[~e['type'].isin(['shot', 'block', 'goal keeper', 'pressure', 'clearance'])]
        # g = e.groupby(by=['match_id', 'possession'])
        # for n in range(3, 76):
        #     pos = g.filter(lambda x: len(x) == n)
        #     x_train = pos[['pass_speed', 'carry_speed', 'pass_angle', 'progression_pct', 'to_goal']]
        #     x_train = x_train.values.reshape(-1, n, 5)
        #     y_train = pos[['match_id', 'possession', 'chance']]
        #     y_train = y_train.groupby(by=['match_id', 'possession']).any().values
        #     y_train = to_categorical(y_train).reshape(y_train.shape[0], -1, 2)
        #     print(x_train.shape)
        #     print(y_train.shape)
        #     yield x_train, y_train

        while True:
            sequence_length = np.random.randint(10, 100)
            x_train = np.random.random((1000, sequence_length, 5))
            print('X shape:', x_train.shape)
            # y_train will depend on past 5 timesteps of x
            y_train = x_train[:, :, 0]
            for i in range(1, 5):
                y_train[:, i:] += x_train[:, :-i, i]
            y_train = to_categorical(y_train > 2.5)
            print(x_train.shape)
            print(y_train.shape)
            yield x_train, y_train


    model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)


def test3():
    from keras import Sequential
    from tensorflow.keras.utils import Sequence
    from keras.layers import LSTM, Dense, Masking
    import numpy as np

    class MyBatchGenerator(Sequence):
        'Generates data for Keras'

        def __init__(self, X, y, batch_size=1, shuffle=True):
            'Initialization'
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.y) / self.batch_size))

        def __getitem__(self, index):
            return self.__data_generation(index)

        def on_epoch_end(self):
            'Shuffles indexes after each epoch'
            self.indexes = np.arange(len(self.y))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, index):
            Xb = np.empty((self.batch_size, *X[index].shape))
            yb = np.empty((self.batch_size, *y[index].shape))
            # naively use the same sample over and over again
            for s in range(0, self.batch_size):
                Xb[s] = X[index]
                yb[s] = y[index]
            return Xb, yb

    # Parameters
    N = 1000
    halfN = int(N / 2)
    dimension = 5
    lstm_units = 3

    e = load_events()
    e = e.loc[~e['type'].isin(['Ball Receipt*'])]
    e.pass_height = e.pass_height.str.split().str[0]
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    e = e.loc[~e['type'].isin(['shot', 'block', 'goal keeper', 'pressure', 'clearance'])]
    g = e.groupby(by=['match_id', 'possession'])
    sequences = []
    target = []
    for ((match_id, possession), events) in g:
        events = events.set_index(events['index'])
        events = events.sort_index()
        seq_events = events[['pass_speed', 'carry_speed', 'pass_angle', 'progression_pct', 'to_goal']]
        seq_events.fillna(value=0.0, inplace=True)
        sequences.append(seq_events.values)
        target.append(events.chance.any())

    df = pd.DataFrame({'seqs': sequences, 'chance': target})
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


    X = np.array(df.seqs.to_list())
    y = np.array([[t] for t in df.chance.to_list()])
    print(X.shape)
    print(y.shape)

    X, x_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(None, dimension)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.BinaryAccuracy(),
                                                                            tf.keras.metrics.Precision(),
                                                                            tf.keras.metrics.Recall(),
                                                                            tf.keras.metrics.FalsePositives(),
                                                                            tf.keras.metrics.FalseNegatives()])
    print(model.summary())
    model.fit_generator(MyBatchGenerator(X, y, batch_size=1), epochs=2)

    y_test = pd.Series(t[0] for t in y_test)
    scores = model.evaluate_generator(MyBatchGenerator(x_test, y_test, batch_size=1), verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict_generator(MyBatchGenerator(x_test, y_test, batch_size=1))]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))
    return pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'prob': y_prob})