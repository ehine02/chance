import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score

from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.preprocessing import sequence
from keras.metrics import Precision, Recall, FalsePositives, FalseNegatives
from keras.losses import BinaryCrossentropy, MeanSquaredError

from sb_interface import load_events
from viz import plot_history, save_embeddings
from utils import location_to_text, perform_oversampling


def apply_text_binning(events):
    events.loc[events.pass_speed != np.nan, 'pass_speed_text'] = pd.qcut(events.pass_speed, 3,
                                                                         labels=['tapped', 'solid', 'pinged'])

    events.loc[events.carry_speed != np.nan, 'carry_speed_text'] = pd.qcut(events.carry_speed, 3,
                                                                           labels=['drifted', 'glided', 'surged'])

    events.loc[events.pass_length != np.nan, 'pass_length_text'] = pd.qcut(events.pass_length, 3,
                                                                           labels=['short', 'midrange', 'longball'])

    events.loc[events.carry_length != np.nan, 'carry_length_text'] = pd.qcut(events.carry_length, 3,
                                                                             labels=['step', 'advance', 'keptgoing'])

    events.loc[events.progression_pct != np.nan, 'progression_text'] = pd.qcut(events.progression_pct, 4,
                                                                               labels=['backwards', 'sideways',
                                                                                       'forwards', 'dangerous'])
    events['location_text'] = events.apply(location_to_text, axis=1)


class EventString(list):
    def add(self, event_str):
        if type(event_str) == str:
            self.append(event_str.lower())

    def to_str(self):
        return ' '.join(self)


def build_text_sequences(target, sample_size=50000):
    e = load_events(sample_size)
    apply_text_binning(e)
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    e = e.loc[e['type'].isin(['shot', 'pass', 'carry', 'dribble', 'dribbled past'])]
    e.xg = e.groupby(by=['match_id', 'possession'])['xg'].transform(lambda x: x.iloc[-1])
    e = e.loc[~e['type'].isin(['shot'])]
    e.pass_height = e.pass_height.str.split().str[0]
    e.pass_type = e.pass_type.str.replace(' ', '')
    e.pass_type = e.pass_type.str.replace('-', '')
    e.type = e.type.str.replace(' ', '')
    g = e.groupby(by=['match_id', 'possession'])
    text = pd.DataFrame()
    for ((match_id, possession), events) in g:
        events = events.set_index(events['index'])
        events = events.sort_index()
        commentary = EventString()
        for _, event in events.iterrows():
            event_type = event['type']
            # commentary.add(event.location_text)
            # commentary.add(event.progression_text)
            if event_type == 'pass':
                commentary.add(event.pass_speed_text)
                commentary.add(event.pass_length_text)
                commentary.add(event.pass_height)
                commentary.add(event.pass_type)
            if event_type == 'carry':
                commentary.add(event.carry_speed_text)
                commentary.add(event.carry_length_text)
            if event.under_pressure == True:
                commentary.add('pressured')
            commentary.add(event_type)
            commentary.add('|')
        match_pos = '_'.join([str(match_id), str(possession)])
        if len(commentary):
            text = text.append({'match_pos': match_pos,
                                'text': commentary.to_str(),
                                'chance': float(events.chance.any()),
                                'xg': events.xg.iloc[-1]},
                               ignore_index=True)

    seq_df = perform_oversampling(text)
    print(seq_df.chance.value_counts())
    return seq_df.text, seq_df[['match_pos', target]], g.index.count().max()


def assemble_model(input_layer, loss_function, metrics):
    model = Sequential()
    model.add(input_layer)
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=metrics)

    print(model.summary())
    return model


def classy(sample_size=10000):
    sequences, targets, longest_sequence = build_text_sequences(target='chance', sample_size=sample_size)

    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(sequences)
    sequences_padded = sequence.pad_sequences(t.texts_to_sequences(sequences), maxlen=longest_sequence)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.chance, test_size=0.1, random_state=0)

    # create the model
    embedding_layer = Embedding(input_dim=longest_sequence, output_dim=32)
    metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives()]
    model = assemble_model(embedding_layer, BinaryCrossentropy(), metrics)

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=1024)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test.chance, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test.chance, y_pred))

    save_embeddings(weights=embedding_layer.get_weights()[0], vocab=t.index_word.values())

    plot_history(h, 'accuracy')
    return (scores[1] * 100, confusion_matrix(y_test.chance, y_pred)), \
           pd.DataFrame({'match_pos': y_test.match_pos, 'actual': y_test.chance, 'predict': y_pred, 'prob': y_prob})


def chancy():
    sequences, targets, longest_sequence = build_text_sequences(target='xg')

    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(sequences)
    sequences_padded = sequence.pad_sequences(t.texts_to_sequences(sequences), maxlen=longest_sequence)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.xg, test_size=0.1, random_state=0)

    # create the model
    embedding_layer = Embedding(input_dim=longest_sequence, output_dim=32)
    model = assemble_model(embedding_layer, MeanSquaredError(), [MeanSquaredError()])

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=1024)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test.xg, verbose=True)
    print("MSE: %.2f%%" % scores[1])
    y_prob = [i[0] for i in model.predict(x_test)]

    print('R2 Score:', r2_score(y_test.xg, y_prob))
    plot_history(h, 'mean_squared_error')
    return (scores[1], r2_score(y_test.xg, y_prob)), \
           pd.DataFrame({'match_pos': y_test.match_pos, 'actual': y_test.xg, 'predict': y_prob})
