import io

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

from viz import plot_history
from xg_utils import XgMap
from utils import list_if_not_nan, split_location, location_to_text, perform_oversampling


def load_events():
    e = pd.read_csv('all_events_orig_bak.csv', nrows=50000)
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
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = \
        round(100 * (e.to_goal_start - e.to_goal_end) / e.to_goal_start, 0)

    e['delta_y'] = e.location_y.diff().abs()
    e['delta_x'] = e.location_x.diff().abs()

    e['chance'] = ~e['shot_type'].isna()
    xg = XgMap()

    def calc_xg(event):
        return xg.value(event.location_x, event.location_y)

    e['xg'] = e.apply(func=lambda event: xg.value(event.location_x, event.location_y), axis=1)
    e.loc[e.type == 'Shot', 'xg'] = e.shot_statsbomb_xg

    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def apply_text_binning(events):
    events['location_text'] = events.apply(location_to_text, axis=1)

    events.loc[events.pass_speed != np.nan, 'pass_speed_text'] = pd.cut(events.pass_speed,
                                                                        bins=[-1, 5, 20, 150],
                                                                        labels=['tapped', 'solid', 'pinged'])

    events.loc[events.carry_speed != np.nan, 'carry_speed_text'] = pd.cut(events.carry_speed,
                                                                          bins=[-1, 2.5, 6, 1500],
                                                                          labels=['drifted', 'glided', 'surged'])

    events.loc[events.progression_pct != np.nan, 'progression_text'] = pd.cut(events.progression_pct,
                                                                              bins=[-1000, 0, 10, 50, 100],
                                                                              labels=['backwards', 'sideways',
                                                                                      'forwards', 'dangerous'])


class EventString(list):
    def add(self, event_str):
        if type(event_str) == str:
            self.append(event_str.lower())

    def to_str(self):
        return ' '.join(self)


def build_text_sequences(target):
    e = load_events()
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
            #commentary.add(event.location_text)
            #commentary.add(event.progression_text)
            if event_type == 'pass':
                # commentary.add(event.pass_outcome)
                commentary.add(event.pass_speed_text)
                commentary.add(event.pass_height)
                commentary.add(event.pass_type)
            if event_type == 'carry':
                commentary.add(event.carry_speed_text)
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
    return seq_df.text, seq_df[target], g.index.count().max()


def assemble_model(input_layer, loss_function, metrics):
    model = Sequential()
    model.add(input_layer)
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64)) #32
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=metrics)

    print(model.summary())
    return model


def classy():
    sequences, targets, longest_sequence = build_text_sequences(target='chance')
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(sequences)
    sequences_padded = sequence.pad_sequences(t.texts_to_sequences(sequences), maxlen=longest_sequence)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    # create the model

    embedding_layer = Embedding(input_dim=longest_sequence, output_dim=32)

    metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives()]

    model = assemble_model(embedding_layer, BinaryCrossentropy(), metrics)

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=1024)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=True)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_prob = [i[0] for i in model.predict(x_test)]
    y_pred = [round(i) for i in y_prob]
    print(confusion_matrix(y_test, y_pred))

    save_embeddings(weights=embedding_layer.get_weights()[0], vocab=t.index_word.values())

    plot_history(h, 'accuracy')
    return pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'prob': y_prob})


def save_embeddings(weights, vocab):
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


def chancy():
    sequences, targets, longest_sequence = build_text_sequences(target='xg')
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(sequences)
    sequences_padded = sequence.pad_sequences(t.texts_to_sequences(sequences), maxlen=longest_sequence)

    x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    # create the model
    embedding_layer = Embedding(input_dim=longest_sequence, output_dim=32)

    model = assemble_model(embedding_layer, MeanSquaredError(), [MeanSquaredError()])

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=1024)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=True)
    print("MSE: %.2f%%" % scores[1])
    y_prob = [i[0] for i in model.predict(x_test)]

    print('R2 Score:', r2_score(y_test, y_prob))
    plot_history(h, 'mean_squared_error')
    return pd.DataFrame({'actual': y_test, 'prob': y_prob})


def label_word(word):
    pass
