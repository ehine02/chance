import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score, classification_report

from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.preprocessing import sequence
from keras.metrics import Precision, Recall, FalsePositives, FalseNegatives
from keras.losses import BinaryCrossentropy, MeanSquaredError

from sb_interface import load_events
from viz import plot_history, save_embeddings
from utils import location_to_text, perform_oversampling


def classification(sample_size=5000, epochs=20):
    nes = TextEventSequence(sample_size=sample_size)
    nes.do_classification(epochs=epochs)
    nes.print_metrics()
    return nes


def regression(sample_size=5000, epochs=20):
    nes = TextEventSequence(sample_size=sample_size)
    nes.do_regression(epochs=epochs)
    nes.print_metrics()
    return nes


class EventString(list):
    def add(self, event_str):
        if type(event_str) == str:
            self.append(event_str.lower())

    def to_str(self):
        return ' '.join(self)


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


class TextEventSequence(object):
    def __init__(self, sample_size=5000):
        self.sample_size = sample_size
        self.sequences = None
        self.model = None
        self.training = None
        self.predicts = None
        self.metrics = None
        self.longest_sequence = None

    def build(self, target):
        e = load_events(self.sample_size)

        apply_text_binning(e)
        e.pass_height = e.pass_height.str.split().str[0]
        e.pass_type = e.pass_type.str.replace(' ', '')
        e.pass_type = e.pass_type.str.replace('-', '')
        e.type = e.type.str.replace(' ', '')
        g = e.groupby(by=['match_id', 'possession'])
        self.longest_sequence = g.index.count().max()
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

        self.sequences = perform_oversampling(text)
        return self.sequences.text, self.sequences[['match_pos', target]], g.index.count().max()

    def pad(self):
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.sequences.text)
        sequences = tokenizer.texts_to_sequences(self.sequences.text)
        return sequence.pad_sequences(sequences, maxlen=self.longest_sequence), tokenizer

    def assemble_model(self, input_layer, loss_function, metrics):
        self.model = Sequential()
        self.model.add(input_layer)
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss=loss_function,
                           optimizer='adam',
                           metrics=metrics)

        print(self.model.summary())
        return self.model

    def do_classification(self, target='chance', epochs=200):
        if self.sequences is None:
            self.build(target=target)
        targets = self.sequences[['match_pos', target]]
        sequences_padded, tokenizer = self.pad()

        x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.chance, test_size=0.1, random_state=0)

        # create the model
        embedding_layer = Embedding(input_dim=self.longest_sequence, output_dim=32)
        metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives()]
        self.model = self.assemble_model(embedding_layer, BinaryCrossentropy(), metrics)

        self.training = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=1024)

        self.predicts = pd.DataFrame({'match_pos': y_test.match_pos,
                                      'actual': y_test.chance,
                                      'predicted': [round(i[0]) for i in self.model.predict(x_test)]})

        scores = self.model.evaluate(x_test, y_test.chance, verbose=True)
        self.metrics = {'accuracy': round(scores[1] * 100, 1),
                        'confusion_matrix': confusion_matrix(y_test.chance, self.predicts.predicted).tolist(),
                        'classification_report': classification_report(y_test.chance, self.predicts.predicted)}

        save_embeddings(weights=embedding_layer.get_weights()[0], vocab=tokenizer.index_word.values())

        return self.metrics, self.predicts

    def do_regression(self, target='xg', epochs=200):
        if self.sequences is None:
            self.build(target=target)
        targets = self.sequences[['match_pos', target]]
        sequences_padded, tokenizer = self.pad()

        x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.xg, test_size=0.1, random_state=0)

        embedding_layer = Embedding(input_dim=self.longest_sequence, output_dim=32)
        self.model = self.assemble_model(embedding_layer, MeanSquaredError(), [MeanSquaredError()])
        self.training = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=1024)

        self.predicts = pd.DataFrame({'match_pos': y_test.match_pos,
                                      'actual': y_test.xg,
                                      'predicted': [i[0] for i in self.model.predict(x_test)]})

        scores = self.model.evaluate(x_test, y_test.xg, verbose=True)
        self.metrics = {'mean_squared_error': round(scores[1], 3),
                        'r2_score': r2_score(y_test.xg, self.predicts.predicted)}

        return self.metrics, self.predicts

    def print_metrics(self):
        [print(f'EDS METRICS{metric_name}:\nEDS METRICS{content}') for metric_name, content in self.metrics.items()]
