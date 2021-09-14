import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Masking, Dropout
from keras.metrics import Precision, Recall, FalsePositives, FalseNegatives
from keras.losses import MeanSquaredError

from sb_interface import load_events
from utils import perform_oversampling
from wame_opt import WAME


def classification(sample_size=5000, epochs=20):
    nes = NumericEventSequence(sample_size=sample_size)
    nes.do_classification(epochs=epochs)
    nes.print_metrics()
    return nes


def regression(sample_size=5000, epochs=20):
    nes = NumericEventSequence(sample_size=sample_size)
    nes.do_regression(epochs=epochs)
    nes.print_metrics()
    return nes


class NumericEventSequence(object):
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

        g = e.groupby(by=['match_id', 'possession'])
        self.longest_sequence = g.index.count().max()
        match_pos = []
        sequences = []
        target_chance = []
        target_xg = []
        for ((match_id, possession), events) in g:
            events = events.set_index(events['index'])
            events = events.sort_index()
            events.fillna(value=0.0, inplace=True)

            match_pos.append(f'{match_id}_{possession}')
            seq_events = events[['pass_speed', 'pass_length', 'carry_speed', 'carry_length']]
            sequences.append(seq_events.values)
            target_chance.append(events.chance.any())
            target_xg.append(events.xg.iloc[-1])

        self.sequences = perform_oversampling(pd.DataFrame({'match_pos': match_pos,
                                                            'sequence': sequences,
                                                            'chance': target_chance,
                                                            'xg': target_xg}))
        print(self.sequences.chance.value_counts())
        return self.sequences.sequence, self.sequences[['match_pos', target]], g.index.count().max()

    def pad(self):
        padding_shape = (self.longest_sequence, len(self.sequences.sequence[0][0])) # longest, width
        padding_value = -10.0
        padded = np.full((len(self.sequences.sequence), padding_shape[0], padding_shape[1]), fill_value=padding_value)
        for index, sequence in enumerate(self.sequences.sequence):
            padded[index, 0:sequence.shape[0], :] = sequence
        return padded, Masking(mask_value=padding_value, input_shape=padding_shape)

    def assemble_model(self, input_layer, loss_function, metrics):
        self.model = Sequential()
        self.model.add(input_layer)
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss=loss_function,
                           optimizer=WAME(learning_rate=0.0001),
                           metrics=metrics)
        print(self.model.summary())
        return self.model

    def do_classification(self, epochs=200, target='chance'):
        if self.sequences is None:
            self.build(target=target)
        targets = self.sequences[['match_pos', target]]
        sequences_padded, masking_layer = self.pad()

        x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.chance, test_size=0.1, random_state=0)

        metrics = ['accuracy', Precision(), Recall(), FalsePositives(), FalseNegatives()]
        self.model = self.assemble_model(masking_layer, 'binary_crossentropy', metrics)
        self.training = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=1024)

        self.predicts = pd.DataFrame({'match_pos': y_test.match_pos,
                                      'actual': y_test.chance,
                                      'predicted': [round(i[0]) for i in self.model.predict(x_test)]})

        scores = self.model.evaluate(x_test, y_test.chance, verbose=True)
        self.metrics = {'accuracy': round(scores[1] * 100, 1),
                        'confusion_matrix': confusion_matrix(y_test.chance, self.predicts.predicted),
                        'classification_report': classification_report(y_test.chance, self.predicts.predicted)}

        return self.metrics, self.predicts

    def do_regression(self, target='xg', epochs=200):
        sequences, targets, longest_sequence = self.build(target=target)
        sequences_padded, masking_layer = self.pad()

        x_train, x_test, y_train, y_test = train_test_split(sequences_padded, targets, test_size=0.2, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.xg, test_size=0.1, random_state=0)

        self.model = self.assemble_model(masking_layer, MeanSquaredError(), [MeanSquaredError()])
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
