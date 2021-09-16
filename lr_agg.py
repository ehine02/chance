import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from sb_interface import load_events
from utils import perform_oversampling


def build_aggregates(sample_size=5000, read_rows=500000):
    e = load_events(sample_size=sample_size, read_rows=read_rows)
    e.type = e.type.str.lower()
    e = e.loc[e.team == e.possession_team]
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    e = e.loc[e['type'].isin(['pass', 'carry'])]
    aggregates = []

    for ((match_id, possession_id), events) in e.groupby(by=['match_id', 'possession']):
        events = events.set_index(events['index'])
        events = events.sort_index()
        events.fillna(value=0.0, inplace=True)
        location_x = events.location_x.iloc[-1]
        location_y = events.location_y.iloc[-1]
        match_pos = f'{match_id}_{possession_id}'
        aggregates.append([match_pos, events.pass_length.sum(), events.pass_length.var(),
                           events.pass_speed.mean(), events.pass_speed.var(), events.pass_speed.max(),
                           events.carry_length.sum(), events.carry_length.var(),
                           events.carry_speed.mean(), events.carry_speed.var(), events.carry_speed.max(),
                           events.delta_x.sum(), events.progression_pct.sum(),
                           location_x, location_y, events.chance.any()])

    aggregates = pd.DataFrame(aggregates, columns=['match_pos', 'sum_pass_length', 'var_pass_length',
                                                   'avg_pass_speed', 'var_pass_speed', 'max_pass_speed',
                                                   'sum_carry_length', 'var_carry_length',
                                                   'avg_carry_speed', 'var_carry_speed', 'max_carry_speed',
                                                   'sum_delta_x', 'sum_progression_pct',
                                                   'location_x', 'location_y', 'chance'])

    aggregates.replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregates.fillna(value=0.0, inplace=True)
    return aggregates, e


def do_logistic_regression(x, inputs, target='chance'):
    y = x.pop(target)

    if inputs is not None:
        x = x[inputs]

    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    model_probs = model.predict_proba(x_test)[:, 1]
    dumb_probs = [0 for _ in range(len(y_test))]

    dumb_auc = roc_auc_score(y_test, dumb_probs)
    model_auc = roc_auc_score(y_test, model_probs)

    print(f'Dumb Constant Guess: ROC AUC={dumb_auc}')
    print(f'Logistic Regression: ROC AUC={model_auc}')
    print(model.score(x_test, y_test))

    y_pred = [round(p) for p in model_probs]
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return x, y, pd.DataFrame({'actual': y_test, 'pred': y_pred, 'prob': model_probs})


def classy(sample_size=2000):
    aggregates, events = build_aggregates(sample_size=sample_size)

    df = perform_oversampling(aggregates)

    # inputs = ['match_pos', 'sum_pass_length', 'var_pass_length',
    #           'avg_pass_speed', 'var_pass_speed', 'max_pass_speed',
    #           'sum_carry_length', 'var_carry_length',
    #           'avg_carry_speed', 'var_carry_speed', 'max_carry_speed', 'sum_progression_pct']

    inputs = ['sum_pass_length', 'var_pass_length', 'sum_carry_length', 'var_carry_length',
              'avg_pass_speed', 'var_pass_speed', 'avg_carry_speed', 'var_carry_speed']

    return do_logistic_regression(df, inputs), events
