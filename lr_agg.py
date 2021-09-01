import keras.optimizers
import pandas as pd
import numpy as np
import ast

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


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
    e = pd.read_csv('all_events.csv', nrows=100000)
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
    e.loc[e.to_goal_end != np.nan, 'progression_pct'] = round(100 * (e.to_goal_start - e.to_goal_end) /
                                                              e.to_goal_start, 0)
    e['delta_y'] = e.location_y.diff()
    e['delta_x'] = e.location_x.diff()

    e['chance'] = ~e['shot_type'].isna()
    e['dribble_success'] = e['dribble_outcome'] == 'Complete'
    e = e.drop(columns=['location', 'pass_end_location', 'carry_end_location'])
    return e


def classy():
    # Parameters
    e = load_events()
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    e = e.loc[e['type'].isin(['pass', 'carry', 'dribble', 'dribbled past'])]
    aggregates = []

    for (_, events) in e.groupby(by=['match_id', 'possession']):
        events = events.set_index(events['index'])
        events = events.sort_index()
        events.fillna(value=0.0, inplace=True)
        ed = 0
        aggregates.append([events.pass_length.sum(),
                           events.pass_length.var(),
                           events.pass_speed.mean(),
                           events.pass_speed.var(),
                           events.carry_length.sum(),
                           events.carry_length.var(),
                           events.carry_speed.mean(),
                           events.carry_speed.var(),
                           events.dribble_success.sum(),
                           events.chance.any()
                           ])

    aggregates = pd.DataFrame(aggregates, columns=['sum_pass_length', 'var_pass_length',
                                                   'avg_pass_speed', 'var_pass_speed',
                                                   'sum_carry_length', 'var_carry_length',
                                                   'avg_carry_speed', 'var_carry_speed', 'sum_dribbles',
                                                   'chance'])

    aggregates.fillna(value=0.0, inplace=True)

    df = aggregates.copy()
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

    print(df.chance.value_counts())

    return log_reg(df)


def log_reg(x, target='chance'):
    y = x.pop(target)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    lr_probs = model.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    print(model.score(x_test, y_test))
    y_pred = [round(p) for p in lr_probs]
    print(confusion_matrix(y_test, y_pred))
    return model, pd.DataFrame({'actual': y_test, 'pred': y_pred, 'prob': lr_probs})
