import keras.optimizers
import pandas as pd
import numpy as np
import ast

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from utils import perform_oversampling


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
    e = pd.read_csv('all_events_orig_bak.csv', nrows=100000)
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


def build_aggregates():
    e = load_events()
    e.type = e.type.str.lower()
    e.chance = e.groupby(by=['match_id', 'possession'])['chance'].transform('any')
    e = e.loc[e['type'].isin(['pass', 'carry'])]
    e = e.loc[e.team == e.possession_team]
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
    return aggregates


def do_plots():
    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'var_pass_length', 'chance')

    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'sum_pass_length', 'chance')

    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'max_pass_speed', 'chance')

    attribute_chance_plot(build_aggregates(), 'sum_progression_pct', 'var_pass_speed', 'chance')

    attribute_chance_plot(build_aggregates(), 'location_x', 'location_y', 'chance')


def classy():
    # Parameters
    aggregates = build_aggregates()

    df = perform_oversampling(aggregates)

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
    return x, y, pd.DataFrame({'actual': y_test, 'pred': y_pred, 'prob': lr_probs})


import numpy as np
from matplotlib import pyplot as plt


def attribute_chance_plot(df, x_value, y_value, label_col):
    df = df[np.abs(df[y_value] - df[y_value].mean()) <= (2 * df[y_value].std())]
    x = df[x_value].array
    y = df[y_value].array
    labels = df[label_col].array
    cdict = {True: 'red', False: 'blue'}

    fig, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x[ix], y[ix], c=cdict[g], label=g, s=50, marker='2')
    ax.legend(labels=['No Chance', 'Chance'])
    plt.title(f'{x_value} vs. {y_value}')
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.show()
