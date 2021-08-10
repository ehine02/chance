import ast
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def log_reg(x, target='chance'):
    y = x[target]
    x = x.drop(columns=[target])
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
    return model


def xg_events():
    e = pd.read_csv('all_events.csv')
    e = e.loc[~e['shot_type'].isin([np.nan, 'Penalty'])]
    e = e[['location', 'shot_type', 'shot_outcome']]
    e['location'] = e.location.apply(ast.literal_eval)
    e['location_x'] = e['location'].apply(lambda x: round(x[0], 0))
    e['location_y'] = e['location'].apply(lambda x: abs(40 - round(x[1], 0)))
    e['goal'] = e['shot_outcome'] == 'Goal'
    e = e.drop(columns=['location', 'shot_type', 'shot_outcome'])
    return e


def xg_model(events=None):
    return log_reg(events or xg_events(), 'goal')


def xg_map():
    try:
        return pd.read_csv('xg_dist.csv', skiprows=2, names=np.arange(0, 120).tolist(), skip_blank_lines=True)
    except FileNotFoundError:
        m = xg_model(xg_events())
        p = pd.DataFrame(np.zeros([80, 120]) * np.nan)
        for y in range(40):
            for x in range(120):
                xg = m.predict_proba([[x, y]])[0][1]
                p.at[40 + y, x] = xg
                p.at[39 - y, x] = xg
        p.to_csv('xg_dist.csv')
        return p