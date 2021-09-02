import ast

import numpy as np
import pandas as pd


def perform_oversampling(unbalanced, target='chance'):
    data = unbalanced.copy()
    # Oversampling performed here
    # first count the records of the majority
    majority_count = data[target].value_counts().max()
    working = [data]
    # group by the target
    for _, grouped in data.groupby(target):
        # append N samples to working list where N is the difference between majority and this band
        working.append(grouped.sample(majority_count - len(grouped), replace=True))
    # add the working list contents to the overall dataframe
    balanced = pd.concat(working)
    return balanced


def euclidean_distance(start, end):
    return np.sqrt(np.power(end[0] - start[0], 2) + np.power(end[1] - start[1], 2))


def list_if_not_nan(x):
    if x is np.nan:
        return np.nan
    else:
        return ast.literal_eval(x)


def split_location(x):
    if x is not np.nan:
        return round(x[0], 0), round(x[1], 0)
    return np.nan, np.nan


def location_to_text(row):
    if row.location_x < 60:
        return 'ownhalf'
    if row.location_x < 80:
        return 'midfield'
    if row.location_y >= 62:
        return 'rightwing'
    if row.location_y <= 18:
        return 'leftwing'
    if row.location_x > 102:
        return 'box'
    return 'boxedge'


def names_v1():
    return ['timestamp', 'carry', 'counterpress', 'dribble', 'duration', 'index', 'location', 'pass',
                                 'period', 'play_pattern', 'player', 'possession', 'possession_team', 'shot', 'team', 'type',
                                 'under_pressure', 'length', 'angle', 'end_location', 'recipient.name', 'height.id',
                                 'height.name',
                                 'type.id', 'type.name', 'body_part.id', 'body_part.name', 'switch', 'through_ball',
                                 'shot_assist']
                                 #'inswinging', 'cut_back', 'miscommunication']]


def names_v2():
    return [
        #'50_50',
        'ball_receipt_outcome',
        #'ball_recovery_recovery_failure',
        #'block_deflection',
        #'block_offensive',
        'carry_end_location',
        #'clearance_aerial_won',
        #'clearance_body_part',
        #'clearance_head',
        #'clearance_left_foot',
        #'clearance_other',
        #'clearance_right_foot',
        'counterpress',
        #'dribble_nutmeg',
        'dribble_outcome',
        #'dribble_overrun',
        'duel_outcome',
        'duel_type',
        'duration',
        #'foul_committed_advantage',
        #'foul_committed_offensive',
        #'foul_committed_penalty',
        #'foul_committed_type',
        #'foul_won_advantage',
        #'foul_won_defensive',
        #'goalkeeper_body_part',
        #'goalkeeper_end_location',
        #'goalkeeper_outcome',
        #'goalkeeper_position',
        #'goalkeeper_punched_out',
        #'goalkeeper_technique',
        #'goalkeeper_type',
        'id',
        'index',
        'interception_outcome',
        'location',
        'match_id',
        'minute',
        #'off_camera',
        #'out',
        #'pass_aerial_won',
        'pass_angle',
        'pass_assisted_shot_id',
        'pass_body_part',
        'pass_cross',
        #'pass_cut_back',
        #'pass_deflected',
        'pass_end_location',
        #'pass_goal_assist',
        'pass_height',
        #'pass_inswinging',
        'pass_length',
        'pass_outcome',
        #'pass_outswinging',
        'pass_recipient',
        #'pass_shot_assist',
        #'pass_straight',
        #'pass_switch',
        #'pass_technique',
        #'pass_through_ball',
        'pass_type',
        'period',
        'play_pattern',
        'player',
        'position',
        'possession',
        'possession_team',
        'related_events',
        'second',
        #'shot_aerial_won',
        'shot_body_part',
        'shot_end_location',
        #'shot_first_time',
        'shot_freeze_frame',
        'shot_key_pass_id',
        'shot_outcome',
        'shot_statsbomb_xg',
        'shot_technique',
        'shot_type',
        'substitution_outcome',
        'substitution_replacement',
        'tactics',
        'timestamp',
        'team',
        'type',
        'under_pressure'
    ]

