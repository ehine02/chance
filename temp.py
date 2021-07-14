import datetime
import pandas as pd
import numpy as np
from statsbombpy import sb


def main():
    match = 18245
    events = sb.events(match_id=match)
    events.to_csv(str(match) + '.csv')

    events = events[events['period']==1]
    events = events.join(pd.json_normalize(events['pass']), rsuffix='_pass')
    events = events.join(pd.json_normalize(events['shot']), rsuffix='_shot')
    events = events.set_index(pd.DatetimeIndex(events['timestamp']))
    events = events.sort_index()
    return events

