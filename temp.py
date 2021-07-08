import datetime
import pandas as pd
from statsbombpy import sb


def main():
    match = 2275038
    events = sb.events(match_id=match)
    events.to_csv(str(match) + '.csv')

    events = events[events['period']==2]
    events = events.set_index(pd.DatetimeIndex(events['timestamp']))
    events = events.sort_index()
    return events

