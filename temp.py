import datetime
import pandas as pd
from statsbombpy import sb


def main():
    match = 2275038
    events = sb.events(match_id=match)
    events.to_csv(str(match) + '.csv')

    #events['timestamp'] = pd.Timestamp(pd.to_datetime(events['timestamp']))
    events = events[events['period']==1]
    events = events.set_index(pd.DatetimeIndex(events['timestamp']))
    events = events.sort_index()
    #print(events.rolling(30).pass_length.sum())
    return events


    # chances = events[events.shot_type.isna()==False].timestamp
    # frame = []
    # for end in chances.to_list():
    #     start = end - datetime.timedelta(seconds=30)
    #     mask = (events['timestamp'] > start) & (events['timestamp'] <= end)
    #     frame.append(events.loc[mask])
    # return frame
