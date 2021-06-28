import datetime
import pandas as pd
from statsbombpy import sb


def main():
    match = 2275038
    events = sb.events(match_id=match)

    events['timestamp'] = pd.to_datetime(events['timestamp'])
    chances = events[events.shot.isna()==False].timestamp
    frame = []
    for end in chances.to_list():
        start = end - datetime.timedelta(seconds=30)
        mask = (events['timestamp'] > start) & (events['timestamp'] <= end)
        frame.append(events.loc[mask])
    return frame
