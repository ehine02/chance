import datetime
import pandas as pd
from statsbombpy import sb


def main():
    print(sb.api_client.VERSIONS)
    match = 18245
    events = sb.events(match_id=match)
    #events.to_csv(str(match) + '.csv')

    events['timestamp'] = pd.to_datetime(events['timestamp'])
    chances = events[events.shot.isna()==False].loc[:, ['timestamp', 'period']]
    frames = []
    for _, row in chances.iterrows():
        end = row['timestamp']
        period = row['period']
        start = end - datetime.timedelta(seconds=30)
        mask = (events['timestamp'] > start) & (events['timestamp'] <= end) & (events['period']==period)
        frame = events.loc[mask].sort_values('index')
        frames.append(frame)
    return frames
