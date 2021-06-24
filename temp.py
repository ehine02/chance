from statsbombpy import sb


def main():
    comps = sb.competitions()
    #print(comps.head(5))

    comp = 37
    season = 42
    matches = sb.matches(competition_id=comp, season_id=season)
    #print(matches.head(5))
    match = 2275038
    lineup = sb.lineups(match_id=match)['West Ham United LFC']
    #print(lineup.head(10))

    events = sb.events(match_id=match)
    print(events.head(20))
