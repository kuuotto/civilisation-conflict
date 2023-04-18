def count_streaks(data):
    """
    Counts the streaks from time stamps.
    data should be a pandas Series which contains the 
    time stamps of the events of interest.
    """
    if len(data) == 0:
        return dict()

    prev_t = data[0]
    streak = 1
    streaks = dict()
    for t in data[1:]:
        if t == prev_t + 1:
            # streak continues
            streak += 1
        else:
            #streak broken
            if streak not in streaks.keys():
                streaks[streak] = 1
            else:
                streaks[streak] += 1
            streak = 1
        prev_t = t

    # add final streak
    if streak not in streaks.keys():
        streaks[streak] = 1
    else:
        streaks[streak] += 1

    return streaks
