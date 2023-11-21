""" Transform library.

Many functions assume data in the form of a pandas series or dataframe
indexed by timevar as level 0 and groupvar as level 1.

Originally written by Frederick Hoyles and Remco B. Jansen for Views 2
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

def delta(s: pd.Series, time: int = 1) -> pd.Series:
    """
    delta

    Returns the time-delta of the input series s, s(t) - s(t-delta)

    Arguments:

         delta: integer specifying how large the time gap should be (defaults to 1)

    """

    return s - tlag(s, time=time)

def greater_or_equal(s: pd.Series, value: float) -> pd.Series:
    """
    greater_or_equal

    Detects where input series is greater than or equal to a threshold value

    Returns 1 if s >= value, else 0

    Arguments:
         value: float specifying threshold

    """

    mask = s >= value
    y = mask.astype(int)

    return y

def smaller_or_equal(s: pd.Series, value: float) -> pd.Series:
    """
    smaller_or_equal

    Detects where input series is less than or equal to a threshold value

    Returns 1 if s <= value, else 0

    Arguments:
        value: float specifying threshold

    """

    mask = s <= value
    y = mask.astype(int)

    return y

def in_range(s: pd.Series, low: float, high: float) -> pd.Series:
    """
    in_range

    Detects where input series lies between two values.

    Returns 1 if low <= s <= high else 0

    Arguments:
        low: float specifying lower threshold
        high: float specifying higher threshold

    """

    y_high = smaller_or_equal(s, high)
    y_low = greater_or_equal(s, low)
    y = y_high + y_low - 1

    return y

def tlag(s: pd.Series, time: int) -> pd.Series:
    """
    tlag

    Shifts input series backwards in time

    Arguments:
        time: int specifying how many timesteps to shift backwards by

    """
    if time < 1:
        msg = f"Time below 1 passed to tlag: {time} \n"
        msg += "Call tlead() instead \n"
        raise RuntimeError(msg)

    return s.groupby(level=1).shift(time)


def tlead(s: pd.Series, time: int) -> pd.Series:
    """
    tlead

    Shifts input series forwards in time

    Arguments:
        time: int specifying how many timesteps to shift forwards by

    """
    if time < 1:
        msg = f"Time below 1 passed to tlead: {time} \n"
        msg += "Call tlag() instead \n"
        raise RuntimeError(msg)

    return s.groupby(level=1).shift(-time)

def moving_average(s: pd.Series, time: int) -> pd.Series:
    """
    moving_average

    Computes moving average over a specified time window

    Arguments:
         window: integer size of moving time window over which to average

    """

    if time < 1:
        msg = f"Time below 1 passed to ma: {time} \n"
        raise RuntimeError(msg)

    # Groupby groupvar
    y = s.groupby(level=1)
    # Divide into rolling time window of size time
    # min_periods=0 lets the window grow with available data
    # and prevent the function from inducing missingness
    y = y.rolling(time, min_periods=0)
    # Compute the mean
    y = y.mean()
    # groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index()
    return y

def moving_sum(s: pd.Series, time: int) -> pd.Series:
    """
    moving_sum

    Computes moving sum over a specified time window

    Arguments:
         window: integer size of moving time window over which to sum

    """

    if time < 1:
        msg = f"Time below 1 passed to ms: {time} \n"
        raise RuntimeError(msg)

    # Groupby groupvar
    y = s.groupby(level=1)
    # Divide into rolling time window of size time
    # min_periods=0 lets the window grow with available data
    # and prevent the function from inducing missingness
    y = y.rolling(time, min_periods=0)
    # Compute the sum
    y = y.sum()
    # groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index()
    return y


def cweq(s: pd.Series, value: float, seed=None) -> pd.Series:
    """
    cweq

    Moving forwards in time, continue to count the number of timesteps while the input series equals value. If
    the series ceases to be equal to the specified value, reset the count to zero.

    Seed specifies the value of the count at the beginning of the series (when no prior data are available)

    Arguments:
        value: float specifying value of series to follow
        seed: assumed count at beginning of series (defaults to None)

    """

    def set_seed(count, s, seed, mask):
        """Set count=seed in first time if mask was True there

        Example: We want time since conflict, which is time in peace.
        So we want count_while(conflict == 0).
        If our conflict series starts at 0 we might assume some longer
        previous history of peace.
        As the time count is summed cumulatively we can "seed" this
        counting sum with a starting value.

        This seed is therefore inserted into the first time period
        of the count IF the country is in peace at that time.
        Being in peace means the count is True, or ==1 as we
        already cast the masks T/F to the counters 1/0.

        """
        ix_timevar = s.index.get_level_values(0)
        first_time = ix_timevar == min(ix_timevar)
        mask_true = mask == 1
        first_time_where_mask_true = first_time & mask_true
        count.loc[first_time_where_mask_true] = seed
        return count

    # Drop NaN's
    s = s.dropna()

    # Boolean mask of where our condition (s==value) is True
    mask = s == value

    # This is a tricky one, print it out if its confusing.
    # Values of block_grouper are incremented when mask is NOT true.
    # This creates values that are constant (not incrementing) through a
    # consecutive spell of mask being True.
    # Grouping by this var thus lets the count.cumsum() restart for
    # each group of consecutive rows where mask is True and stay at
    # zero for the rows where block_grouper keeps incrementing,
    # which are the rows where mask is not met.
    # Note that mask is True when the criteria is fullfilled
    # Basically lets us assign a grouping id to each consecutive
    # spell of our condition being True.
    block_grouper = (~mask).groupby(level=1).cumsum()

    # Our mask becomes the basis for the count by casting it to int
    count = mask.astype(int)

    if seed:
        count = set_seed(count, s, seed, mask)

    # Get the groupvar-level index to group by
    ix_groupvar = s.index.get_level_values(1)

    # The time elapsed while condition is true
    y = count.groupby([block_grouper, ix_groupvar]).cumsum()
    y = y.astype(int)

    return y

def time_since(s, value=0, seed=None) -> pd.Series:
    """
    time_since

    time since event in series, where an event is where the series devaites from value.

    In order to compute a variable like "time since previous conflict
    event" we must apply a timelag to cweq() to get a series because
    for fitting a simultanous model we do not want the counter to be
    simultaneous to the event.

    Consider the data:

    event  : 0, 0, 1, 1, 0, 0 # Event
    cweq_0 : 1, 2, 0, 0, 1, 2 # count event while equals zero
    tisiev : ., 1, 2, 0, 0, 1 # time since event

    Fitting a model like "event ~ cweq0" makes no sense as cweq0 is
    always 0 if event=1.
    A model like "event ~ tsnp" makes more sense.
    We must apply a time lag to event before computing the counter to
    see how long time has elapsed since the previous event.

    Of course this isn't necessary for OSA modelling where all the
    rhs variables are time-lagged anyway but this is useful for
    dynamic simulation where X and predicted y are simulatenous.

    Arguments:
        value: float specifying value of series to follow
        seed: assumed time_since at beginning of series (defaults to None)

    """

    return cweq(s=tlag(s=s, time=1), value=value, seed=seed)

def decay(s: pd.Series, halflife: float) -> pd.Series:
    """
    decay

    Decay function, returning 2**(-s/halflife)

    See half-life formulation at
    https://en.wikipedia.org/wiki/Exponential_decay

    Arguments:
        halflife: float specifying time over which decay by a factor of 2 occurs

    """

    return 2 ** ((-1 * s) / halflife)

def mean(s: pd.Series) -> pd.Series:
    """
    mean

    Computes the arithmetic mean over time for each spatial unit

    Arguments:
        None

    """

    return s.groupby(level=1).transform("mean")

def ln(s: pd.Series) -> pd.Series:
    """
    ln

    Returns natural log of s+1

    Arguments:
        None

    """

    return np.log1p(s)

def demean(s: pd.Series) -> pd.Series:
    """
    demean

    Computes difference between value and mean of input, grouped by spatial unit

    Returns s - mean_group(s)

    Arguments:
        None

    """

    s_mean = s.groupby(level=1).transform("mean")
    return s - s_mean

def rollmax(s: pd.Series, window: int) -> pd.Series:
    """
    rollmax

    Computes rolling maximum over a specified time window

    Arguments:
         window: integer size of moving time window over which to compute maximum

    """
    # See https://github.com/pandas-dev/pandas/issues/14013
    y = s.groupby(level=1).apply(
        lambda x: x.rolling(window=window, min_periods=0).max()
    )

    return y

def onset_possible(s: pd.Series, window: int) -> pd.Series:
    """
    onset_possible

    Helper function which detects whether an onset (change from zero to non-zero state after at least window zero
    values) is possible. This function detects if no event occured in the preceeding window timesteps

    Arguments:
         window: integer specifying how many zero values must exist before a non-zero value to constitute an onset

    """

    # fillna() is so that the first t in a group is always a possible onset
    return (~rollmax(tlag(s, 1).fillna(0), window).astype(bool)).astype(int)

def onset(s: pd.Series, window: int) -> pd.Series:
    """
    onset

    Computes onsets, where an onset occurs if, given the specified window, an onset is possible, and the value of s is
    non-zero

    Arguments:
         window: integer specifying how many zero values must exist before a non-zero value to constitute an onset

    """
    s_onset_possible = (
        onset_possible(s, window).astype(bool) & s.astype(bool)
    ).astype(int)
    return s_onset_possible

def tick_time_since(s_event: pd.Series, s_time_since: pd.Series) -> np.ndarray:
    """Special time since ticker

    Special case time_since counter.
    In dynasim we need a fast update of time_since variables.
    The normal time_since(s) is very fast for a single pass but for updates
    at each t it is too slow.
    This function uses previously computed time since column and the event
    column to speed up computation.
    Use only in loops over t and pass in only two times at a time, t and t-1
    like so:
    for t in times:
        df.loc["tick_time_since"] = tick_time_since(
            s_event=df.loc[t-1:t, "event"],
            s_time_since=df.loc[t-1:t, "time_since_event"]
        )

    The first t must be at least the THIRD time in the available times
    so that the original "time_since_event" col is defined properly, which
    happens in the second t as it uses a lag (missing in first t)
    and can be lagged by 1.


    """
    # Event from previous time
    event_lag = tlag(s_event, 1).dropna()
    # Time since count from previous time
    ts_lag = tlag(s_time_since, 1).dropna()
    # If no event in previous time, increment the time_since counter
    # If there was an event in previous t, return 0
    # Numpy+pandas magic takes care of the broadcasting when
    # assigning back in with df.loc[t, "tick"]
    # return np.where(~event_lag.astype(bool), ts_lag + 1, 0)
    return np.where(event_lag.astype(bool), 0, ts_lag + 1)
