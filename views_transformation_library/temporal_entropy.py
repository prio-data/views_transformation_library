import numpy as np
import pandas as pd

from views_transformation_library import utilities


def get_temporal_entropy(
        df,
        window,
        offset=0.

):
    """"
    get_temporal_entropy created 04/03/2022 by Jim Dale

    Computed entropy along the time axis within a window of length specified by 'window'.

    The entropy of a feature x over a window of length w is

    sum_(i=1,w) (x_i/X)log_2(x_i/X) where X = sum_(i=1,w) (x_i)

    Arguments:

    df:                a dataframe of series to be splagged

    window:            integer size of window

    offset:            datasets containing mostly zeros will return
                       NaNs or Infs for entropy most or all of the time.
                       Since this is unlikely to be desirable, an
                       offset can be added to all feature values. so
                       that sensible values for entropy are returned.

    Returns:

    A df containing the entropy computed for all times for all columns

    """
    df = df.fillna(0.0)
    if not df.index.is_monotonic:
        df = df.sort_index()

    df_index = df.index

    times, time_to_index, index_to_time = utilities._map_times(df)

    features = utilities._map_features(df)

    pgids, pgid_to_index, index_to_pgid = utilities._map_pgids_1d(df)

    tensor3d = utilities._df_to_tensor_strides(df)

    tensor3d += offset

    sum_over_window = np.zeros_like(tensor3d)

    entropy = np.zeros_like(tensor3d)

    for itime in range(len(times)):
        if itime < window - 1:
            istart = 0
        else:
            istart = itime - window + 1

        sum_over_window[itime, :, :] = np.sum(tensor3d[istart:itime+1], axis=0)

        entropy[itime, :, :] = -np.sum(tensor3d[istart:itime+1]/sum_over_window[itime, :, :] *
                                       np.log2(tensor3d[istart:itime+1]/sum_over_window[itime, :, :]), axis=0)
#    individual_values = (tensor3d / sum_over_window) * np.log2(tensor3d / sum_over_window)

#    for itime in range(len(times)):
#        if itime < window -1 :
#            istart = 0
#        else:
#            istart = itime - window + 1


#        print(individual_values[istart:itime+1])
#        entropy[itime, :, :] = -np.sum(individual_values[istart:itime+1], axis=0)

    df_entropy = entropy_to_df_strides(entropy,times,pgids,features,df_index)

    return df_entropy


def entropy_to_df_strides(
        entropy,
        times,
        pgids,
        features,
        df_index
):
    dim0, dim1, dim2 = len(times), len(pgids), len(features)

    entropy_strides = entropy.strides

    offset2 = entropy_strides[2]

    offset1 = entropy_strides[1]

    offset0 = entropy_strides[0]

    flat = np.lib.stride_tricks.as_strided(entropy, shape=(dim0 * dim1, dim2),
                                           strides=(offset1, offset2))

#    df_column_names = ['entropy_'+ feature for feature in features]

    df_column_names = df.columns

    df_entropy = pd.DataFrame(flat, index=df_index, columns=df_column_names)

    return df_entropy
