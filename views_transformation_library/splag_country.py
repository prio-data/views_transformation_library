import numpy as np
import pandas as pd
import stepshift
from views_transformation_library import utilities


def get_splag_country(
    df, kernel_inner=1, kernel_width=1, kernel_power=0, norm_kernel=0
):

    """
    get_splag_country

    Performs convolutional spatial lags at the country level.

    Country first-order neighbours are obtained directly from the country_country_month_
    expanded table and represented as a month x country x country tensor.

    n-th order neighbours are obtained iteratively.

    Arguments:

    df:            single-column dataframe containing feature to be transoformed

    kernel_inner:  inner radius of convolution kernel - '1' represents the target
                   country, '2' represents the target country plus its first-order
                   neighbours, and so on

    kernel_outer:  width of convolution kernel - '1' represents (kernel_inner+1)-th
                   order neighbours,  '2' represents (kernel_inner+2)-th order
                   neighbours, and so on

    kernel_power:  countries are weighted by (distance from target country)^kernel_power
                   kernel_power=0 results in unweighted results

    norm_kernel:   if set to 1, the sum of the weights over all neighbours for a given
                   target country is normalised to 1.0

    """

    month_to_index, country_to_index, splag = splag_cm(
        df, kernel_inner, kernel_width, kernel_power, norm_kernel
    )

    df_splag = splags_to_df(
        df,
        splag,
        kernel_inner,
        kernel_width,
        kernel_power,
        norm_kernel,
        month_to_index,
        country_to_index,
    )

    return df_splag


def get_country_neighbours(
    neighbs,
    ninner,
    nouter,
    month_id,
    month_to_index,
    country_to_index,
    index_to_country,
    neighb_tensor_data,
):

    """

    get_country_neighbours

    For a given target country, an inner radius and an outer radius, fetches list of
    countries between the two radii.

    """

    inner_neighbs = neighbs.copy()
    inner_neighbs = get_nth_order_neighbours_from_tensor(
        inner_neighbs,
        month_id,
        ninner,
        month_to_index,
        country_to_index,
        index_to_country,
        neighb_tensor_data,
    )

    outer_neighbs = neighbs.copy()
    outer_neighbs = get_nth_order_neighbours_from_tensor(
        outer_neighbs,
        month_id,
        nouter,
        month_to_index,
        country_to_index,
        index_to_country,
        neighb_tensor_data,
    )

    neighbs = np.sort(list(set(outer_neighbs).difference(set(inner_neighbs))))

    return neighbs


def get_nth_order_neighbours_from_tensor(
    neighbs,
    month_id,
    norder,
    month_to_index,
    country_to_index,
    index_to_country,
    neighb_tensor_data,
):

    """

    get_nth_order_neighbours

    Fetches nth order neighbours around target country *including the target country*

    """

    if norder == 0:
        return neighbs
    else:

        neighbscopy = neighbs.copy()

        month_index = month_to_index[month_id]

        for country in neighbs:

            country_index = country_to_index[country]

            neighb_row = neighb_tensor_data[month_index, country_index, :]

            new_neighb_indices = np.where(neighb_row)[0]

            new_neighbs = [index_to_country[n] for n in new_neighb_indices]

            for newn in new_neighbs:

                if newn not in neighbscopy:

                    neighbscopy.append(newn)

        neighbs = neighbscopy

        norder -= 1

        return get_nth_order_neighbours_from_tensor(
            neighbs,
            month_id,
            norder,
            month_to_index,
            country_to_index,
            index_to_country,
            neighb_tensor_data,
        )


def splag_cm(df, kernel_inner, kernel_width, kernel_power, norm_kernel):

    """

    splag_cm

    Uses neighbours and distances tensors to compute, for every country at every month,
    a spatial lag over the requested sets of neighbours.

    Returns a month x country tensor.

    """

    df_matrix = stepshift.cast.views_format_to_castable(df)

    df_tensor = stepshift.cast.time_unit_feature_cube(df_matrix)

    data_month_ids = df_tensor.coords["time"].values
    data_country_ids = df_tensor.coords["unit"].values

    # build dicts to transform between month and index and country and index in the
    # data, and in the neighbours tensor

    data_month_to_index = {}
    data_index_to_month = {}
    for imonth, month in enumerate(data_month_ids):
        data_month_to_index[month] = imonth

    data_country_to_index = {}
    data_index_to_country = {}
    for icountry, country in enumerate(data_country_ids):
        data_country_to_index[country] = icountry
        data_index_to_country[icountry] = country

    neighb_tensor = utilities._get_country_neighbours_tensor()

    neighb_month_ids = neighb_tensor.coords["month"].values
    neighb_country_ids = neighb_tensor.coords["country"].values

    neighb_month_to_index = {}
    neighb_index_to_month = {}
    for imonth, month in enumerate(neighb_month_ids):
        neighb_month_to_index[month] = imonth

    neighb_country_to_index = {}
    neighb_index_to_country = {}
    for icountry, country in enumerate(neighb_country_ids):
        neighb_country_to_index[country] = icountry
        neighb_index_to_country[icountry] = country

    distances = utilities._get_country_distances(
        data_country_ids, data_country_to_index
    )

    neighb_tensor_data = neighb_tensor.values

    splag = np.zeros_like(df_tensor.values)

    ninner = kernel_inner - 1
    nouter = ninner + kernel_width

    for month_id in data_month_ids:
        data_month_index = data_month_to_index[month_id]
        if month_id in neighb_month_ids:

            for country_id in data_country_ids:
                data_country_index = data_country_to_index[country_id]

                if country_id in neighb_country_ids:
                    neighbs = get_country_neighbours(
                        [
                            country_id,
                        ],
                        ninner,
                        nouter,
                        month_id,
                        neighb_month_to_index,
                        neighb_country_to_index,
                        neighb_index_to_country,
                        neighb_tensor_data,
                    )

                    neighbs = [n for n in neighbs if n in data_country_ids]
                else:
                    neighbs = []

                neighbs_data_indices = [data_country_to_index[n] for n in neighbs]

                vals = df_tensor[data_month_index, neighbs_data_indices, 0].values

                weights = (
                    distances[data_country_index, neighbs_data_indices] ** kernel_power
                )

                if norm_kernel:
                    weights /= np.sum(weights)

                vals[vals == np.inf] = 0.0

                splag[data_month_index, data_country_index] = np.sum(vals * weights)

    return data_month_to_index, data_country_to_index, splag


def splags_to_df(
    df,
    splag,
    kernel_inner,
    kernel_outer,
    kernel_power,
    norm_kernel,
    month_to_index,
    country_to_index,
):

    """

    splags_to_df

    Transforms splag tensor to a dataframe.

    """

    months = np.array(list(idx[0] for idx in df.index.values))

    countries = np.array(list(idx[1] for idx in df.index.values))

    flat = np.empty((len(countries)))

    iflat = 0
    for month, country in zip(months, countries):
        imonth = month_to_index[month]
        icountry = country_to_index[country]

        flat[iflat] = splag[imonth, icountry]

        iflat += 1

    df_index = df.index

    feature = df.columns

    df_column_names = [
        "splag_"
        + str(kernel_inner)
        + "_"
        + str(kernel_outer)
        + "_"
        + str(kernel_power)
        + "_"
        + str(norm_kernel)
        + "_"
        + feature,
    ]

    df_splag = pd.DataFrame(flat, index=df_index, columns=df_column_names)

    return df_splag
