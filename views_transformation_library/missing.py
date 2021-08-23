""" Missing data filling functionality 

Transformations for handling missing values, such as simple replacements, and
more advanced extrapolations.

"""

from typing import Any, List, Optional, Literal
import multiprocessing as mp

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from sklearn.experimental import enable_iterative_imputer  # type: ignore # noqa: F401, E501 # pylint: disable=unused-import
from sklearn.impute import IterativeImputer  # type: ignore
from sklearn.linear_model import BayesianRidge  # type: ignore

def replace_na(df: pd.DataFrame, replacement = 0):
    return df.replace(np.nan,replacement)

def list_totally_missing(df: pd.DataFrame) -> List[str]:
    """ Get a list of columns for which all values are missing """

    cols = []
    for col in df:
        if df[col].isnull().mean() == 1.0:
            cols.append(col)

    return cols



def fill_groups_with_time_means(df: pd.DataFrame) -> pd.DataFrame:
    """ Fill completely missing groups with time means """

    # Only fill numeric cols
    cols = list(df.select_dtypes(include=[np.number]).columns.values)
    for _, g_df in df.groupby(level=1):
        # If missing everything from a group
        if g_df.isnull().all().all():
            # Get the times for this group
            times_group = g_df.index.get_level_values(0)
            # Fill all columns with the time mean
            df.loc[g_df.index, cols] = (
                df.loc[times_group, cols].groupby(level=0).mean().values
            )
    return df



def fill_with_group_and_global_means(df: pd.DataFrame) -> pd.DataFrame:
    """ Impute missing values to group-level or global means. """

    for col in df.columns:
        # impute with group level mean
        df[col].fillna(
            df.groupby(level=1)[col].transform("mean"), inplace=True
        )
        # fill remaining NaN with df level mean
        df[col].fillna(df[col].mean(), inplace=True)

    return df



def extrapolate(
    df: pd.DataFrame,
    limit_direction: str = "both",
    limit_area: Optional[str] = None,
) -> pd.DataFrame:
    """ Interpolate and extrapolate """
    return (
        df.sort_index()
        .groupby(level=1)
        .apply(
            lambda group: group.interpolate(
                limit_direction=limit_direction, limit_area=limit_area
            )
        )
    )

def _fill_by_group(
    group: Any,
    limit_direction: Literal["forward", "backward", "both"],
    limit_area: Optional[Literal["inside", "outside"]],
) -> Any:
    # Get the outer boundaries of the group data.
    first_id = group.first_valid_index()
    last_id = group.last_valid_index()
    # Fill group according to set params.
    if limit_area is not None:
        # Assume forward if default "both" is passed with area "inside".
        if limit_area == "inside" and limit_direction != "backward":
            group[first_id:last_id] = group[first_id:last_id].ffill()
        if limit_area == "inside" and limit_direction == "backward":
            group[first_id:last_id] = group[first_id:last_id].bfill()
        if limit_area == "outside":

            id_min, id_max = group.index.min(), group.index.max()
            group[id_min:first_id] = group[id_min:first_id].bfill()
            group[last_id:id_max] = group[last_id:id_max].ffill()
    elif limit_direction == "forward":
        group = group.ffill()
    elif limit_direction == "backward":
        group = group.bfill()
    else:
        group = group.ffill().bfill()

    return group



def fill(
    s: pd.Series,
    limit_direction: Literal["forward", "backward", "both"] = "both",
    limit_area: Optional[Literal["inside", "outside"]] = None,
) -> pd.Series:
    """ Fill column in dataframe with optional direction and area.

    Args:
        s: Pandas series to apply filling to.
        limit_direction: Direction in which to fill.
        limit_area: Area to fill. Default None refers to the entire series.
    """

    
    return (
        s.sort_index()
        .groupby(level=1)
        .apply(
            lambda group: _fill_by_group(
                group=group,
                limit_direction=limit_direction,
                limit_area=limit_area,
            ),
        )
    )



def _fill_iterative(
    df: pd.DataFrame,
    seed: int = 1,
    max_iter: int = 10,
    estimator: Any = BayesianRidge(),
):
    """ Gets a single imputation using IterativeImputer from sklearn.

    Uses BayesianRidge() from sklearn.

    Changed default of sample_posterior to True as we're doing
    multiple imputation.

    Clips imputed values to min-max of observed values to avoid
    brokenly large values. When imputation model doesn't converge
    nicely we otherwise end up with extreme values that are out of
    range of the float32 type used by model training, causing crashes.
    Consider this clipping a workaround until a more robust imputation
    strategy is in place.

    """
    # Only impute numberic cols
    cols_numeric = list(df.select_dtypes(include=[np.number]).columns.values)
    cols_not_numeric = [col for col in df.columns if col not in cols_numeric]

    # Get bounds so we can clip imputed values to not be outside
    # observed values
    observed_min = df[cols_numeric].min()
    observed_max = df[cols_numeric].max()

    df_imputed = df.loc[:, []].copy()
    for col in df:
        df_imputed[col] = np.nan

    df_imputed[cols_numeric] = IterativeImputer(
        random_state=seed, max_iter=max_iter, estimator=estimator
    ).fit_transform(df[cols_numeric])
    df_imputed[cols_not_numeric] = df[cols_not_numeric]

    # Clip imputed values to observed min-max range
    df_imputed[cols_numeric] = df_imputed[cols_numeric].clip(
        observed_min, observed_max, axis=1
    )

    return df_imputed


def impute_mice_generator(
    df, n_imp, estimator=None, parallel=False, n_jobs=mp.cpu_count()
):
    """ Impute df with MICE """

    if parallel:
        with mp.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
            results = [
                pool.apply_async(_fill_iterative, (df, imp, 10, estimator,))
                for imp in range(n_imp)
            ]
            for result in results:
                yield result.get()

    else:

        for imp in range(n_imp):

            yield _fill_iterative(df, seed=imp, estimator=estimator)
