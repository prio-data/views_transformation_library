import numpy as np
import pandas as pd


def add_column(df_left, df_right):

    assert(validate_dfs(df_left, df_right))

    df_sum = pd.DataFrame(index = df_left.index)

    df_sum['sum'] = df_left.values + df_right.values

    return df_sum


def subtract_column(df_left, df_right):

    assert (validate_dfs(df_left, df_right))

    df_diff = pd.DataFrame(index=df_left.index)

    df_diff['diff'] = df_left.values - df_right.values

    return df_diff


def multiply_column(df_left, df_right):

    assert (validate_dfs(df_left, df_right))

    df_mult = pd.DataFrame(index=df_left.index)

    df_mult['mult'] = df_left.values * df_right.values

    return df_mult


def divide_column(df_left, df_right):

    assert (validate_dfs(df_left, df_right))

    df_div = pd.DataFrame(index=df_left.index)

    df_div['div'] = df_left.values / df_right.values

    return df_div


def validate_dfs(df_left, df_right):

    passed = True

    if df_left.index.names != df_right.index.names:
        passed = False

    if not(np.array_equal(df_left.index.to_numpy(), df_right.index.to_numpy())):
        passed = False

    return passed
