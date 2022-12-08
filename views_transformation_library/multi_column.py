import numpy as np
import pandas as pd


def add_column(df):

    df_left, df_right = get_left_right(df)

    df_sum = pd.DataFrame(index=df_left.index)

    df_sum[df_left.columns[0]] = df_left.values + df_right.values

    return df_sum


def subtract_column(df):

    df_left, df_right = get_left_right(df)

    df_diff = pd.DataFrame(index=df_left.index)

    df_diff[df_left.columns[0]] = df_left.values - df_right.values

    return df_diff


def multiply_column(df):

    df_left, df_right = get_left_right(df)

    df_mult = pd.DataFrame(index=df_left.index)

    df_mult[df_left.columns[0]] = df_left.values * df_right.values

    return df_mult


def divide_column(df):

    df_left, df_right = get_left_right(df)

    df_div = pd.DataFrame(index=df_left.index)

    df_div[df_left.columns[0]] = df_left.values / df_right.values

    return df_div


def get_left_right(df):

    assert len(df.columns) == 2

    left_column = df.columns[0]

    right_column = df.columns[1]

    df_left = pd.DataFrame(df[left_column])

    df_right = pd.DataFrame(df[right_column])

    return df_left, df_right
