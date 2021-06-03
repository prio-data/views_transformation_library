"""
Transformations for handling missing values, such as simple replacements, and
more advanced extrapolations.
"""
import numpy as np
import pandas as pd

def replace_na(df: pd.DataFrame, replacement = 0):
    return df.replace(np.nan,replacement)

