# -*- coding: utf-8 -*-
"""
Created on Tue May 27 22:26:32 2025

@author: Administrator
"""

import pandas as pd
import numpy as np

def introduce_missing_values(df, missing_fraction=0.1, seed=None):
    """
    Randomly introduce missing values into a DataFrame.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        missing_fraction (float): Fraction of total cells to make missing (between 0 and 1).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: New DataFrame with missing values introduced.
    """
    if seed is not None:
        np.random.seed(seed)

    df_copy = df.copy()
    total_cells = df_copy.size
    num_missing = int(total_cells * missing_fraction)

    for _ in range(num_missing):
        i = np.random.randint(0, df_copy.shape[0])
        j = np.random.randint(0, df_copy.shape[1])
        df_copy.iat[i, j] = np.nan

    return df_copy
