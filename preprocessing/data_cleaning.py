import numpy as np
import pandas as pd


def clean_data(df):
    print("Starting data cleaning...")

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check missing values
    missing = df.isnull().sum().sum()
    print(f"Total missing values before drop: {missing}")

    # Drop rows with NaN
    df.dropna(inplace=True)

    print("Missing values removed.")

    # Remove duplicates
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]

    print(f"Duplicates removed: {before - after}")

    print("Data cleaning completed.")
    return df