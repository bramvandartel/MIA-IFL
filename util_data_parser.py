import os.path

import pandas as pd
import numpy as np


# This can be used when parsing data from pandas to a huggingface dataset.

def handle_pandas(features: pd.DataFrame, labels: pd.DataFrame, partition_col=None):
    labels = labels.to_numpy().transpose()

    if not partition_col:
        features = features.fillna(-1)
        features = features.to_numpy()
        features = features.astype(np.int64)
        concatenated_features = ['[' + ', '.join(map(str, row)) + ']' for row in features]
        df = pd.DataFrame({'img': concatenated_features, 'label': labels})
    else:
        partition = features[partition_col]
        features = features.drop(columns=[partition_col])
        features = features.fillna(-1)
        features = features.to_numpy()
        features = features.astype(np.int64)
        concatenated_features = ['[' + ', '.join(map(str, row)) + ']' for row in features]
        df = pd.DataFrame({'img': concatenated_features, 'label': labels, 'partition': partition})

    return df


def encode_features(dataframe, columns, one_hot=False):
    for col in columns:
        dataframe.loc[:, col] = dataframe[col].astype('category').cat.codes
        if one_hot and len(dataframe[col].unique()) > 2:
            dataframe = pd.get_dummies(dataframe, columns=[col])
            dataframe = dataframe.replace({True: 1, False: 0})
    return dataframe


def get_string_columns(df):
    cols = []
    for col in df.columns:
        if df[col].dtype == object:
            cols.append(col)
    return cols


def store_csv(df, name):
    pth = os.path.join(f"data/{name}")
    os.makedirs(pth, exist_ok=True)
    df.to_csv(pth + "/data.csv", index=False)
