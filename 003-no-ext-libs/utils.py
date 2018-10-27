import datetime
import time
import atexit
import sys
import os
import pandas as pd
from contextlib import contextmanager

from sklearn.metrics import mean_squared_error

from log import *

TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

start_time = time.time()
N_JOBS = 4


def estimate_csv(file_name, nrows=200, test_file_name='test_row_count.csv'):
    """Estimate big csv file params (size, row count, row size)
    :param file_name: csv file to estimate
    :param nrows=200: rows to read. File must has more rows than nrows!
    :param test_file_name = 'test_row_count.csv': name for test file (saved to disk).
    :return: tuple(
        rows count estimation,
        single row size estimation,
        total memory usage estimation)
    """

    file_size = os.path.getsize(file_name)
    df = pd.read_csv(file_name, nrows=nrows)
    df.to_csv(test_file_name, header=False)
    row_size = os.path.getsize(test_file_name) / nrows

    rows = file_size / row_size
    size = rows * sys.getsizeof(df) / nrows
    return {'rows': int(rows), 'row_size': int(row_size), 'total_size': int(size)}


def read_csv(file_name, nrows):
    if nrows == -1:
        nrows = None

    log('file {}'.format(file_name))
    df = pd.read_csv(file_name, low_memory=False, nrows=nrows)
    log('dataset shape: {}, nrows: {})'.format(df.shape, nrows))
    return df


def optimize_dataframe(df):
    """Optimize pandas dataframe size:
    - downcast numeric (int and float) types columns.
    - convert to Categorical type categorical columns with 2x or more "values/unique" values rate.
    :param df:
    :return:
    """

    # return df  # TODO: remove - check for failure!!!

    int_cols = []
    float_cols = []
    category_cols = []
    other_cols = []

    old_size = sys.getsizeof(df)

    for col_name in df.columns:
        col_type = df.dtypes[col_name]

        if col_type in ['int', 'int32', 'int64']:
            int_cols.append(col_name)
        elif col_type in ['float', 'float32', 'float64']:
            float_cols.append(col_name)
        elif col_type == 'object':
            total = len(df[col_name])
            n_uniq = df[col_name].nunique()
            if n_uniq / total < 0.5:
                category_cols.append(col_name)
            else:
                other_cols.append(col_name)
        else:
            other_cols.append(col_name)

    df_opt = pd.DataFrame()

    if len(int_cols) > 0:
        df_opt[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')

    if len(float_cols) > 0:
        df_opt[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')

    if len(category_cols) > 0:
        df_opt[category_cols] = df[category_cols].astype('category')

    if len(other_cols) > 0:
        df_opt[other_cols] = df[other_cols]

    new_size = sys.getsizeof(df_opt)
    log('optimize dataframe ({} to {}, ratio: {})'.format(old_size, new_size, round(old_size/new_size, 2)))

    return df


def neg_mean_squared_error(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)

