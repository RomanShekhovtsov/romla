import datetime
import time
import atexit
import sys
import os
import pandas as pd
from contextlib import contextmanager


def parse_dt(x):
    if not isinstance(x, str):
        return datetime.datetime.strptime('0001-01-01', '%Y-%m-%d')
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.strptime('0001-01-01', '%Y-%m-%d')


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]

    df_dates = pd.DataFrame()
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df_dates['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df_dates['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df_dates['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df_dates['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        #df_dates['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        #df_dates['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)

    return df_dates


LOG_FILE = 'logs\\{}.log'.format(time.strftime("%Y-%m-%d_%H", time.localtime()))
log_file = open(LOG_FILE, mode='a')
start_time = -1
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def log(*args):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(time_str, *args)
    print(time_str, *args, file=log_file)
    log_file.flush()


def log_start():
    global start_time
    start_time = time.time()


def log_time(*args):
    log(*args, '[{} sec]'.format(round(time.time() - start_time, 2)))

# @contextmanager
# def log_time(*args):
#     t = time.time()
#     yield
#     log(*args, '[{} sec]'.format(round(time.time() - t, 2)))


def log_trail(char='-', end='\n\n'):
    log(char * 60, '\n\n')


def close_log():
    log_file.close()


atexit.register(close_log)


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
