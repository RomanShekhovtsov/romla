import argparse
import os
import pandas as pd
import time

from utils import transform_datetime_features, parse_dt, log, log_time
from sklearn.decomposition import PCA

ONEHOT_MAX_UNIQUE_VALUES = 20
metrics = dict()
model_config = {}

def read_data(args):
    train_csv = args.train_csv
    file_size = os.path.getsize(train_csv)

    if args.fast_test == 'y':
        nrows = 1000
    else:
        nrows = round(10 ** 7 / df.shape[1], -1)  # empiric coefficient to fit into 16GB

    df = pd.read_csv(train_csv, nrows=1)
    test_row_file_name = 'test_row_size.csv'
    df.to_csv(test_row_file_name, header=False)
    row_size = os.path.getsize(test_row_file_name)
    rows_estimation = file_size / row_size * 0.8  # 20% for estimation error

    if rows_estimation < nrows:
        nrows = None  # cheat pandas

    return pd.read_csv(train_csv, low_memory=False, nrows=nrows)

# drop constant features
def drop_const_features(df):
    constant_columns = [
        col_name
        for col_name in df.columns
        if df[col_name].nunique() == 1
    ]
    df.drop(constant_columns, axis=1, inplace=True)
    metrics['dp.constant_columns'] = len(constant_columns)

def categorical_encoding(df):
    log('ONEHOT_MAX_UNIQUE_VALUES', ONEHOT_MAX_UNIQUE_VALUES)
    metrics['dp.ONEHOT_MAX_UNIQUE_VALUES'] = ONEHOT_MAX_UNIQUE_VALUES

    # categorical encoding
    start_time = time.time()
    categorical_values = {}
    for col_name in list(df.columns):
        col_unique_values = df[col_name].unique()
        if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)
    model_config['categorical_values'] = categorical_values
    metrics['dp.categorical_values.count'] = len(categorical_values)
    log_time('onehot encoding', start_time)

def PCA( df ):
    # PCA
    pca = PCA(copy=False)
    pca.fit_transform(df)
    expl_var_ratio = pca.explained_variance_ratio_

    i = 0
    sum_ratio = 0
    for x in expl_var_ratio:
        sum_ratio += x
        i += 1
        if sum_ratio > .99:
            break

    model_config['PCA'] = pca
    model_config['PCA_components'] = i

def preproccess_data(args):

    data_preprocessing_start_time = time.time()

    start_time = time.time()
    df = read_data(args)
    df_y = df.target
    df_X = df.drop('target', axis=1)
    metrics['dp.read_csv'] = time.time() - start_time
    log('Train dataset read, shape {}'.format(df_X.shape))
    log_time('Read train file', start_time)

    metrics['dp.rows'] = df_X.shape[0]
    metrics['dp.cols'] = df_X.shape[1]

    # dict with data necessary to make predictions
    start_time = time.time()
    # features from datetime
    df_X = transform_datetime_features(df_X)
    log_time('transform_datetime_features()', start_time)

    categorical_encoding(df_X)

    # drop constant features
    start_time = time.time()
    drop_const_features(df_X)
    log_time('drop constant features', start_time)

    # use only numeric columns
    start_time = time.time()
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number') or col_name.startswith('onehot')
    ]
    df_X = df_X[used_columns]
    log_time('drop ununsed columns', start_time)

    """
    start_time = time.time()
    print('detecting correlation >', hyper_params_corr_limit, ':')
    corr_cols = {}
    corr = df_X.corr()

    for i in range(corr.shape[0]):
        for j in range(i, corr.shape[1]):
            v = corr.iloc[i, j]
            if abs(v) > hyper_params_corr_limit and i != j:
                corr_cols[corr.columns[j]] = True
                #print(corr.index[i], corr.columns[j], v)
    print(corr_cols.keys())

    df_X.drop( list(corr_cols.keys()), axis=1, inplace=True )
    used_columns = df_X.columns
    """
    model_config['used_columns'] = used_columns
    metrics['dp.used_columns_count'] = len(used_columns)

    log_time('remove high correlated columns', start_time)

    # missing values
    start_time = time.time()
    if any(df_X.isnull()):
        model_config['missing'] = True

        onehot_columns = [
            col_name
            for col_name in df_X.columns
            if col_name.startswith('onehot')
        ]
        metrics['dp.onehot_columns_count'] = len(onehot_columns)

        number_columns = [
            col_name
            for col_name in df_X.columns
            if col_name.startswith('number')
        ]
        metrics['dp.number_columns_count'] = len(number_columns)
        """
        for col_name in onehot_columns:
            most_freq_value = df_X[col_name].value_counts().index[0]
            df_X[col_name].fillna( most_freq_value, inplace=True)

        for col_name in number_columns:
            df_X[col_name].fillna( df_X[col_name].max(), inplace=True)
        """
        df_X.fillna(-1, inplace=True)
        metrics['dp.fillna_type'] = 'df_X.fillna(-1)'
    log_time('impute missing values', start_time)

    # scaling
    start_time = time.time()
    scaler = StandardScaler()
    df_X = scaler.fit_transform(df_X)
    model_config['scaler'] = scaler
    log_time('scaling', start_time)

    metrics['dp.preprocessing_time'] = time.time() - preprocessing_start_time
    log_time('Total data preprocessing', data_preprocessing_start_time)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()
    preproccess_data(args)