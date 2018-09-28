import argparse
import os
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import log, log_time

ONEHOT_MAX_UNIQUE_VALUES = 20


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
    return {'train.constant_columns': len(constant_columns)}


def train(args):
    metrics = dict()

    hyper_params_corr_limit = 0.95  # columns with correlation module greater then corr_limit, will be removed

    metrics['hyper_params.corr_limit'] = hyper_params_corr_limit

    preprocessing_start_time = time.time()

    start_time = time.time()
    df = read_data(args)
    df_y = df.target
    df_X = df.drop('target', axis=1)
    metrics['train.read_csv'] = time.time() - start_time
    log('Dataset read, shape {}'.format(df_X.shape))
    log_time('Read train file', start_time)

    metrics['train.rows'] = df_X.shape[0]
    metrics['train.cols'] = df_X.shape[1]

    log('ONEHOT_MAX_UNIQUE_VALUES', ONEHOT_MAX_UNIQUE_VALUES)
    metrics['train.ONEHOT_MAX_UNIQUE_VALUES'] = ONEHOT_MAX_UNIQUE_VALUES

    # dict with data necessary to make predictions
    model_config = {}

    start_time = time.time()
    # features from datetime
    df_X = transform_datetime_features(df_X)
    log_time('transform_datetime_features()', start_time)

    # categorical encoding
    start_time = time.time()
    categorical_values = {}
    for col_name in list(df_X.columns):
        col_unique_values = df_X[col_name].unique()
        if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)
    model_config['categorical_values'] = categorical_values
    metrics['train.categorical_values.count'] = len(categorical_values)
    log_time('onehot encoding', start_time)

    # drop constant features
    start_time = time.time()
    metrics.update(drop_const_features(df_X))
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

    # PCA

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
    metrics['train.used_columns_count'] = len(used_columns)

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
        metrics['train.onehot_columns_count'] = len(onehot_columns)

        number_columns = [
            col_name
            for col_name in df_X.columns
            if col_name.startswith('number')
        ]
        metrics['train.number_columns_count'] = len(number_columns)
        """
        for col_name in onehot_columns:
            most_freq_value = df_X[col_name].value_counts().index[0]
            df_X[col_name].fillna( most_freq_value, inplace=True)

        for col_name in number_columns:
            df_X[col_name].fillna( df_X[col_name].max(), inplace=True)
        """
        df_X.fillna(-1, inplace=True)
        metrics['train.fillna_type'] = 'df_X.fillna(-1)'
    log_time('impute missing values', start_time)

    # scaling
    start_time = time.time()
    scaler = StandardScaler()
    df_X = scaler.fit_transform(df_X)
    model_config['scaler'] = scaler
    log_time('scaling', start_time)

    metrics['train.preprocessing_time'] = time.time() - preprocessing_start_time
    log('Total preprocessing', preprocessing_start_time)

    start_time = time.time()

    # fitting
    model_config['mode'] = args.mode
    if args.mode == 'regression':
        model = xgboost.XGBRegressor(n_jobs=4)
        metrics['model'] = 'XGBRegressor'
    else:
        model = xgboost.XGBClassifier(n_jobs=4)
        metrics['model'] = 'XGBClassifier'

    model.fit(df_X, df_y)
    model_config['model'] = model

    metrics['train.train_time'] = time.time() - start_time
    log('Train', start_time)

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()
    train(args)