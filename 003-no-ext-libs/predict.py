import argparse
import os
import sys
import pandas as pd
import pickle
import time

from utils import *

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
start_predict_time = time.time()

def preprocess_test_data(args, model_config=None):

    if model_config is None:
        # load model
        model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
        with open(model_config_filename, 'rb') as fin:
            model_config = pickle.load(fin)

    is_big = model_config['is_big']

    # read dataset
    log_start()
    df = read_csv(args.test_csv, args.nrows)

    # missing values
    log_start()
    if model_config['missing']:
        df.fillna(-1, inplace=True)
    elif df.isnull().values.any():
        df.fillna(value=-1, inplace=True)
    log_time('impute missing values')

    df = optimize_dataframe(df)

    line_ids = df['line_id']
    df.drop('line_id', axis=1, inplace=True)

    if not is_big:
        # features from datetime
        log_start()
        df_dates = transform_datetime_features(df)
        log_time('features from datetime ({} columns)'.format(len(df_dates.columns)))

        log_start()
        if df_dates.isnull().values.any():
            model_config['missing_dates'] = True
            df_dates.fillna(-1, inplace=True)
        log_time('missing dates values')

        df_dates = optimize_dataframe(df_dates)
        df = pd.concat( (df, df_dates), axis=1 )
        df_dates = None

    # categorical encoding
    log_start()
    df_cat = pd.DataFrame()
    for col_name, unique_values in model_config['categorical_values'].items():
     for unique_value in unique_values:
         df_cat['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)
    log_time('categorical encoding ({} columns)'.format(len(df_cat.columns)))

    df_cat = optimize_dataframe(df_cat)
    df = pd.concat( (df, df_cat), axis=1 )
    df_cat = None

    # filter columns
    used_columns = model_config['used_columns']
    df = df[used_columns]

    #drop_const_cols(df)

    # scale
    log_start()
    X_scaled = model_config['scaler'].transform(df)
    log_time('scale')

    return X_scaled, line_ids, model_config

def predict(X_scaled, model):
    start_predict_time = time.time()

    log_start()
    prediction = model.predict(X_scaled)
    log_time('predict')
    return prediction


def save_prediction(args, line_ids, prediction):
    log_start()
    df = pd.DataFrame()
    df['line_id'] = line_ids
    df['prediction'] = prediction
    df.to_csv(args.prediction_csv, index=False)
    log_time('save prediction')
    log('PREDICTION TIME: {}'.format(time.time() - start_predict_time))
    log_trail()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--nrows')
    args = parser.parse_args()

    X_scaled, line_ids, model_config = preprocess_test_data(args)
    model = model_config['model']
    prediction = predict(X_scaled, model)

    save_prediction(args, line_ids, prediction)

