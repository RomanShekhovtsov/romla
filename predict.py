import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_datetime_features, log

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

def predict(args):
    metrics = dict()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    start_time = time.time()
    df = pd.read_csv(args.test_csv)
    log('Dataset read, shape {}'.format(df.shape))
    metrics['predict.rows'] = df.shape[0]
    metrics['predict.cols'] = df.shape[1]
    metrics['predict.read_csv'] = time.time() - start_time
    log('Read .csv time: {}'.format( metrics['predict.read_csv'] ))

    start_time = time.time()
    # features from datetime
    df = transform_datetime_features(df)

    # missing values
    if model_config['missing']:
        df.fillna(-1, inplace=True)
    elif any(df.isnull()):
        df.fillna(value=df.mean(axis=0), inplace=True)

    # categorical encoding
    for col_name, unique_values in model_config['categorical_values'].items():
        for unique_value in unique_values:
            df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)

    # filter columns
    used_columns = model_config['used_columns']

    # scale
    X_scaled = model_config['scaler'].transform(df[used_columns])

    metrics['predict.preprocessing_time'] =  time.time() - start_time
    log('Preprocessing time: {}'.format( metrics['predict.preprocessing_time'] ))

    start_time = time.time()
    model = model_config['model']
    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(X_scaled)
    elif model_config['mode'] == 'classification':
        df['prediction'] = model.predict(X_scaled)

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    metrics['predict.prediction_time'] = time.time() - start_time
    log('Prediction time: {}'.format( metrics['predict.prediction_time'] ))
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    predict(args)
