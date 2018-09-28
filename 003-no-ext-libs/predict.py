import argparse
import os
import pandas as pd
import pickle
import time

from utils import transform_datetime_features, log, log_start, log_time

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

def preprocess_test_data(args, model_config=None):

    if model_config is None:
        # load model
        model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
        with open(model_config_filename, 'rb') as fin:
            model_config = pickle.load(fin)

    if args.nrows == '':
        nrows = None
    else:
        nrows = int(args.nrows)

    # read dataset
    log_start()
    df = pd.read_csv(args.test_csv, low_memory=False, nrows=nrows)
    log('Dataset read, shape {}'.format(df.shape))
    log_time('read dataset')

    line_ids = df[['line_id']]
    df.drop('line_id', axis=1, inplace=True)

    # features from datetime
    log_start()
    df = transform_datetime_features(df)
    log_time('features from datetime')

    # missing values
    log_start()
    if model_config['missing']:
        df.fillna(-1, inplace=True)
    elif any(df.isnull()):
        df.fillna(value=df.mean(axis=0), inplace=True)
    log_time('missing values')

    # categorical encoding
    log_start()
    for col_name, unique_values in model_config['categorical_values'].items():
        for unique_value in unique_values:
            df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)
    log_time('categorical encoding')

    # filter columns
    used_columns = model_config['used_columns']
    df = df[used_columns]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_predict_time = time.time()

    X_scaled, line_ids, model_config = preprocess_test_data(args)
    model = model_config['model']
    prediction = predict(X_scaled, model)

    save_prediction(args, line_ids, prediction)

    log('Prediction time: {}'.format(time.time() - start_predict_time))