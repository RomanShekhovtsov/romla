import argparse
import os
import pandas as pd
import pickle
import time

#from cffi import model

from utils import transform_datetime_features

#from auto_ml import Predictor
from auto_ml import utils_models

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

def predict( args ):
    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    df = pd.read_csv(args.test_csv)
    print('Dataset read, shape {}'.format(df.shape))

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

    # scale - in auto_ml
    #X_scaled = model_config['scaler'].transform(df[used_columns])

    column_descriptions = {
        'target': 'output'
    }

    model_names = model_config['model_names']
    file_name = model_config['model_file']
    if 'DeepLearningRegressor' in model_names:
        utils_models.get_model_from_name( 'DeepLearningRegressor' )
    model =  utils_models.load_ml_model(file_name)

    if model_config['mode'] == 'regression':
        type_of_estimator = 'regressor'
        df['prediction'] = model.predict( df[used_columns] )
    elif model_config['mode'] == 'classification':
        type_of_estimator = 'classifier'
        df['prediction'] = model.predict_proba( df[used_columns] )[:, 1]

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    predict( args )
