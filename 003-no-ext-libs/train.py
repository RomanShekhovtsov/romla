import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from predict import preprocess_test_data, predict
from scorer import score, read_test_target
from utils import transform_datetime_features, log, log_start, log_time

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

ONEHOT_MAX_UNIQUE_VALUES = 20

def train(args):
    start_train_time = time.time()

    if args.nrows == '':
        nrows = None
    else:
        nrows = int(args.nrows)

    log_start()
    df = pd.read_csv(args.train_csv, low_memory=False, nrows=nrows)
    df_y = df.target
    df_X = df.drop('target', axis=1)
    log_time('read dataset, shape: ', df_X.shape, 'nrows', nrows)

    # dict with data necessary to make predictions
    model_config = {}

    # features from datetime
    log_start()
    df_X = transform_datetime_features(df_X)
    log_time('features from datetime')

    # missing values
    log_start()
    if any(df_X.isnull()):
        model_config['missing'] = True
        df_X.fillna(-1, inplace=True)
    log_time('missing values')

    # categorical encoding
    log_start()
    categorical_values = {}
    for col_name in list(df_X.columns):
        col_unique_values = df_X[col_name].unique()
        if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)
    model_config['categorical_values'] = categorical_values
    log_time('categorical encoding')

    # drop constant features
    log_start()
    constant_columns = [
        col_name
        for col_name in df_X.columns
        if df_X[col_name].nunique() == 1
    ]
    df_X.drop(constant_columns, axis=1, inplace=True)
    log_time('drop constant features')

    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number') or col_name.startswith('onehot')
    ]

    df_X = df_X[used_columns]
    model_config['used_columns'] = used_columns

    # scaling
    log_start()
    scaler = StandardScaler()
    df_X = scaler.fit_transform(df_X)
    model_config['scaler'] = scaler
    log_time('scaling')

    # fitting
    log_start()
    model_config['mode'] = args.mode
    regression = ( args.mode == 'regression' )
    if regression:
        # model = Ridge()
        models = [
            GradientBoostingRegressor(),
            Ridge()
        ]
        # scoring = 'neg_mean_squared_error'
    else:
        # model = LogisticRegression()
        models = [
            GradientBoostingClassifier(),
            LogisticRegression()
        ]
        # scoring = 'roc_auc'

    #    param_grid = {
    #        'min_samples_leaf': range(1, 30),
    #        'max_features': np.arange(0.1, 1.1, 0.1),
    #    }

    #    cv = KFold( n_splits=5, shuffle=True)
    #    grid = GridSearchCV( model, param_grid, cv=cv, scoring=scoring, n_jobs=3)
    #    grid.fit(df_X, df_y)
    scores = []
    X_test_scaled, _, _ = preprocess_test_data(args, model_config)
    y_test = read_test_target(args)
    for model in models:
        model.fit(df_X, df_y)
        prediction = predict(X_test_scaled, model)
        scores.append( score(args, y_test, prediction ) )
    log_time('grid fitting')

    if regression:
        best_index =  np.argmin(scores)
    else:
        best_index = np.argmax(scores)

    #model = grid.best_estimator_
    model_config['model'] = models[best_index]

    log('Train time: {}'.format(time.time() - start_train_time))
    return model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    model_config = train(args)
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)
