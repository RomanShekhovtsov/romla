import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestClassifier, RandomForestRegressor, \
    ExtraTreesClassifier, ExtraTreesRegressor, \
    AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#from predict import preprocess_test_data, predict
#from scorer import score, read_test_target
from utils import transform_datetime_features, log, log_start, log_time
#from utils import read_csv, optimize_dataframe, estimate_csv
from utils import *

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
N_JOBS =4

ONEHOT_MAX_UNIQUE_VALUES = 20
MAX_DATASET_COLUMNS = 1000
BIG_DATASET_SIZE = 300 * 1024 * 1024 # 300MB
MAX_MODEL_SELECTION_ROWS = 10**5

NEG_MEAN_SQUARED_ERROR = 'neg_mean_squared_error'

def iterate_models( models, X, y, scoring):
    scores = []
    for model in models:
        log_start()
        #TODO: вместо cross-fold использовать fit, если n < 1000
        if y.shape[0] <= 3000:
            cv=2
        else:
            cv=3
        score = np.mean(cross_val_score(model, X, y=y, scoring=scoring, cv=cv, n_jobs=N_JOBS))
        scores.append(np.mean(score))
        if scoring == NEG_MEAN_SQUARED_ERROR:
            score = (-score)**0.5
        log_time('cross validation {} rows (score:{})'.format(y.shape[0], score))
        log(model)
    return scores

def train(args):
    start_train_time = time.time()

    # dict with data necessary to make predictions
    model_config = {}

    # TODO: FAIL CHECK!!!
    is_big = True #os.path.getsize(args.train_csv) > BIG_DATASET_SIZE

    model_config['is_big'] = is_big

    #train_estimate = estimate_csv(args.train_csv)
    #log('estimate', args.train_csv, train_estimate)
    df = read_csv(args.train_csv, args.nrows)

    # missing values
    log_start()
    model_config['missing'] = False
    if df.isnull().values.any():
        model_config['missing'] = True
        df.fillna(-1, inplace=True)
    log_time('impute missing values')

    df = optimize_dataframe(df)

    df_y = df.target
    df_X = df.drop('target', axis=1)
    df = None

    train_rows, train_cols = df_X.shape

    if not is_big:
        # features from datetime
        log_start()
        df_dates = transform_datetime_features(df_X)
        log_time('features from datetime ({} columns)'.format(len(df_dates.columns)))

        # missing values
        log_start()
        if df_dates.isnull().values.any():
            model_config['missing_dates'] = True
            df_dates.fillna(-1, inplace=True)
        log_time('missing dates values')

        #optimize
        df_dates = optimize_dataframe(df_dates)
        df_X = pd.concat( (df_X, df_dates), axis=1 )
        df_dates = None

    # calculate unique values
    log_start()
    df_unique = df_X.apply(lambda x: x.nunique())
    df_const = df_unique[df_unique == 1]
    df_unique = df_unique[df_unique > 2]
    df_unique = df_unique[df_unique <= ONEHOT_MAX_UNIQUE_VALUES]
    df_unique.sort_values(inplace=True)
    log_time('calculate unique values')

    # drop constant features
    df_X.drop(df_const.index, axis=1, inplace=True)

    # categorical encoding
    log_start()
    categorical_values = dict()

    if not is_big:
        df_cat = pd.DataFrame()
        for col_name in df_unique.index:
            if df_X.shape[1] + df_cat.shape[1] + df_unique[col_name] > MAX_DATASET_COLUMNS:
                break

            col_unique_values = df_X[col_name].unique()
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df_cat['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)

        log_time('categorical encoding ({} columns)'.format(len(df_cat.columns)))
        df_cat = optimize_dataframe(df_cat)
        df_X = pd.concat( (df_X, df_cat), axis=1 )
        df_cat = None

    model_config['categorical_values'] = categorical_values


    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number') or col_name.startswith('onehot')
    ]

    df_X = df_X[used_columns]

    model_config['used_columns'] = used_columns
    log('used columns: {}, size: {}'.format(len(used_columns), sys.getsizeof(df_X)))

    # if any(df_X.isnull()):
    #     model_config['missing'] = True
    #     df_X.fillna(-1, inplace=True)

    # scaling
    log_start()
    scaler = StandardScaler()
    df_X = scaler.fit_transform(df_X)
    log_time('scaling (df_X size:', sys.getsizeof(df_X), ')')
    model_config['scaler'] = scaler

    # fitting
    log_start()
    model_config['mode'] = args.mode
    regression = ( args.mode == 'regression' )
    if regression:
        scoring = NEG_MEAN_SQUARED_ERROR
        models = [
            GradientBoostingRegressor(),
            RandomForestRegressor(),
            ExtraTreesRegressor(),
            AdaBoostRegressor()
        ]

    else:
        scoring = 'roc_auc'
        models = [
            GradientBoostingClassifier(),
            RandomForestClassifier(),
            ExtraTreesClassifier(),
            AdaBoostClassifier()
        ]
        if train_rows < 10**4:
            models.append( LinearSVC() )
            models.append( SVC() )

    #    param_grid = {
    #        'min_samples_leaf': range(1, 30),
    #        'max_features': np.arange(0.1, 1.1, 0.1),
    #    }

    #    cv = KFold( n_splits=5, shuffle=True)
    #    grid = GridSearchCV( model, param_grid, cv=cv, scoring=scoring, n_jobs=N_JOBS)
    #    grid.fit(df_X, df_y)

    #X_test_scaled, _, _ = preprocess_test_data(args, model_config)
    #y_test = read_test_target(args)

    sampling_rates = (5000, 10000, 15000)
    log('Starting models selection by sampling data by {} rows'.format(sampling_rates))

    for samples in sampling_rates:
        X_selection = df_X[:min(samples, train_rows)]
        y_selection = df_y[:min(samples, train_rows)]

        scores = iterate_models(models, X_selection, y_selection, scoring)

        #already cross-validate on full dataset, go best model selection
        if y_selection.shape[0] == train_rows:
            log('Sample size meet dataset size, stop sampling')
            break

        # select models better than average
        selected = []
        for i in range(len(models)):
            if scores[i] > np.mean(scores):
                selected.append(models[i])

        # only one model? go model selection
        if len(selected) == 1:
            log('Only one model survive, stop sampling')
            break

        models = selected  # leave only best models for next iteration
        log('Survive {} models'.format( len(models) ))

    best_index = np.argmax(scores)
    model = models[best_index]

    log_start()
    model.fit(df_X, y=df_y)
    log_time('fit best model', model)
    model_config['model'] = model

    log('Train time: {}'.format(time.time() - start_train_time))
    log_trail()
    return model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    parser.add_argument('--nrows')
    args = parser.parse_args()

    model_config = train(args)

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

