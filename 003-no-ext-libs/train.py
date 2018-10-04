import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import math
from enum import Enum
from copy import deepcopy
from contextlib import contextmanager

import xgboost as xgb
import lightgbm as lgb
import catboost

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestClassifier, RandomForestRegressor, \
    ExtraTreesClassifier, ExtraTreesRegressor, \
    AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score

from utils import *
from metrics import *

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))
TIME_RESERVE_SECONDS = 20  # we must finish 20 seconds prior to time limit
TIME_RESERVE_COEFF= 0.8  # we won't exceed 80% of TIME_LIMIT
start_time = time.time()
N_JOBS = 4

ONEHOT_MAX_UNIQUE_VALUES = 20
MAX_DATASET_COLUMNS = 1000
BIG_DATASET_SIZE = 500 * 1024 * 1024  # 300MB
MAX_MODEL_SELECTION_ROWS = 10 ** 5

TRAIN_TEST_SPLIT_TEST_SIZE = 0.25
SAMPLING_RATES = [10000]
MIN_TRAIN_ROWS = SAMPLING_RATES[0] * (1 - TRAIN_TEST_SPLIT_TEST_SIZE)

NEG_MEAN_SQUARED_ERROR = 'neg_mean_squared_error'
MIN_NUMBER = 1e-10  #small number to prevent division by zero

metrics = dict()

class ModelParamsSearchStrategy(Enum):
    GRID_SEARCH = 'random'
    FIRST_BEST = 'first_best'


# seconds left to our work
def time_left():
    t_left = TIME_LIMIT - (time.time() - start_time)
    return min(TIME_RESERVE_SECONDS, TIME_RESERVE_COEFF * t_left)


def evaluate_model(model, X, y, scoring, full_train):

    speed = dict()

    if full_train:
        rows = X.shape[0]

        t = time.time()
        model.fit(X, y=y)
        speed['fit'] = rows / (time.time() - t + MIN_NUMBER)

        t = time.time()
        prediction = model.predict(X)
        speed['predict'] = rows / (time.time() - t + MIN_NUMBER)

        score = calc_score(scoring, y, prediction)

    # if rows < min_train_rows:
    #     cv = math.ceil((min_train_rows / rows)) + 1  # make X_train_rows >= rows * (nfolds - 1)
    #     cv = min(rows, cv)  # correction for extra-small datasets
    #     method = 'cross validation ' + str(cv) + '-folds'
    #     score = np.mean(cross_val_score(model, X, y=y, scoring=scoring, cv=cv, n_jobs=N_JOBS))

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_TEST_SIZE)


        t = time.time()
        model.fit(X_train, y=y_train)
        speed['fit'] = X_train.shape[0] / (time.time() - t)

        t = time.time()
        prediction = model.predict(X_test)
        speed['predict'] = X_test.shape[0] / (time.time() - t)

        score = calc_score(scoring, y_test, prediction)

    return score, speed


def calc_score(scoring, y_test, prediction):

    if scoring == NEG_MEAN_SQUARED_ERROR:
        score = -mean_squared_error(y_test, prediction)
    else:
        score = roc_auc_score(y_test, prediction)

    return score


@contextmanager
def time_metric(name):
    t = time.time()
    yield
    metrics[name] = time.time() - t
    log(name,'[{} sec]'.format(round(metrics[name], 2)))


def iterate_models(models, X, y, scoring, min_train_rows=MIN_TRAIN_ROWS):

    scores = []
    speeds = []
    rows = y.shape[0]

    full_train = rows < min_train_rows

    for model in models:

        log_start()
        score, speed = evaluate_model(model, X, y, scoring, full_train)
        scores.append(score)
        speeds.append(speed)

        if scoring == NEG_MEAN_SQUARED_ERROR:
            print_score = (-score) ** 0.5
        else:
            print_score = score

        log_time('evaluate model, full train: {}; rows: {}; score:{};'.format(full_train, rows, print_score))
        log(model)

    return scores, speeds


def get_model_name(model):
    return model.__class__.__name__


# calculate sample size (rows) to perform model parameters search

def calc_sample_size(test_size, total_rows, fit_speed, predict_speed, n_iter):

    time_to_fit_all = total_rows / fit_speed
    time_to_search = time_left() - time_to_fit_all
    time_iteration = time_to_search / n_iter

    # equation for sample calculated from: max t_iteration = train_rows/Sfit + test_rows/Spredict
    # train_rows = (1-test_size)*sample
    # test_rows = test_size*sample
    sample_size = time_iteration / ((1 - test_size) / fit_speed + test_size / predict_speed)
    sample_size = min(sample_size, total_rows)

    return int(sample_size)


def model_params_search(model, X, y, scoring, speed):
    scores = []
    rows = y.shape[0]

    model_name = get_model_name(model)

    n_iter = None
    if model_name in('XGBRegressor','XGBClassifier'):
        strategy = ModelParamsSearchStrategy.FIRST_BEST
        params = {'max_depth': list(range(2,16))}
        n_iter = 15

    elif model_name in('LGBMRegressor','LGBMClassifier'):

    # if model_name == 'GradientBoostingRegressor':
    #
    #     # def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
    #     #              subsample=1.0, criterion='friedman_mse', min_samples_split=2,
    #     #              min_samples_leaf=1, min_weight_fraction_leaf=0.,
    #     #              max_depth=3, min_impurity_decrease=0.,
    #     #              min_impurity_split=None, init=None, random_state=None,
    #     #              max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
    #     #              warm_start=False, presort='auto'):
    #     strategy = ModelParamsSearchStrategy.GRID_SEARCH
    #     # estimator = GradientBoostingRegressor()
    #     params = {'learning_rate': [0.05, 0.1, 0.15],
    #               'min_samples_split': [2, 4, 8],
    #               'min_samples_leaf': [1, 2, 4],
    #               'max_depth': [2, 3, 4]
    #               }
    #     const_params = {}
    #     # n_iter = 81

        # def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
        #              learning_rate=0.1, n_estimators=100,
        #              subsample_for_bin=200000, objective=None, class_weight=None,
        #              min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
        #              subsample=1., subsample_freq=0, colsample_bytree=1.,
        #              reg_alpha=0., reg_lambda=0., random_state=None,
        #              n_jobs=-1, silent=True, importance_type='split', **kwargs):
        #
        strategy = ModelParamsSearchStrategy.FIRST_BEST
        params = {'max_depth': list(range(2,16))}
        n_iter = 15

    # elif model_name == 'RandomForestRegressor':
        # n_estimators = 10,
        # criterion = "mse",
        # max_depth = None,
        # min_samples_split = 2,
        # min_samples_leaf = 1,
        # min_weight_fraction_leaf = 0.,
        # max_features = "auto",
        # max_leaf_nodes = None,
        # min_impurity_decrease = 0.,
        # min_impurity_split = None,
        # bootstrap = True,
        # oob_score = False,
        # n_jobs = 1,
        # random_state = None,
        # verbose = 0,
        # warm_start = False):
        # strategy = ModelParamsSearchStrategy.GRID_SEARCH
        # params = {'min_samples_split': [2, 4, 8],
        #           'max_features': ['auto', 'sqrt', 'log2'],
        #           'min_samples_leaf': [1, 2, 4],
        #           'max_depth': [3, 5, 7]
        #           }
    #
    # elif model_name == 'ExtraTreesRegressor':
        # n_estimators = 10,
        # criterion = "mse",
        # max_depth = None,
        # min_samples_split = 2,
        # min_samples_leaf = 1,
        # min_weight_fraction_leaf = 0.,
        # max_features = "auto",
        # max_leaf_nodes = None,
        # min_impurity_decrease = 0.,
        # min_impurity_split = None,
        # bootstrap = False,
        # oob_score = False,
        # n_jobs = 1,
        # random_state = None,
        # verbose = 0,
        # warm_start = False):
        # strategy = ModelParamsSearchStrategy.GRID_SEARCH
        # estimator = ExtraTreesRegressor()
        # params = {'min_samples_split': [2, 4, 8],
        #           'max_features': ['auto', 'sqrt', 'log2'],
        #           'min_samples_leaf': [1, 2, 4],
        #           'max_depth': [3, 5, 7]
        #           }
        #
    # elif model_name == 'AdaBoostRegressor':
    # elif model_name == 'GradientBoostingClassifier':
    # elif model_name == 'RandomForestClassifier':
    # elif model_name == 'ExtraTreesClassifier':
    # elif model_name == 'AdaBoostClassifier':
    # elif model_name == 'LinearSVC':
    # elif model_name == 'SVC':

    else:
        raise Exception('UNKNOWN MODEL: ', model_name)

    # def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
    #              fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
    #              verbose=0, pre_dispatch='2*n_jobs', random_state=None,
    #              error_score='raise', return_train_score="warn"):
    if strategy == ModelParamsSearchStrategy.GRID_SEARCH:

        # n_iter = 0
        # for param_name in params.keys():
        #     param_values = params[param_name]
        #     n_iter += len(param_values)

        estimator = deepcopy(model)  # TODO: excessive copy for gridsearch?
        GridSearchCV()
        searcher = GridSearchCV(estimator,
                                params,
                                scoring=scoring,
                                n_jobs=4,
                                cv=3,
                                return_train_score=False)
        searcher.fit(X, y)
        best_estimator = searcher.best_estimator_

    elif strategy == ModelParamsSearchStrategy.FIRST_BEST:

        test_size = 0.25
        time_to_fit_all =  X.shape[0] / speed['fit']

        samples = calc_sample_size(test_size, X.shape[0], speed['fit'], speed['predict'], n_iter)
        log('calculated samples:', samples)
        X_train, X_test, y_train, y_test = train_test_split(X[:samples], y[:samples], test_size=test_size)

        param_name = list(params.keys())[0]  # only one param for this strategy
        param_values = params[param_name]
        prev_score = None

        iteration_time = 0
        for param_value in param_values:

            if time_left() < time_to_fit_all + iteration_time:
                log('stop model params search due to time limit:',
                    'time left={}; time to fit all={}; last iteration time: {}',
                    time_left(), time_to_fit_all, iteration_time)
                break

            iteration_time = time.time()
            estimator = deepcopy(model)
            est_params = {param_name: param_value}
            log('estimate', est_params)
            estimator.set_params( **est_params)
            estimator.fit(X_train, y_train)
            predict = estimator.predict(X_test)
            score = calc_score(scoring, y_test, predict)

            if (not prev_score is None) and (prev_score > score):
                log('early finish at {}={}'.format(param_name, param_value))
                best_estimator = estimator
                break

            best_estimator = estimator
            prev_score = score
            iteration_time = time.time() - iteration_time


    return best_estimator


def train(args):
    start_train_time = time.time()

    # dict with data necessary to make predictions
    model_config = {}

    # TODO: FAIL CHECK!!!

    # train_estimate = estimate_csv(args.train_csv)
    # log('estimate', args.train_csv, train_estimate)
    metrics['dataset'] = args.train_csv
    with time_metric('read dataset'):
        df = read_csv(args.train_csv, args.nrows)
    #metrics['read_csv'] = time.time() - t

    initial_dataset_size = sys.getsizeof(df)
    is_big = initial_dataset_size > BIG_DATASET_SIZE
    model_config['is_big'] = is_big

    # missing values
    model_config['missing'] = False
    with time_metric('impute missing values'):
        if df.isnull().values.any():
            model_config['missing'] = True
            df.fillna(-1, inplace=True)
        else:
            log('dataset has no missing values')

    with time_metric('optimize dataframe'):
        optimize_dataframe(df)

    df_y = df.target
    df_X = df.drop('target', axis=1)
    df = None

    train_rows, train_cols = df_X.shape
    if train_rows < 2:
        raise Exception('TRAIN SIZE {} < 2.'.format(train_rows))

    metrics['remove low correlated features'] = 0
    metrics['process datetime features'] = 0

    if is_big:
        with time_metric('remove low correlated features'):
            new_feature_count = min(train_cols,
                                    int(train_cols / (initial_dataset_size / BIG_DATASET_SIZE)))
            # take only high correlated features
            correlations = np.abs([
                np.corrcoef(df_y, df_X[col_name])[0, 1]
                for col_name in df_X.columns if col_name.startswith('number')
            ])
            new_columns = df_X.columns[np.argsort(correlations)[-new_feature_count:]]
            df_X = df_X[new_columns]
            log('remove {} low correlated features'.format(train_cols - new_feature_count))

    else:

        # features from datetime
        with time_metric('process datetime features'):
            df_dates = transform_datetime_features(df_X)
            log('features from datetime ({} columns)'.format(len(df_dates.columns)))

            # missing values
            if df_dates.isnull().values.any():
                model_config['missing_dates'] = True
                df_dates.fillna(-1, inplace=True)
            else:
                log('no missing values in datetime features')

            # optimize
            optimize_dataframe(df_dates)
            df_X = pd.concat((df_X, df_dates), axis=1)
            df_dates = None

    # calculate unique values
    with time_metric('process categorical features'):
        df_unique = df_X.apply(lambda x: x.nunique())
        df_const = df_unique[df_unique == 1]
        df_unique = df_unique[df_unique > 2]
        df_unique = df_unique[df_unique <= ONEHOT_MAX_UNIQUE_VALUES]
        df_unique.sort_values(inplace=True)

        # drop constant features
        df_X.drop(df_const.index, axis=1, inplace=True)
        log('{} constant features dropped'.format(df_const.shape[0]))

        # categorical encoding
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

            log('categorical encoding ({} columns)'.format(len(df_cat.columns)))
            optimize_dataframe(df_cat)
            df_X = pd.concat((df_X, df_cat), axis=1)
            df_cat = None

        model_config['categorical_values'] = categorical_values

    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number') or col_name.startswith('onehot')
    ]

    df_X = df_X[used_columns]
    if len(df_X.columns) < 1:
        raise Exception('ALL FEATURES DROPPED, STOPPING')

    model_config['used_columns'] = used_columns
    log('used columns: {}, size: {}'.format(len(used_columns), sys.getsizeof(df_X)))

    with time_metric('impute missing values before scale'):
        if df_X.isnull().values.any():
            model_config['missing'] = True
            df_X.fillna(-1, inplace=True)

    # if any(df_X.isnull()):
    #     model_config['missing'] = True
    #     df_X.fillna(-1, inplace=True)

    # scaling
    with time_metric('scale'):
        scaler = StandardScaler()
        X = scaler.fit_transform(df_X.values).astype(np.float16)
        df_X = None
        log('scale (X size: {})'.format(sys.getsizeof(X)))
        model_config['scaler'] = scaler

    # fitting
    model_config['mode'] = args.mode
    regression = (args.mode == 'regression')

    xgb_const_params = {'n_jobs': 4}
    lgb_const_params = {'boosting_type': 'gbdt',
                        'objective': 'regression' if args.mode == 'regression' else 'binary',
                        'metric': 'rmse',
                        'learning_rate': 0.01,
                        'num_leaves': 200,
                        "feature_fraction": 0.70,
                        "bagging_fraction": 0.70,
                        'bagging_freq': 4,
                        'max_depth': -1,
                        'verbosity': -1,
                        'reg_alpha': 0.3,
                        'reg_lambda': 0.1,
                        # "min_split_gain":0.2,
                        "min_child_weight": 10,
                        'zero_as_missing': True,
                        'num_threads': 4,
                        }

    if regression:
        scoring = NEG_MEAN_SQUARED_ERROR

        lgbr = lgb.sklearn.LGBMRegressor()
        lgbr.set_params(**lgb_const_params)

        xgbr = xgb.sklearn.XGBRegressor()
        xgbr.set_params(**xgb_const_params)
        models = [xgbr, lgbr]

            # RandomForestRegressor(),
            # ExtraTreesRegressor(),
            # AdaBoostRegressor()

    else:
        scoring = 'roc_auc'

        lgbc = lgb.LGBMClassifier()
        lgbc.set_params(**lgb_const_params)

        xgbc = xgb.sklearn.XGBClassifier()
        xgbc.set_params(**xgb_const_params)

        models = [xgbc, lgbc]

        #     RandomForestClassifier(),
        #     ExtraTreesClassifier(),
        #     AdaBoostClassifier()

        # if train_rows < 10 ** 4:
        #     models.append(LinearSVC())
        #     models.append(SVC())

    #    param_grid = {
    #        'min_samples_leaf': range(1, 30),
    #        'max_features': np.arange(0.1, 1.1, 0.1),
    #    }

    #    cv = KFold( n_splits=5, shuffle=True)
    #    grid = GridSearchCV( model, param_grid, cv=cv, scoring=scoring, n_jobs=N_JOBS)
    #    grid.fit(df_X, df_y)

    # X_test_scaled, _, _ = preprocess_test_data(args, model_config)
    # y_test = read_test_target(args)

    log('starting models selection by sampling data by {} rows'.format(SAMPLING_RATES))
    model_selection_start = time.time()

    for samples in SAMPLING_RATES:
        X_selection = X[:min(samples, train_rows)]
        y_selection = df_y[:min(samples, train_rows)]

        scores, speeds = iterate_models(models, X_selection, y_selection, scoring)

        # already cross-validate on full dataset, go best model selection
        if y_selection.shape[0] == train_rows:
            log('sample size meet dataset size, stop sampling')
            break

        # select models better than average
        selected = []
        for i in range(len(models)):
            if scores[i] > np.mean(scores):
                selected.append(models[i])

        # only one model? go model selection
        if len(selected) == 1:
            log('only one model survive, stop sampling')
            break

        models = selected  # leave only best models for next iteration
        log('survive {} models'.format(len(models)))

    metrics['model selection'] = time.time() - model_selection_start

    best_index = np.argmax(scores)
    best_model = models[best_index]
    speed = speeds[best_index]

    log('best score:', scores[best_index])
    log('best model:', best_model)

    with time_metric('model_params_search'):
        best_model = model_params_search(best_model, X, df_y, scoring, speed)
        # best_model = model

    with time_metric('fit_best_model'):
        best_model.fit(X, y=df_y)
        model_config['model'] = best_model

    save_metrics(metrics, 'train')

    log('Train time: {}'.format(time.time() - start_train_time))
    log_trail()
    return model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    parser.add_argument('--nrows', type=int)
    args = parser.parse_args()

    model_config = train(args)

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)
