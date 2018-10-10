import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import math
import traceback

import catboost

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestClassifier, RandomForestRegressor, \
    ExtraTreesClassifier, ExtraTreesRegressor, \
    AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import LinearSVC, SVC

from utils import *
from metrics import *
from log import *
from preprocessing import preprocessing

from xgboost_wrapper import XGBoostWrapper
from lightgbm_wrapper import LightGBMWrapper
from model import Model
from estimator import NEG_MEAN_SQUARED_ERROR

MAX_MODEL_SELECTION_ROWS = 10 ** 5

TRAIN_TEST_SPLIT_TEST_SIZE = 0.25
SAMPLING_RATES = [10000]
MIN_TRAIN_ROWS = SAMPLING_RATES[0] * (1 - TRAIN_TEST_SPLIT_TEST_SIZE)

metrics = get_metrics()


def train(args):
    try:
        return _train(args)
    except BaseException as e:
        log('EXCEPTION:', e)
        log(traceback.format_exc())
        exit(1)

def _train(args):

    start_train_time = time.time()

    # dict with data necessary to make predictions
    model_config = {}

    log('time limit:', TIME_LIMIT)
    metrics['dataset'] = args.train_csv

    X, y = preprocessing(args, model_config)
    metrics['X_size'] = sys.getsizeof(X)
    train_rows = X.shape[0]

    # fitting
    model_config['mode'] = args.mode
    regression = (args.mode == 'regression')


    if regression:
        scoring = NEG_MEAN_SQUARED_ERROR
        models = [XGBoostWrapper.get_regressor(),
                  LightGBMWrapper.get_regressor()]
            # RandomForestRegressor(),
            # ExtraTreesRegressor(),
            # AdaBoostRegressor()

    else:
        scoring = 'roc_auc'
        models = [XGBoostWrapper.get_classifier(),
                  LightGBMWrapper.get_classifier()]

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

    selected = models
    for samples in SAMPLING_RATES:
        X_selection = X[:min(samples, train_rows)]
        y_selection = y[:min(samples, train_rows)]

        models = selected # leave only best models for next iteration
        scores, speeds = iterate_models(models, X_selection, y_selection, scoring)

        # already cross-validate on full dataset, go best model selection
        if y_selection.shape[0] == train_rows:
            log('sample size meet dataset size, stop sampling')
            break

        # select models better than average
        selected = []
        for i in range(len(models)):
            if scores[i] >= np.mean(scores):  # >= for case of equal scores for all models
                selected.append(models[i])

        # only one model? go model selection
        if len(selected) == 1:
            log('only one model survive, stop sampling')
            break

        log('survive {} model(s)'.format(len(selected)))

    metrics['model selection'] = time.time() - model_selection_start

    best_index = np.argmax(scores)
    best_model = models[best_index]
    speed = speeds[best_index]

    log('best score:', scores[best_index])
    log('best model:', best_model)
    log('best model speed:', speed)

    metrics['best_method'] = best_model.get_name()

    with time_metric('model_params_search'):
        best_model = best_model.model.model_params_search(best_model, X, y, scoring, speed)
        # best_model = model

    with time_metric('fit_best_model'):
        best_model.fit(X, y=y)
        model_config['model'] = best_model

    save_metrics('train')

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
