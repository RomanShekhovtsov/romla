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
from sklearn.metrics import roc_auc_score

from utils import *
from metrics import *
from log import *
from preprocessing import preprocessing
from pipeline import Pipeline
from step import StepInstance, Step
from model import *

from xgboost_wrapper import XGBoostWrapper
from lightgbm_wrapper import LightGBMWrapper
from model import Model
from estimator import NEG_MEAN_SQUARED_ERROR

REGRESSION = 'regression'
CLASSIFICATION = 'classification'

MAX_MODEL_SELECTION_ROWS = 10 ** 5

TRAIN_TEST_SPLIT_TEST_SIZE = 0.25
SAMPLING_RATES = [10000]
MIN_TRAIN_ROWS = SAMPLING_RATES[0] * (1 - TRAIN_TEST_SPLIT_TEST_SIZE)

metrics = get_metrics()
start_train_time = time.time()


def train(args):
    try:
        return _train(args)
    except BaseException as e:
        log('EXCEPTION:', e)
        log(traceback.format_exc())
        exit(1)


def _train(args):

    # dict with data necessary to make predictions
    model_config = {}

    log('time limit:', TIME_LIMIT)
    metrics['run_date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    metrics['dataset'] = args.train_csv

    X, y = preprocessing(args, model_config)
    metrics['X_size'] = sys.getsizeof(X)
    train_rows = X.shape[0]

    # fitting
    model_config['mode'] = args.mode
    regression = (args.mode == REGRESSION)

    if regression:
        scorer = neg_mean_squared_error
        models = [XGBoostWrapper().get_regressor(),
                  LightGBMWrapper().get_regressor()
                  ]

    else:
        scorer = roc_auc_score
        models = [XGBoostWrapper().get_classifier(),
                  LightGBMWrapper().get_classifier()
                  ]

    step = Step( models, scorer=scorer)
    steps = [step]
    p = Pipeline(steps, time_left(), args.mode)
    p.train(X, y)
    print(p.best_score)
    best_model = step.best_model.wrapper.estimator
    log('best score:', p.best_score)
    log('best model:', best_model)
    # log('best model speed:', speed)

    metrics['best_method'] = step.best_model.get_name()
    metrics['best_score'] = p.best_score

    # with time_metric('model_params_search'):
    #     best_model = best_model.model.model_params_search(best_model, X, y, scoring, speed)
    #     # best_model = model
    #
    # with time_metric('fit_best_model'):
    #     best_model.fit(X, y=y)

    model_config['model'] = best_model

    save_metrics('train')

    log('Train time: {}'.format(time.time() - start_train_time))
    log_trail()
    return model_config


def time_left():
    t_left = TIME_LIMIT - (time.time() - start_train_time)
    return max(t_left, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=[CLASSIFICATION, REGRESSION], required=True)
    parser.add_argument('--nrows', type=int)
    args = parser.parse_args()

    model_config = train(args)

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

