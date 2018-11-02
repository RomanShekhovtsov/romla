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
from step import StepResult, Step
from model import *

from xgboost_wrapper import XGBoostWrapper
from lightgbm_wrapper import LightGBMWrapper
from catboost_wrapper import CatBoostWrapper

from model import Model
from estimator import NEG_MEAN_SQUARED_ERROR

REGRESSION = 'regression'
CLASSIFICATION = 'classification'

MAX_MODEL_SELECTION_ROWS = 10 ** 5

# use this to stop the algorithm before time limit exceeds
TIME_RESERVE_SECONDS = 20  # we must finish 20 seconds prior to time limit
TIME_RESERVE_COEFF = 0.8  # we won't exceed 80% of TIME_LIMIT

TRAIN_TEST_SPLIT_TEST_SIZE = 0.25
SAMPLING_RATES = [10000]
MIN_TRAIN_ROWS = SAMPLING_RATES[0] * (1 - TRAIN_TEST_SPLIT_TEST_SIZE)

metrics = get_metrics()
start_train_time = None

def train(args):
    try:
        return _train(args)
    except BaseException as e:
        log('EXCEPTION:', e)
        log(traceback.format_exc())
        exit(1)


def _train(args):
    global start_train_time
    start_train_time = time.time()
    # dict with data necessary to make predictions
    model_config = {}

    log('time limit:', TIME_LIMIT)
    metrics['run_date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    metrics['dataset'] = args.train_csv

    x, y = preprocessing(args, model_config)
    metrics['X_size'] = sys.getsizeof(x)

    # fitting
    model_config['mode'] = args.mode
    regression = (args.mode == REGRESSION)

    if regression:
        scorer = neg_mean_squared_error
        models = [  # XGBoostWrapper().get_regressor(),
                  # CatBoostWrapper().get_regressor(model_config['categorical_indices']),
                  LightGBMWrapper().get_regressor()  # model_config['categorical_columns'], model_config['used_columns'])
                  ]

    else:
        scorer = roc_auc_score
        models = [  # XGBoostWrapper().get_classifier(),
                  # CatBoostWrapper().get_classifier(model_config['categorical_indices']),
                  LightGBMWrapper().get_classifier()  # model_config['categorical_columns'], model_config['used_columns'])
                  ]

    x = x.values
    y = y.values
    # feature selection:
    # if len(x) > 5 * 10**4:
    #     log('FEATURE SELECTION PIPELINE RUN:')
    #     step = Step(models, scorer=scorer)
    #     steps = [step]
    #     p = Pipeline(steps, 30, args.mode)
    #     sample_size = int(len(x) / 10)
    #
    #     p.train(x[:sample_size], y[:sample_size], feature_selection=True)
    #     feature_importance = None
    #     for step_instance in step.step_results:
    #         if feature_importance is None:
    #             feature_importance = step_instance.instance.wrapper.estimator.feature_importance()
    #         else:
    #             feature_importance = np.add(feature_importance,
    #                                         step_instance.instance.wrapper.estimator.feature_importance())
    #
    #     x = x[:, feature_importance > 0]
    #     model_config['used_columns'] = list(np.array(model_config['used_columns'])[feature_importance > 0])
    #     feature_used = len(model_config['used_columns'])
    #     feature_removed = len(feature_importance) - feature_used
    #     log('feature selection: {} of {} used, {} removed'.format(
    #         feature_used,
    #         len(feature_importance),
    #         feature_removed
    #     ))

    # train pipeline
    step = Step(models, scorer=scorer)
    steps = [step]
    p = Pipeline(steps, time_left(), args.mode)
    p.train(x, y)

    #print(p.best_score)
    best_model = step.best_model.wrapper.estimator

    if regression:
        log('best score:', (-p.best_score) ** 0.5)
    else:
        log('best score:', p.best_score)

    log('best model:', step.best_model.get_name(), step.best_model.wrapper.params)
    # log('best model speed:', speed)

    metrics['best_method'] = step.best_model.get_name() + ' ' + str(step.best_model.params)
    metrics['best_score'] = p.best_score

    # with time_metric('model_params_search'):
    #     best_model = best_model.model.model_params_search(best_model, x, y, scoring, speed)
    #     # best_model = model
    #
    # with time_metric('fit_best_model'):
    #     best_model.fit(x, y=y)

    model_config['model'] = best_model
    save_metrics('train')
    log('Train time: {}'.format(time.time() - start_train_time))
    log_trail()

    return model_config


def time_left():
    t_left = TIME_LIMIT - (time.time() - start_train_time)
    t_left = TIME_RESERVE_COEFF * (t_left - TIME_RESERVE_SECONDS)
    return max(t_left, 0)

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

