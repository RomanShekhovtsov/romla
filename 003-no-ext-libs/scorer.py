import argparse
import os
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, roc_auc_score

from utils import *
from log import *

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))


def read_test_target(args):
    nrows = args.nrows
    if nrows == -1:
        nrows = None

    # read dataset
    log_start()
    target = pd.read_csv(args.test_target_csv, low_memory=False, nrows=nrows)
    log_time('read', args.test_target_csv, 'shape', target.shape)
    return target['target']


def score(args, target=None, prediction=None):
    nrows = args.nrows
    if nrows == -1:
        nrows = None

    if target is None:
        target = read_test_target(args)

    if prediction is None:
        # read dataset
        log_start()
        predict = pd.read_csv(args.prediction_csv, low_memory=False, nrows=nrows)
        log_time('read', args.prediction_csv, 'shape', prediction.shape)
        prediction = predict['prediction']

    if args.mode == 'regression':
        score = mean_squared_error(target, prediction) ** 0.5
        log('RMSE:', score)
    else:
        score = roc_auc_score(target, prediction)
        log('ROC-AUC:', score)

    log_trail()
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--test-target-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    score(args)
