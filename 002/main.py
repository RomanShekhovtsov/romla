import argparse
import os
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, roc_auc_score

from train import train
from predict import predict
from scorer import score

from utils import log, log_start, log_time

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    start_total_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--test-target-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    parser.add_argument('--nrows')

    tests = {
        1: 'regression',
        2: 'regression',
        3: 'regression',
        4: 'classification',
        5: 'classification',
        6: 'classification',
        7: 'classification',
        8: 'classification',
    }
    for i in tests.keys():
        start_total_time = time.time()
        folder = r'..\..\check_' +  str(i) + '_' + tests[i][0] + '\\'
        argv = [
            '--train-csv', folder + 'train.csv',
            '--test-csv', folder + 'test.csv',
            '--prediction-csv', folder + 'predict.csv',
            '--test-target-csv', folder + 'test-target.csv',
            '--model-dir', '.',
            '--nrows', '10000' if i in [4,5,6] else '1000' if i in [7,8] else '',
            '--mode', tests[i] ]
        args = parser.parse_args(argv)

        log('processing', folder)
        train(args)
        predict(args)
        score(args)
        log('Total time: {}'.format(time.time() - start_total_time ))
        log('-'*80)