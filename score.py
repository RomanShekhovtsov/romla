import argparse
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from utils import log

def score(args):
    metrics = dict()
    start_time = time.time()

    test_target = pd.read_csv(args.test_target_csv)
    prediction = pd.read_csv(args.prediction_csv)

    if args.mode == 'regression':
        rmse = mean_squared_error(test_target['target'], prediction['prediction']) ** 0.5
        print('RMSE:', rmse)
        metrics['score.score'] = rmse
    else:
        roc_auc = roc_auc_score(test_target['target'], prediction['prediction'] )
        print('ROC-AUC:', roc_auc)
        metrics['score.score'] = roc_auc

    metrics['score.score_time'] = time.time() - start_time
    print('Score time: {}'.format( metrics['score.score_time'] ))

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-target-csv', required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)

    argv = ['--test-target-csv', r'..\check_1_r\test-target.csv',
            '--prediction-csv', r'..\check_1_r\prediction.csv',
            '--mode', 'regression']
    log (argv )
    args = parser.parse_args(argv)
    score(args)