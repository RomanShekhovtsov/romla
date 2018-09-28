import argparse
import time
import pandas as pd
import os

from dask.array.ufunc import da_frompyfunc

from train import  train
from predict import predict
from score import score
from utils import log

TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--test-target-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    parser.add_argument('--fast-test')

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
    for i in [8]: #tests.keys(): #[1,2,3,4,5,6]: #[3,5]:
        start_time = time.time()
        folder = r'..\check_' + str(i) + '_' + tests[i][0] + '\\'
        argv = ['--train-csv', folder + 'train.csv',
                '--prediction-csv', folder + 'prediction.csv',
                '--test-csv', folder + 'test.csv',
                '--test-target-csv', folder + 'test-target.csv',
                '--model-dir', r'.',
                '--mode', tests[i],
                '--fast-test', 'y'
                ]
        args = parser.parse_args(argv)

        metrics = dict()
        metrics['folder'] = folder
        metrics['mode'] = tests[i]

        log('processing',folder)

        metrics.update( train(args) )
        metrics.update( predict(args) )
        metrics.update( score(args) )
        metrics['Total time'] = time.time() - start_time

        log('Total time: {}'.format( metrics['Total time'] ))

        d_metrics = pd.DataFrame([metrics])
        METRICS_FILE = 'metrics.csv'
        header = not os.path.isfile( METRICS_FILE )
        d_metrics.to_csv(METRICS_FILE, mode='a', index=False, header=header)

        print('-'*80)
