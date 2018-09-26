import argparse
import os
import pandas as pd
import time
from sklearn.metrics import mean_squared_error

from train import train
from predict import predict

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', required=True)
    parser.add_argument('--test-target-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    #train( args )
    #predict( args )

    # load model
    # read dataset
    target = pd.read_csv(args.test_target_csv)
    print('test-target read, shape {}'.format(target.shape))

    predict = pd.read_csv(args.prediction_csv)
    print('prediction read, shape {}'.format(predict.shape))

    rmse = mean_squared_error(target, predict) ** 0.5
    print('RMSE:', rmse)

    print('Total time: {}'.format(time.time() - start_time))
