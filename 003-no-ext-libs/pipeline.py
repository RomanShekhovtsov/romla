import time
import math
from typing import List, Any
from copy import deepcopy

import numpy as np
from contextlib import contextmanager

from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from log import *
from step import StepResult, Step


#MAX_INITIAL_SAMPLE_SIZE = 10000 # rows to first iteration
MAX_INSTANCES = 400  # max step_results to evaluate at each step
TEST_SIZE = 0.3

REGRESSION = 'regression'
CLASSIFICATION = 'classification'

MIN_TRAIN_TEST_SPLIT_SAMPLE = 1000
N_SPLITS = 5


# Abstract AutoML pipeline.
# Responsibilities:
# 1. Execute pipeline steps
# 2. Time management & sub-sampling
# 3. Models elimination
class Pipeline:

    def __init__(self, steps: List[Step], time_budget, mode=CLASSIFICATION):
        self.steps = steps
        self.time_budget = time_budget
        self.mode = mode

        # self.best_pipeline: PipelineInstance = None
        self.best_score = None
        self.__start_time = None
        self.total_rows = None
        self.survive_fraction = 0.5
        self.__times = {}

        self.__split_types = []
        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []

    # def predict(self, x):
    #     self.__start_time = time.time()
    #
    #     step_instances = self.best_pipeline.stepInstances
    #     for index in range(len(step_instances) - 1):
    #         step_instance = step_instances[index]
    #         x = step_instance.fit_transform(x)
    #
    #     return step_instances[-1].predict(x)

    def train(self, x, y=None):

        self.__start_time = time.time()
        steps_count = len(self.steps)
        log('pipeline started ( step(s): {}; time budget: {}'.format(steps_count, self.time_budget))

        rows = len(x)
        cols = len(x[0])
        self.total_rows = rows

        # rows / 10))
        # initial sample size:
        sample_size = int(np.max((
            cols * 2,
            333 * self.time_left() / cols,
            2000
        )))

        while True:
            step_instances = self.do_sample(x, y, sample_size, rows)
            if sample_size < rows:
                sample_size = rows
            else:
                break

        best_index = np.argmax(list(map(lambda p: p.score, step_instances)))
        self.best_score = step_instances[best_index].score
        log('train finished, best score: {}'.format(self.best_score))
        log_trail('=')

        return self.best_score

    def do_sample(self, x, y, sample_size, rows):
        run_time = time.time()
        steps_count = len(self.steps)

        # TODO: sampling methods
        # TODO: modify for re-fit (start new sample from end of previous one)
        sample_rows = min(sample_size, rows)
        is_subsampling = (sample_rows < rows)
        log('SAMPLE ROWS: {}'.format(sample_rows))

        # run pipelines for sample
        if self.mode == CLASSIFICATION and is_subsampling:
            y_indexed = np.vstack((np.arange(len(y)), y.astype(int)))
            zeros = y_indexed[0][y_indexed[1] == 0]
            ones = y_indexed[0][y_indexed[1] == 1]

            zeros_count = int(sample_size/2)
            if zeros_count > len(zeros):
                zeros_count = len(zeros)

            ones_count = int(sample_size - zeros_count)
            if ones_count > len(ones):
                ones_count = len(ones)
                zeros_count = sample_size - ones_count

            zeros_index = np.random.choice(len(zeros), zeros_count, replace = False)
            ones_index = np.random.choice(len(ones), ones_count, replace=False)
            sample_index = np.hstack((zeros[zeros_index], ones[ones_index]))
        else:
            sample_index = np.random.choice(rows, sample_rows, replace=False)  # random subsample

        # TODO: multiclass support

        step_instances = [StepResult(None, x[sample_index], y[sample_index])]
        for index in range(steps_count):
            step_instances = self.iterate_step(index, step_instances, is_subsampling)

        log_trail()

        # # check time & re-calc sample size
        # run_time = time.time() - run_time
        # time_left = self.time_left()
        # have_time = time_left > run_time

        # if not have_time:
        #      log('pipeline stopped by time limit (time left: {}; last iteration time: {})'.format(
        #          time_left, run_time))

        # define stop condition
        return step_instances

    # iterate different sample sizes.
    # - for each sample:
    #   - train test split input datasets
    #   - iterate input datasets
    #     - for each dataset iterate all stepInstances
    #   - then eliminate stepInstances (and datasets).
    def iterate_step(self, step_index, inputs: List[StepResult], is_subsampling):

        log('STEP {} STARTED'.format(step_index))
        step = self.steps[step_index]

        # # generate step_results
        # if len(step.step_results) == 0:
        #     step.init_instances(MAX_INSTANCES)

        # train/test split
        if step.scoring:
            self.split_datasets(inputs)

        # iterate step for all inputs
        datasets_count = len(inputs)
        for index in range(datasets_count):
            log('input data {} of {}'.format(index + 1, datasets_count))

            if len(step.step_results) == 1 or not step.scoring:  # if only one model, fit to all dataset

                x = inputs[index].x
                y = inputs[index].y
                step_results: List[StepResult] = step.iterate(x, y, is_subsampling=is_subsampling)

            else:

                x_train = self.__X_trains[index]
                y_train = self.__y_trains[index]
                x_test = self.__X_tests[index]
                y_test = self.__y_tests[index]

                if self.__split_types[index] == 'KFold':

                    # TODO: replace with hyperopt
                    if len(step.step_results) == 0:
                        step.init_instances(MAX_INSTANCES)

                    folds_count = len(x_train)
                    instances_count = len(step.step_results)
                    scores = np.zeros(instances_count)

                    for fold_index in range(folds_count):
                        fold_time = time.time()
                        log('processing fold {} of {}'.format(fold_index + 1, folds_count))
                        # will use last fold results as output
                        step_results: List[StepResult] = step.iterate(x_train[fold_index],
                                                                      y_train[fold_index],
                                                                      x_test[fold_index],
                                                                      y_test[fold_index],
                                                                      is_subsampling,
                                                                      disable_elimination=True,
                                                                      time_budget=self.time_left())

                        step_scores = list(map(lambda p: p.score, step_results))
                        scores = np.add(scores, step_scores)

                    # set KFold average score as instance scores and eliminate by score
                    scores = np.divide(scores, folds_count)  # average score
                    for instance_index in range(instances_count):
                        step_results[instance_index].score = scores[instance_index]

                    step_results = step.eliminate_by_score()

                else:

                    if len(step.step_results) == 0:
                        work_time = time.time()
                        step_results = step.init_instances_hyperopt(x_train, y_train, x_test, y_test, self.time_left() / 2)
                        step_results = step.eliminate_by_score(0.5)

                    else:
                        work_time = time.time()
                        # step_iterations_forecast = min(
                        #     math.ceil(np.log2(len(step.step_results))),
                        #     math.ceil(np.log2(self.total_rows / len(x_train)))
                        # )
                        # log('step_iterations_forecast:', step_iterations_forecast)

                        step_results = step.iterate(
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            is_subsampling,
                            time_budget=self.time_left())

            if step.scoring:
                # cleanup input data
                self.__X_trains[index] = None
                self.__y_trains[index] = None
                self.__X_tests[index] = None
                self.__y_tests[index] = None

        self.clean_train_test()
        return step_results

    # eliminate step results with time budgeting
    # def massacre(self, work_time, step, x):
    #     work_time = time.time() - work_time
    #     instances_count = len(step.step_results)
    #     iterations_could = self.time_left() / work_time
    #     iterations_need = self.total_rows / len(x)
    #     if iterations_need < iterations_could:
    #         self.survive_fraction = 1
    #     else:
    #         self.survive_fraction = (1 / instances_count) ** (1 / iterations_could)
    #     return step.eliminate_by_score(self.survive_fraction)

    # seconds left to work
    def time_left(self):
        t_left = self.time_budget - (time.time() - self.__start_time)
        return max(t_left, 0)

    @contextmanager
    def timer(self, name):
        t = time.time()
        yield
        self.__times[name] = time.time() - t

    # add train/test split for input dataset
    def split_datasets(self, datasets):

        self.clean_train_test()

        for index in range(len(datasets)):

            x = datasets[index].x
            y = datasets[index].y

            if len(x) < MIN_TRAIN_TEST_SPLIT_SAMPLE:  # KFold
                split_type = 'KFold'
                log('KFold split dataset {}'.format(index))

                x_train = []
                x_test = []
                y_train = []
                y_test = []

                if self.mode == CLASSIFICATION:
                    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=1, shuffle=False)
                    split = kf.split(x, y)
                else:
                    kf = KFold(n_splits=N_SPLITS, random_state=1, shuffle=True)
                    split = kf.split(x)

                for train_index, test_index in split:
                    x_train.append(x[train_index])
                    y_train.append(y[train_index])
                    x_test.append(x[test_index])
                    y_test.append(y[test_index])

            else:  # train/test split
                split_type = 'test_train_split'
                log('test/train split dataset {}'.format(index))

                # define stratify or not
                stratify = None
                if self.mode == CLASSIFICATION:
                    stratify = y

                # log('x:', x)
                # log('y:', y)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=stratify, random_state=1)
                # log('x_train:', x_train)
                # log('y_train:', y_train)

            self.__split_types.append(split_type)
            self.__X_trains.append(x_train)
            self.__y_trains.append(y_train)
            self.__X_tests.append(x_test)
            self.__y_tests.append(y_test)

    def clean_train_test(self):
        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []
