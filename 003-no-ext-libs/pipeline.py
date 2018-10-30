import time
from typing import List, Any
from copy import deepcopy

import numpy as np
from contextlib import contextmanager

from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from log import *
from step import StepInstance, Step


INITIAL_SAMPLE_SIZE = 1000 # rows to first iteration
MAX_INSTANCES = 100  # max instances to evaluate at each step
TEST_SIZE = 0.3

REGRESSION = 'regression'
CLASSIFICATION = 'classification'

# use this to stop the algorithm before time limit exceeds
TIME_RESERVE_SECONDS = 20  # we must finish 20 seconds prior to time limit
TIME_RESERVE_COEFF = 0.8  # we won't exceed 80% of TIME_LIMIT

N_SPLITS = 10


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
        log('pipeline of {} step(s) started'.format(steps_count))

        sample_size = INITIAL_SAMPLE_SIZE
        rows = len(x)

        continue_sampling = True
        while continue_sampling:

            run_time = time.time()

            log('SAMPLE SIZE: {}'.format(sample_size))
            # TODO: sampling methods
            # TODO: modify for re-fit (start new sample from end of previous one)
            sample_rows = min(sample_size, rows)
            is_subsampling = (sample_rows < rows)

            # run pipelines for sample
            random_sample_index = np.random.choice(rows, sample_rows, replace=False)  # random subsample
            step_instances = [StepInstance(None, x[random_sample_index], y[random_sample_index])]
            for index in range(steps_count):
                step_instances = self.iterate_step(index, step_instances, is_subsampling)

            # check time & re-calc sample size
            run_time = time.time() - run_time
            time_left = self.time_left()
            have_time = time_left > run_time * 2

            if not have_time:
                log('pipeline stopped by time limit (time left: {}; last iteration time: {})'.format(
                    time_left, run_time))

            # define stop condition
            continue_sampling = have_time and is_subsampling

            if continue_sampling:
                if len(step_instances) > 1:
                    sample_size = sample_size * 2
                else:
                    # only one survived - let's do last fit_transform on full dataset
                    sample_size = rows
            log_trail()

        best_index = np.argmax(list(map(lambda p: p.score, step_instances)))
        self.best_score = step_instances[best_index].score
        log('train finished, best score: {}'.format(self.best_score))
        log_trail('=')

        return self.best_score

    # iterate different sample sizes.
    # - for each sample:
    #   - train test split input datasets
    #   - iterate input datasets
    #     - for each dataset iterate all stepInstances
    #   - then eliminate stepInstances (and datasets).
    def iterate_step(self, step_index, inputs: List[StepInstance], is_subsampling):

        log('STEP {} STARTED'.format(step_index))
        step = self.steps[step_index]

        # generate instances
        if len(step.instances) == 0:
            step.init_instances(MAX_INSTANCES)

        # train/test split
        if step.scoring:
            self.split_datasets(inputs, is_subsampling)

        # iterate step for all inputs
        datasets_count = len(inputs)
        for index in range(datasets_count):

            if step.scoring and len(step.instances) > 1:  # if only one model, fit to all dataset
                log('input data {} of {}'.format(index + 1, datasets_count))

                x_train = self.__X_trains[index]
                y_train = self.__y_trains[index]
                x_test = self.__X_tests[index]
                y_test = self.__y_tests[index]
                if self.__split_types[index] == 'KFold':

                    folds_count = len(x_train)
                    instances_count = len(step.instances)
                    scores = np.zeros(instances_count)

                    for fold_index in range(folds_count):
                        log('processing fold {} of {}'.format(fold_index + 1, folds_count))
                        step_results: List[StepInstance] = step.iterate(x_train[fold_index],
                                                                        y_train[fold_index],
                                                                        x_test[fold_index],
                                                                        y_test[fold_index],
                                                                        is_subsampling,
                                                                        disable_elimination=True)
                        step_scores = list(map(lambda p: p.score, step_results))
                        scores = np.add(scores, step_scores)
                        # will use last fold results as output

                    # set KFold average score as instance scores and eliminate by score
                    scores = np.divide(scores, folds_count)  # average score
                    for instance_index in range(instances_count):
                        step_results[instance_index].score = scores[instance_index]

                    step_results = step.eliminate_by_score()

                else:
                    step_results: List[StepInstance] = step.iterate(x_train, y_train, x_test, y_test, is_subsampling)

            else:

                log('input data {} of {}'.format(index, datasets_count))

                x = inputs[index].x
                y = inputs[index].y

                step_results: List[StepInstance] = step.iterate(x, y, is_subsampling=is_subsampling)

            if step.scoring:
                # cleanup input data
                self.__X_trains[index] = None
                self.__y_trains[index] = None
                self.__X_tests[index] = None
                self.__y_tests[index] = None

        self.clean_train_test()
        return step_results

    # seconds left to work
    def time_left(self):
        t_left = self.time_budget - (time.time() - self.__start_time)
        t_left = TIME_RESERVE_COEFF * (t_left - TIME_RESERVE_SECONDS)
        return max(t_left, 0)

    @contextmanager
    def timer(self, name):
        t = time.time()
        yield
        self.__times[name] = time.time() - t

    # add train/test split for input dataset
    def split_datasets(self, datasets, is_subsampling):

        self.clean_train_test()

        for index in range(len(datasets)):

            x = datasets[index].x
            y = datasets[index].y

            if len(x) <= INITIAL_SAMPLE_SIZE and not is_subsampling:  # KFold
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
