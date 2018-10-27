import time
from typing import List, Any
from copy import deepcopy

import numpy as np
from contextlib import contextmanager

from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split

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
            step_instances = [StepInstance(None, x[:sample_rows], y[:sample_rows])]
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
        inputs_count = len(inputs)

        # generate instances
        if len(step.instances) == 0:
            step.init_instances(MAX_INSTANCES)

        # train/test split
        if step.scoring:
            self.train_test_splits(inputs)

        # iterate step for all inputs
        for index in range(inputs_count):

            log('input {} of {}'.format(index, inputs_count))

            if step.scoring:
                x_train = self.__X_trains[index]
                y_train = self.__y_trains[index]
                x_test = self.__X_tests[index]
                y_test = self.__y_tests[index]

                step_results: List[StepInstance] = step.iterate(x_train, y_train, x_test, y_test, is_subsampling)

            else:
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
    def train_test_splits(self, step_results):

        self.clean_train_test()

        for index in range(len(step_results)):

            x = step_results[index].x
            y = step_results[index].y

            log('train/test split dataset {}'.format(index))

            # define stratify or not
            stratify = None
            if self.mode == CLASSIFICATION:
                stratify = y

            # log('x:', x)
            # log('y:', y)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=stratify)
            # log('x_train:', x_train)
            # log('y_train:', y_train)

            self.__X_trains.append(x_train)
            self.__y_trains.append(y_train)
            self.__X_tests.append(x_test)
            self.__y_tests.append(y_test)

    def clean_train_test(self):
        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []
