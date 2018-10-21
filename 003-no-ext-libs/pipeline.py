import time
from typing import List, Any
from copy import deepcopy

import numpy as np
from contextlib import contextmanager
from sklearn.model_selection import train_test_split

from log import *
from step import IterationData


INITIAL_SAMPLE_SIZE = 1000 # rows to first iteration
MAX_INSTANCES = 100  # max instances to evaluate at each step
TEST_SIZE = 0.3

REGRESSION = 'regression'
CLASSIFICATION = 'classification'

# use this to stop the algorithm before time limit exceeds
TIME_RESERVE_SECONDS = 20  # we must finish 20 seconds prior to time limit
TIME_RESERVE_COEFF = 0.8  # we won't exceed 80% of TIME_LIMIT


class PipelineInstance:

    def __init__(self, x=None, y=None):
        self.models = []
        self.score = None
        self.x = x
        self.y = y


# Abstract AutoML pipeline.
# Responsibilities:
# 1. Execute pipeline steps
# 2. Time management & sub-sampling
# 3. Models elimination
class Pipeline:

    def __init__(self, steps, time_budget, mode=CLASSIFICATION):
        self.steps = steps
        self.time_budget = time_budget
        self.mode = mode

        self.best_pipeline = None
        self.__start_time = None
        self.__times = {}

        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []

    def predict(self, x_train, x_test=None, y_train=None):
        self.__start_time = time.time()

        if x_test is None:
            x_test = x_train

        x = x_train
        for step in self.steps:
            x = step.best_model.fit_transform(x, y=y_train)

        last_step = self.steps[len(self.steps) - 1]
        prediction = last_step.predict(x_test)

        return prediction

    def train(self, x, y=None):

        self.__start_time = time.time()
        steps_count = len(self.steps)
        log('pipeline of {} step(s) started'.format(steps_count))

        sample_size = INITIAL_SAMPLE_SIZE
        rows = len(x)

        pipelines = [PipelineInstance()]
        continue_sampling = True
        while continue_sampling:

            run_time = time.time()

            log('sample size: {}'.format(sample_size))
            # TODO: sampling methods
            # TODO: modify for re-fit (start new sample from end of previous one)
            sample_rows = min(sample_size, rows)
            is_subsampling = (sample_rows < rows)

            # init pipelines with new sample
            for pipeline in pipelines:
                pipeline.x = x[:sample_rows]
                pipeline.y = y[:sample_rows]

            # run pipelines for that sample
            for index in range(steps_count):
                pipelines = self.__iterate_step(index, pipelines, is_subsampling)

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
                if len(pipelines) > 1:
                    sample_size = sample_size * 2
                else:
                    # only one survived - let's do last fit_transform on full dataset
                    sample_size = rows

        best_index = np.argmax(map(lambda p: p.score, pipelines))
        self.best_pipeline = pipelines[best_index]
        self.best_score = pipelines[best_index].score
        log('train finished, best score: {}'.format(self.best_score))
        log_trail()

        return self.best_score

    # iterate different sample sizes.
    # - for each sample:
    #   - train test split input datasets
    #   - iterate input datasets
    #     - for each dataset iterate all models
    #   - then eliminate models (and datasets).
    def __iterate_step(self, step_index, pipelines, is_subsampling):

        log('STEP {} STARTED'.format(step_index))
        step = self.steps[step_index]
        pipelines_count = len(pipelines)

        # generate instances
        step.init_instances(MAX_INSTANCES)

        # train/test split
        if step.scoring:
            self.train_test_splits(pipelines)

        # iterate step for all pipelines
        for index in range(pipelines_count):

            log('starting pipeline {}'.format(index))

            if step.scoring:
                x_train = self.__X_trains[index]
                y_train = self.__y_trains[index]
                x_test = self.__X_tests[index]
                y_test = self.__y_tests[index]

                step_results: List[IterationData] = step.iterate(x_train, y_train, x_test, y_test, is_subsampling)

            else:
                x = pipelines[index].x
                y = pipelines[index].y

                step_results: List[IterationData] = step.iterate(x, y, is_subsampling=is_subsampling)

            # replace base pipeline with steps results
            for step_result in step_results:
                new_pipeline = PipelineInstance(step_result.x, step_result.y)
                new_pipeline.score = step_result.score
                new_pipeline.models = deepcopy(pipelines[index].models)
                new_pipeline.models.append(step_result.instance)
                pipelines.append(new_pipeline)
            pipelines.pop(index)

            if step.scoring:
                # cleanup input data
                self.__X_trains[index] = None
                self.__y_trains[index] = None
                self.__X_tests[index] = None
                self.__y_tests[index] = None

        # eliminate pipelines
        if step.scoring:
            pipelines = self.__eliminate_by_score(pipelines, step.elimination_policy)

        self.clean_train_test()
        return pipelines

    # eliminate instances by score
    def __eliminate_by_score(self, pipelines, elimination_policy):

        before_elimination = len(pipelines)

        scores = list(map(lambda x: x.score, pipelines))
        best_index = np.argmax(scores)
        self.best_pipeline = pipelines[best_index]

        filtered_results = []

        if elimination_policy == 'median':
            median = np.median(scores)
            for i in range(len(scores)):
                if scores[i] >= median:
                    filtered_results.append(pipelines[i])

        elif elimination_policy == 'one_best':
            filtered_results = [pipelines[best_index]]

        else:
            raise Exception('UNKNOWN ELIMINATION POLICY')

        log('elimination: {} of {} instances survived'.format(len(filtered_results), before_elimination))
        return filtered_results

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
    def train_test_splits(self, pipelines):

        self.clean_train_test()

        for index in range(len(pipelines)):

            x = pipelines[index].x
            y = pipelines[index].y

            log('train/test split dataset {}'.format(index))

            # define stratify or not
            stratify = None
            if self.mode == CLASSIFICATION:
                stratify = y

            #log('x:', x)
            #log('y:', y)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=stratify)
            #log('x_train:', x_train)
            #log('y_train:', y_train)

            self.__X_trains.append(x_train)
            self.__y_trains.append(y_train)
            self.__X_tests.append(x_test)
            self.__y_tests.append(y_test)

    def clean_train_test(self):
        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []
