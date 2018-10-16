import time
import numpy as np
from contextlib import contextmanager
from log import *


INITIAL_SAMPLE_SIZE = 1000 # rows to first iteration
MAX_INSTANCES = 100  # max instances to evaluate at each step

# use this to stop the algorithm before time limit exceeds
TIME_RESERVE_SECONDS = 20  # we must finish 20 seconds prior to time limit
TIME_RESERVE_COEFF = 0.8  # we won't exceed 80% of TIME_LIMIT


# Abstract AutoML pipeline.
# Responsibilities:
# 1. Execute pipeline steps
# 2. Time management & sub-sampling
class Pipeline:

    def __init__(self, steps, time_budget):
        self.steps = steps
        self.time_budget = time_budget

        self.best_instance = None
        self.best_score = None

        self.__start_time = None
        self.__times = {}

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

        x_list = [x]
        y_list = [y]
        steps_count = len(self.steps)
        log('pipeline of {} step(s) started'.format(steps_count))

        for index in range(steps_count):
            x_list, y_list = self.__iterate_step(index, x_list, y_list)

        last_step = self.steps[len(self.steps) - 1]
        best_index = np.argmax(last_step.scores)
        self.best_instance = last_step.iterated_instances[best_index]
        self.best_score = last_step.scores[best_index]
        log('pipeline finished, best score: {}'.format(self.best_score))
        log_trail()

        return self.best_score

    # iterate different sample sizes.
    # - for each sample:
    #   - train test split input datasets
    #   - iterate input datasets
    #     - for each dataset iterate all models
    #   - then eliminate models (and datasets).
    def __iterate_step(self, step_index, x_list, y_list):

        log('step {} started'.format(step_index))
        step = self.steps[step_index]
        datasets_count = len(x_list)


        # initial sample size
        sample_size = None
        if step.sampling:
            sample_size = INITIAL_SAMPLE_SIZE

        # generate instances
        step.init_instances(MAX_INSTANCES)

        continue_sampling = True
        while continue_sampling:
            # TODO: stop if all datasets fully proceed, or survived only one model

            iteration_time = time.time()

            # train/test split for each sample
            for index in range(len(x_list)):

                X = x_list[index]
                y = y_list[index]

                if X is None:
                    continue

                log('train/test split dataset {}'.format(index))
                step.add_train_test_split(X, y)

            # iterate input datasets
            x_outputs, y_outputs = step.iterate_datasets(sample_size)

            # check time & re-calc sample size
            iteration_time = time.time() - iteration_time
            time_left = self.time_left()
            have_time = time_left > iteration_time * 2
            
            if not have_time:
                log('step {} stopped by time limit (time left: {}; last iteration time: {})'.format(
                    step_index, time_left, iteration_time))
                    
            # check if only one instance survive
            many_instance_survived = len(x_outputs) > 1

            # define stop condition
            continue_sampling = have_time and (many_instance_survived and (sample_size is not None))

            if continue_sampling:
                if many_instance_survived:
                    sample_size = sample_size * 2
                else:
                    # only one survived - let's do last fit_transform on full dataset
                    sample_size = None

        return x_outputs, y_outputs

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