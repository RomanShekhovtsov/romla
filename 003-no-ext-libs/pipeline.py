import time
import numpy as np
from contextlib import contextmanager

from step import IterationResults

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

    time_budget = None
    steps = None

    best_instance = None
    best_score = None

    __start_time = None
    __times = {}

    def __init__(self, steps, time_budget):
        self.steps = steps
        self.time_budget = time_budget

    def run(self, x, y=None):
        self.__start_time = time.time()

        X_list = [x]
        y_list = [y]
        for index in range(len(self.steps)):
            X_list = self.__iterate_step(index, X_list, y_list)

        last_step = self.steps[len(self.steps) - 1]
        best_index = np.argmax(last_step.scores)
        self.best_instance = last_step.iterated_instances[best_index]
        self.best_score = last_step.scores[best_index]

        return self.best_score

    # iterate different sample sizes.
    # - for each sample:
    #   - train test split input datasets
    #   - iterate input datasets
    #     - for each dataset iterate all models
    #   - then eliminate models (and datasets).
    def __iterate_step(self, step_index, x_list, y_list):

        step = self.steps[step_index]
        datasets_count = len(x_list)

        # initial sample size
        sample_size = None
        if step.sampling:
            sample_size = INITIAL_SAMPLE_SIZE

        # generate (datasets_count * MAX_INSTANCES) instances
        step.init_instances(datasets_count, MAX_INSTANCES)

        continue_sampling = True
        while continue_sampling:
            # TODO: stop if all datasets fully proceed, or survived only one model

            full_cycle_time = time.time()

            # train/test split for each sample
            for index in range(len(x_list)):

                X = x_list[index]
                y = y_list[index]

                if X is None:
                    continue

                step.add_train_test_split(X, y)

            # iterate input datasets
            output = step.iterate_datasets(sample_size)

            # check time & re-calc sample size
            full_cycle_time = time.time() - full_cycle_time
            time_left = self.time_left()
            have_time = time_left > full_cycle_time * 2

            # check if only one instance survive
            many_instance_survived = len(output) > 1

            # define stop condition
            continue_sampling = have_time and (many_instance_survived or (sample_size is not None))

            if continue_sampling:
                if many_instance_survived:
                    sample_size = sample_size * 2
                else:
                    # only one survived - let's do last fit_transform on full dataset
                    sample_size = None

        return output

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