import time
import numpy as np
from contextlib import contextmanager

SPEED_RUN_SAMPLE_SIZE = 100  # rows to speed run through all step and models to score velocity
INITIAL_SAMPLE_SIZE = 1000 # rows to first iteration

# use this to stop the algorithm before time limit exceeds
TIME_RESERVE_SECONDS = 20  # we must finish 20 seconds prior to time limit
TIME_RESERVE_COEFF = 0.8  # we won't exceed 80% of TIME_LIMIT


# Abstract AutoML pipeline.
# Responsibilities:
# 1. Execute pipeline steps
# 2. Time management & sub-sampling
# 3. Models elimination after each cycle.
class Pipeline:

    time_budget = None
    steps = None

    __start_time = None
    __times = {}

    def __init__(self, steps, time_budget):
        self.steps = steps
        self.time_budget = time_budget

    def run(self, X, y=None):
        self.__start_time = time.time()
        X_speed = [X]
        y_speed = [y]

        for index in range(len(self.steps)):
            X_speed, y_speed = self.__iterate_step(0, X_speed, y_speed, speed_run = True)

        X = [X]
        y = [y]
        for index in range(len(self.steps)):
            X, y = self.__iterate_step(0, X, y)

    def __iterate_step(self, step_index, X_list, y_list, speed_run=False):

        step_outputs = []
        step_scores = []
        step = self.steps[step_index]

        #initial sample size
        sample_size = None
        if step.sampling:
            if speed_run:
                sample_size = SPEED_RUN_SAMPLE_SIZE
            else:
                sample_size = INITIAL_SAMPLE_SIZE

        first_run = True
        have_time = True
        while have_time:


            for index in range(len(X_list)):

                X = X_list[index]
                y = y_list[index]

                if X is None:
                    continue


                # different data may have different len() (skip or not skip NaNs, for example).
                rows = len(X)
                if sample_size is None:
                    sample_rows = rows
                else:
                    sample_rows = min(sample_size, rows)

                # initiate train/test once per data sample
                with self.timer('train_test_split' + instance.id):
                    step.init_data(X[:sample_rows], y[:sample_rows])

                for instance in step.instances(speed_run=speed_run):

                    with self.timer('fit' + instance.id):
                        # TODO: sampling methods, class balancing
                        instance_data = instance.fit()

                    with self.timer('process output'):

                        # scorer sampling save_output make_score
                        #   X      X         +           X
                        #   X      +         +           X
                        #   +      X         +           +
                        #   +      +         X           +
                        if step.scorer is None:
                            # just save output data
                            step_outputs.append(instance_data)
                        else:
                            # save scores
                            step_scores.append(step.scorer.score(instance_data))
                            if sample_rows is None:  # for sub-sampling
                                step_outputs.append(instance_data)

                if sample_rows is None or sample_rows == rows:
                     X_list[index] = None  # full dataset is processed, stop future processing

            # re-calc sample size


            # check time
            if not speed_run:

            first_run = False

        X_out, y_out, step.instances = self.__eliminate_by_score(step, step_outputs)

        return X_out, y_out

    def __eliminate_by_score(self, step, X_out, y_out):

        scores = []

        for X in X_out:
            score = step.score(X_out)
            scores.append(score)

        if step.elimination_policy == 'median':
            median = np.median(scores)
            X_results = []
            y_results = []
            result_instances = []

            for i in range(len(scores)):
                if scores[i] >= median:
                    X_results.append(X_out[i])
                    y_results.append(y_out[i])
                    result_instances.append(step.instances[i])

        elif step.elimination_policy == 'none':
            X_results = X_out
            y_results = y_out
            result_instances = step.instances()

        else:
            raise Exception('UNKNOWN ELIMINATION POLICY')

        return X_results, y_results, result_instances

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
