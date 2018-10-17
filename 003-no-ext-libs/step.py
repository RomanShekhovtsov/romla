import random
import numpy as np
from copy import deepcopy

from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.7


# iteration results
class IterationResult:

    def __init__(self, instance, x_output, y_output):
        self.instance = instance
        self.x_output = x_output
        self.y_output = y_output
        self.score = None


# Abstract step in AutoML pipeline
# Responsibilities:
# 1. Train/test split for each input dataset
# 2. Initiate models instances for iteration
# 2. Iterate all input datasets through all instances
# 3. Instances elimination after each cycle.
class Step:

    def __init__(self, models, scorer=None, elimination_policy=None, sampling=False):

        # params validation
        if scorer is None and sampling:
            raise Exception('FOR SUBSAMPLING WE NEED SCORER')

        self.models = models
        self.elimination_policy = elimination_policy
        self.sampling = sampling
        self.scoring = scorer is not None
        self.__scorer = scorer

        self.instances = []

        self.iteration_results = []
        # self.x_outputs = []
        # self.y_outputs = []
        # self.scores = []

        self.best_score = None
        self.best_model = None

        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []

        self.__X_train_sample = None
        self.__y_train_sample = None

    def clear_train_test(self):
        self.__X_trains = []
        self.__y_trains = []
        self.__X_tests = []
        self.__y_tests = []

    # add train/test split for input dataset
    def add_train_test_split(self, X, y, sample):
        x_train, y_train, x_test, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, stratify=y)
        self.__X_trains.append(x_train)
        self.__y_trains.append(y_train)
        self.__X_tests.append(x_test)
        self.__y_tests.append(y_test)

    # for given sample size, iterate all input datasets through all models
    # return output datasets (when sample size == dataset size) and save scores
    # TODO: clean data after iteration
    def iterate_datasets(self, sample_size):

        self.iteration_results = []

        datasets_count = len(self.__X_trains)
        for index in range(datasets_count):

            x_train = self.__X_trains[index]
            y_train = self.__y_trains[index]
            x_test = self.__X_tests[index]
            y_test = self.__y_tests[index]

            if x_train is None:
                continue  # skip fully proceed datasets

            # get sample dataset
            # datasets might have different len() (skip or not skip NaNs, for example).
            # TODO: sampling methods
            # TODO: modify for re-fit (start new sample from end of previous one)
            rows = len(x_train)
            if self.sampling:
                sample_rows = min(sample_size, rows)
            else:
                sample_rows = rows

            is_subsampling = (sample_rows < rows)
            if is_subsampling:
                x_train = x_train[:sample_rows]
                y_train = y_train[:sample_rows]

                test_rows = int(sample_rows / 2)
                x_test = x_test[:test_rows]
                y_test = y_test[:test_rows]

            # iterate sample through instances
            for instance in self.instances:

                iterated_instance = deepcopy(instance)
                x_output, y_output = iterated_instance.fit_transform(x_train, y_train)

                if is_subsampling:
                    # dataset not proceeded, clear output
                    x_output = None
                    y_output = None

                iteration_result = IterationResult(iterated_instance, x_output, y_output)
                self.iteration_results.append(iteration_result)

                # if self.sampling:
                #    # for sub-sampling we need only scores
                #    _output = None

                if self.scoring is not None:
                    # save scores
                    prediction = instance.predict(x_test)
                    iteration_result.score = self.__scorer.score(y_test, prediction)

            if not is_subsampling:
                # dataset proceeded, clean input data and prevent future processing
                self.__X_trains[index] = None
                self.__y_trains[index] = None
                self.__X_tests[index] = None
                self.__y_tests[index] = None

            if self.scoring:
                self.__eliminate_by_score()

        return self.iteration_results

    # eliminate instances and datasets by score
    def __eliminate_by_score(self):

        scores = list(map(lambda x: x.score, self.iteration_results))
        best_index = np.argmax(scores)
        best_iteration = self.iteration_results[best_index]
        self.best_score = best_iteration.score
        self.best_model = best_iteration.instance

        filtered_results = []

        if self.elimination_policy == 'median':
            median = np.median(scores)
            for i in range(len(scores)):
                if scores[i] >= median:
                    filtered_results.append(self.iteration_results[i])

        elif self.elimination_policy == 'one_best':
            filtered_results = [self.iteration_results[best_index]]

        else:
            raise Exception('UNKNOWN ELIMINATION POLICY')

        self.iteration_results = filtered_results

    # initiate instances
    def init_instances(self, max_instances):

        instances_left = max_instances
        models_count = len(self.models)

        for i in range(models_count):

            model = self.models[i]

            # calculate instances count for this model
            instances_for_model = int(instances_left / (models_count - i))
            instances_for_model = min(instances_for_model, model.param_space_cardinality())

            for j in range(instances_for_model):
                instance = model.new_instance()
                params = model.sample_param_space()
                instance.set_params(params)
                self.instances.append(instance)
                instances_left -= 1