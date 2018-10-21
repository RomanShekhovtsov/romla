import random
import numpy as np
from copy import deepcopy


from log import *


# iteration results
class IterationData:

    def __init__(self, instance, x, y):
        self.instance = instance
        self.x = x
        self.y = y
        self.score = None


# Abstract step in AutoML pipeline
# Responsibilities:
# 1. Train/test split for each input dataset
# 2. Initiate models instances for iteration
# 2. Iterate all input datasets through all instances
# 3. Instances elimination after each cycle.
class Step:

    def __init__(self, models, scorer=None, elimination_policy='median'):

        self.models = models
        self.elimination_policy = elimination_policy

        self.scoring = scorer is not None
        self.__scorer = scorer

        self.instances = []
        self.all_datasets_proceed = False

        self.iteration_data = []
        # self.x_outputs = []
        # self.y_outputs = []
        # self.scores = []

        self.best_score = None
        self.best_model = None

    # for given sample size, iterate all input datasets through all models
    # return output datasets (when sample size == dataset size) and save scores
    # TODO: clean data after iteration
    def iterate(self, x_train, y_train, x_test=None, y_test=None, is_subsampling=False):

        # params validation
        if self.__scorer is None and is_subsampling:
            raise Exception('FOR SUBSAMPLING WE NEED SCORER')

        # iterate sample through instances
        for instance in self.instances:

            iterated_instance = deepcopy(instance)
            x_output, y_output = iterated_instance.fit_transform(x_train, y_train)

            if is_subsampling:
                # dataset not proceed, clear output
                x_output = None
                y_output = None

            iteration_result = IterationData(iterated_instance, x_output, y_output)

            if self.scoring is not None:
                # save scores
                prediction = instance.predict(x_test)
                iteration_result.score = self.__scorer.score(y_test, prediction)

            self.iteration_data.append(iteration_result)

        return self.iteration_data

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