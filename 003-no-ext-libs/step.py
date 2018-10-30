import random
import numpy as np
from copy import deepcopy

from model import Model
from log import *


# iteration results
class StepInstance:

    def __init__(self, instance, x=None, y=None):
        self.instance: Model = instance
        self.x = x
        self.y = y
        self.score = None

    def fit(self, x, y):
        return self.instance.fit(x, y=y), y

    def predict(self, x):
        return self.instance.predict(x)


# Abstract step in AutoML pipeline
# Responsibilities:
# 1. Train/test split for each input dataset
# 2. Initiate stepInstances instances for iteration
# 2. Iterate all input datasets through all instances
# 3. Instances elimination after each cycle.
class Step:

    def __init__(self, models, scorer=None, elimination_policy='median'):

        self.models = models
        self.elimination_policy = elimination_policy

        self.scoring = scorer is not None
        self.__scorer = scorer

        self.instances: list[StepInstance] = []

        # self.x_outputs = []
        # self.y_outputs = []
        # self.scores = []

        self.best_score = None
        self.best_model : Model = None

    # for given sample size, iterate all input datasets through all stepInstances
    # return output datasets (when sample size == dataset size) and save scores
    # TODO: clean data after iteration
    def iterate(self,
                x_train,
                y_train=None,
                x_test=None,
                y_test=None,
                is_subsampling=False,
                disable_elimination=False):

        # params validation
        if self.__scorer is None and is_subsampling:
            raise Exception('FOR SUBSAMPLING WE NEED SCORER')

        # iterate sample through instances
        for instance in self.instances:

            x_output, y_output = instance.fit(x_train, y_train)

            if is_subsampling:
                # dataset not proceed, clear output
                x_output = None
                y_output = None

            instance.x = x_output
            instance.y = y_output

            if self.scoring and x_test is not None:
                # save scores
                prediction = instance.predict(x_test)
                instance.score = self.__scorer(y_test, prediction)

        # eliminate results
        if self.scoring and not disable_elimination:
            self.eliminate_by_score()

        return self.instances

    # initiate instances
    def init_instances(self, max_instances=100):

        instances_left = max_instances
        models_count = len(self.models)

        for i in range(models_count):

            model = self.models[i]

            if callable(getattr(model, 'param_space_cardinality')):

                # calculate instances count for this model
                instances_for_model = int(instances_left / (models_count - i))
                instances_for_model = min(instances_for_model, model.param_space_cardinality())

                for j in range(instances_for_model):
                    instance = model.new_instance()
                    params = model.sample_param_space()
                    instance.set_params(params)
                    self.instances.append(StepInstance(instance))
                    instances_left -= 1
            else:
                self.instances.append(StepInstance(deepcopy(model)))

    # eliminate instances by score
    def eliminate_by_score(self):

        # scores = list(map(lambda x: -np.inf if x.score is None else x.score, step_results))
        scores = list(map(lambda x: x.score, self.instances))
        log('scores:', scores)

        best_index = np.argmax(scores)
        self.best_model = self.instances[best_index].instance
        self.best_score = self.instances[best_index].score

        survived = []

        if self.elimination_policy == 'median':
            median = np.median(scores)
            #log('median:', median)
            for i in range(len(scores)):
                if scores[i] >= median:
                    survived.append(self.instances[i])

        elif self.elimination_policy == 'one_best':
            survived = [self.instances[best_index]]

        else:
            raise Exception('UNKNOWN ELIMINATION POLICY: ' + self.elimination_policy)

        log('elimination: {} of {} instances survived'.format(len(survived), len(scores)))
        self.instances = survived

        return self.instances
