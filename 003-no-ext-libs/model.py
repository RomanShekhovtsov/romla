import time
import random
from copy import deepcopy
from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

from utils import *


class ModelParamsSearchStrategy(Enum):
    GRID_SEARCH = 'random'
    FIRST_BEST = 'first_best'


class Model:

    def __init__(self, estimator, param_space):
        self.estimator = estimator
        self.param_space = param_space
        self.params = []

    def get_name(self):
        return self.estimator.__class__.__name__

    def new_instance(self):
        return deepcopy(self)

    # return params combinations count
    def param_space_cardinality(self):
        cardinality = 1
        for key in self.param_space.keys():
            cardinality *= len(self.param_space[key])
        return cardinality

    # sample params from param space
    def sample_param_space(self):

        param_sample = {}

        for key in self.param_space.keys():
            param_sample[key] = self.sample_param(key)

        return param_sample

    # sample parameter from space
    def sample_param(self, key):
        # TODO: sampling types; avoid random value duplications.
        param_distribution = self.param_space[key]
        distribution_index = random.randint(1, len(param_distribution)) - 1
        return param_distribution[distribution_index]

    def set_params(self, params):
        self.params = params

    def fit_transform(self, x, y=None):
        return self.estimator.fit_transform(x, y=y), y

    def predict(self, x):
        return self.estimator.predict(x)
