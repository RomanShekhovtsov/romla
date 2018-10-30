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


class ParamRuleType(Enum):
    RANDOM = 'random'
    SEQUENCE = 'sequence'


class WrongParamRule(Exception):
    pass


class Model:

    def __init__(self, wrapper=None, param_space={}, param_rules={}):
        self.wrapper = wrapper
        self.param_space = param_space
        self.param_rules = param_rules

        self.param_counters = {}
        self.params = []

    def get_name(self):
        return self.wrapper.__class__.__name__

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
        param_rule = self.param_rules.get(key, ParamRuleType.RANDOM)

        if param_rule == ParamRuleType.SEQUENCE:
            distribution_index = self.param_counters.get(key, 0) % len(param_distribution)
            self.param_counters[key] = distribution_index + 1

        elif param_rule == ParamRuleType.RANDOM:
            distribution_index = random.randint(1, len(param_distribution)) - 1

        else:
            raise WrongParamRule('UNKNOWN PARAM RULE TYPE: \'{}\', supported types: {}'.format(
                param_rule,
                [p.value for p in ParamRuleType]
            ))

        return param_distribution[distribution_index]

    def set_params(self, params):
        self.params = params
        self.wrapper.set_params(params)

    def fit(self, x, y=None):
        log('fitting model {} {}'.format(self.get_name(), self.params))
        return self.wrapper.fit(x, y=y)

    def predict(self, x):
        return self.wrapper.predict(x)
