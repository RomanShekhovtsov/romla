import unittest
import random
from copy import deepcopy
import numpy as np
from sklearn.metrics import mean_squared_error
from preprocessing import CsvLoader

from step import *
from model import Model, ParamRuleType, WrongParamRule
from utils import *

class ModelMock:

    def __init__(self, param_space_cardinality):
        self.coeff = 0
        self.__cardinality = param_space_cardinality
        self.estimator = lambda x, coeff: np.dot(x, coeff)

    def get_name(self):
        return self.estimator.__class__.__name__ + str(self.__cardinality)

    def fit(self, x, y=None):
        return self.estimator(x, self.coeff)

    def predict(self, x):
        return self.estimator(x, self.coeff)

    def param_space_cardinality(self):
        return self.__cardinality

    def new_instance(self):
        return deepcopy(self)

    def sample_param_space(self):
        self.coeff +=1
        return {'coeff': self.coeff % self.__cardinality}

    def set_params(self, params):
        self.coeff = params['coeff']


class StepTest(unittest.TestCase):

    step = None

    # def setUp(self):
    #     self.step = Step([], scorer=None, elimination_policy='median')

    def test_init(self):
        policy = 'median'
        step = Step([], scorer=None, elimination_policy=policy)
        self.assertEqual(len(step.models), 0)
        self.assertEqual(step.scoring, False)
        self.assertEqual(step.elimination_policy, policy)
        self.assertEqual(len(step.step_results), 0)
        self.assertEqual(step.best_score, None)
        self.assertEqual(step.best_model, None)

    def test_init_instances(self):

        instances_count = 10

        # summary param spaces < max step_results
        step = Step([ModelMock(1), ModelMock(2), ModelMock(3)], scorer=neg_mean_squared_error)
        self.assertEqual(len(step.models), 3)
        step.init_instances(instances_count)
        self.step = step

        self.assertEqual(len(step.step_results), 1 + 2 + 3)  # < max step_results

        cardinalities = [1, 2, 2, 3, 3, 3]
        for i in range(len(cardinalities)):
            self.assertEqual(step.step_results[i].instance.get_name(), 'function' + str(cardinalities[i]))

        # summary param spaces > max step_results
        step = Step([ModelMock(2), ModelMock(6), ModelMock(5)], scorer=neg_mean_squared_error)
        step.init_instances(instances_count)

        # check step_results-models distribution
        cardinalities = [2, 2, 6, 6, 6, 6, 5, 5, 5]
        for i in range(len(cardinalities)):
            self.assertEqual(step.step_results[i].instance.get_name(), 'function' + str(cardinalities[i]))

        self.assertEqual(len(step.step_results), instances_count)

        self.step = step
        for instance in step.step_results:
            self.assertIsNotNone(instance.instance.coeff)

    # @unittest.skip
    def test_iterate(self):
        self.test_init_instances()
        self.step.iterate([1, 2, 3, 4],
                          [2, 4, 6, 8],
                          [5, 6, 7],
                          [10, 12, 14])
        self.assertEqual(0, self.step.best_score)

    # @unittest.expectedFailure
    @staticmethod
    def test_wrong_param_rule():
        try:
            m = Model(None, {'p': [1, 2, 3]},{'p': 'wrong'})
            m.sample_param_space()

        except WrongParamRule as e:
            print(e)

        except Exception as e:
            raise Exception(e)


if __name__ == '__main__':
    unittest.main()