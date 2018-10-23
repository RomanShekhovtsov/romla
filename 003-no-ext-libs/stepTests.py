import unittest
import random
from copy import deepcopy
import numpy as np

from step import *


class ModelMock():

    def __init__(self, param_space_cardinality):
        self.koeff = None
        self.__cardinality = param_space_cardinality
        self.estimator = lambda x, koeff: np.dot(x, koeff)

    def get_name(self):
        return self.estimator.__class__.__name__ + str(self.__cardinality)

    def fit_transform(self, x, y_train):
        return self.estimator(x, self.koeff), y_train

    def predict(self, x):
        pass

    def param_space_cardinality(self):
        return self.__cardinality

    def new_instance(self):
        return deepcopy(self)

    def sample_param_space(self):
        return {'koeff': random.randint(1, self.__cardinality)}

    def set_params(self, params):
        self.koeff = params['koeff']


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
        self.assertEqual(len(step.instances), 0)
        self.assertEqual(step.best_score, None)
        self.assertEqual(step.best_model, None)

    def test_init_instances(self):

        instances_count = 10

        # summary param spaces < max instances
        step = Step([ModelMock(1), ModelMock(2), ModelMock(3)], scorer=None)
        self.assertEqual(len(step.models), 3)
        step.init_instances(instances_count)
        self.step = step

        self.assertEqual(len(step.instances), 1 + 2 + 3)  # < max instances

        cardinalities = [1, 2, 2, 3, 3, 3]
        for i in range(len(cardinalities)):
            self.assertEqual(step.instances[i].get_name(), 'function' + str(cardinalities[i]))

        # summary param spaces > max instances
        step = Step([ModelMock(2), ModelMock(6), ModelMock(5)], scorer=None)
        step.init_instances(instances_count)

        # check instances-models distribution
        cardinalities = [2, 2, 6, 6, 6, 6, 5, 5, 5]
        for i in range(len(cardinalities)):
            self.assertEqual( step.instances[i].get_name(), 'function' + str(cardinalities[i]))

        self.assertEqual(len(step.instances), instances_count)

        self.step = step
        for instance in step.instances:
            self.assertIsNotNone(instance.koeff)

    # @unittest.skip
    def test_iterate(self):
        self.test_init_instances()
        self.step.iterate([1, 2, 3, 4], [2, 4, 6, 8])


if __name__ == '__main__':
    unittest.main()