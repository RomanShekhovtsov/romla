import unittest
import numpy as np
from copy import deepcopy

from pipeline import *
from step import IterationData
from stepTests import *


class StepMock:

    def __init__(self, models, scorer=None, elimination_policy='median'):

        self.models = models
        self.scorer = scorer
        self.scoring = scorer is not None
        self.elimination_policy = elimination_policy

        self.instances = []

    def init_instances(self, max_instances):
        self.instances = self.models

    def iterate(self, x_train, y_train, x_test=None, y_test=None, is_subsampling=False):

        x = deepcopy(x_train)
        iteration_results = []

        for instance in self.instances:
            iteration_result = IterationData(None, list(map(instance, x)), deepcopy(y_train))

            if self.scorer is not None:
                iteration_result.score = sum(list(map(self.scorer, iteration_result.x, y_train)))
                #print(iteration_result.x, y_train)
            iteration_results.append(iteration_result)
            #log(iteration_result.x)

        return iteration_results

    # def instances(self):
    #     n = 0
    #     while n < 2:
    #         yield InstanceMock()
    #         n += 1


class InstanceMock:

    def fit(self, data):
        return data + 1


class ScorerMock:

    def score(self, data):
        return data


class PipelineTest(unittest.TestCase):

    def test_train_on_mocks(self, time_budget=22,rows=100):

        steps_count = 3

        x = np.arange(rows)
        y = x % 2

        transform_model = lambda x: x + 1
        steps = []
        for i in range(steps_count-1):
            steps.append(StepMock([transform_model]))

        predict_models = [lambda x: x % 2,
                          lambda x: (x - 1) % 2,
                          lambda x: (x - 2) % 2,
                          ]
        scorer = lambda x, y: 1 if x == y else 0
        steps.append(StepMock(predict_models, scorer=scorer))

        p = Pipeline(steps, time_budget)
        best_score = p.train(x, y)

        self.assertEqual(best_score, rows * (1 - TEST_SIZE))

        # self.assertEqual(p.steps[0].best_score, None)
        # self.assertEqual(p.steps[1].best_score, None)
        # self.assertEqual(p.steps[2].best_score, best_score)
        #
        # self.assertEqual(p.steps[0].best_score, rows + 1)
        # self.assertEqual(p.steps[1].best_score, rows + 2)
        # self.assertEqual(p.steps[2].best_score, rows + 3)
        #
        # self.assertEqual(p.steps[0].scores[0], rows + 1)
        # self.assertEqual(p.steps[1].scores[0], rows + 2)
        # self.assertEqual(p.steps[2].scores[0], rows + 3)

    def test_zero_time_budget(self):
        self.test_train_on_mocks(0, 999)

    @unittest.expectedFailure
    # must fail after fist sample
    def test_zero_time_budget(self):
        self.test_train_on_mocks(0, INITIAL_SAMPLE_SIZE + 1)


if __name__ == '__main__':
    unittest.main()