import unittest
import numpy as np
from copy import deepcopy

from pipeline import *
from stepTests import *


class StepMock:

    sampling = False

    instances = []

    iterated_instances = []
    scores = []

    __x_list = []
    __y_list = []

    best_score = None

    def init_instances(self, max_instances):
        pass

    def add_train_test_split(self, x, y):
        self.__x_list.append(deepcopy(x))
        self.__y_list.append(deepcopy(y))

    def iterate_datasets(self, sample_size):

        self.scores = deepcopy(self.__x_list[0])
        self.iterated_instances = deepcopy(self.__x_list[0])

        self.scores[0] += 1
        self.best_score = max(self.scores)

        self.__x_list.append(deepcopy(self.__x_list[0]))
        self.__y_list.append(deepcopy(self.__y_list[0]))

        return deepcopy(self.__x_list), deepcopy(self.__y_list)

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

    def test_train_on_mocks(self, time_budget=21):
        steps = [StepMock(), StepMock(), StepMock()]
        p = Pipeline(steps, time_budget)

        data = list(range(5, 0, -1))
        best_score = p.train(data)

        self.assertEqual(best_score, 5 + 3)

        self.assertEqual(p.steps[0].scores[0], 6)
        self.assertEqual(p.steps[1].scores[0], 7)
        self.assertEqual(p.steps[2].scores[0], 8)

        self.assertEqual(p.steps[0].best_score, 6)
        self.assertEqual(p.steps[1].best_score, 7)
        self.assertEqual(p.steps[2].best_score, 8)

        self.assertEqual(p.steps[0].scores[0], 6)

    # @unittest.expectedFailure
    def test_zero_time_budget(self):
        self.test_train_on_mocks(0)

    @unittest.skip
    def test_run(self):
        steps = [Step()]*3
        steps[2].scorer = ScorerMock()
        p = Pipeline(steps)

        data = 1
        res = p.run(data)
        # print(res)
        # self.assertIsNotNone(res[0])
        self.assertEqual(res, [3, 4, 3, 4])


if __name__ == '__main__':
    unittest.main()