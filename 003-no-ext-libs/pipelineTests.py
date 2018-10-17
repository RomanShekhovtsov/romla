import unittest
import numpy as np
from copy import deepcopy

from pipeline import *
from step import IterationResult
from stepTests import *


class StepMock:

    def __init__(self, models, scorer=None, elimination_policy=None, sampling=False):
        self.sampling = sampling

        self.models = models
        self.instances = []
        self.__x_list = []
        self.__y_list = []
        self.iteration_results = []
        self.best_score = None

    def init_instances(self, max_instances):
        pass

    def clear_train_test(self):
        self.__x_list = []
        self.__y_list = []

    def add_train_test_split(self, x, y):
        # print('add_train start',self.__x_list)
        self.__x_list.append(x)
        self.__y_list.append(y)
        # print('add_train',self.__x_list)

    def iterate_datasets(self, sample_size):

        self.__x_list[0][0] += 1
        self.scores = deepcopy(self.__x_list[0])
        self.iterated_instances = deepcopy(self.__x_list[0])

        self.best_score = max(self.scores)

        self.__x_list.append(deepcopy(self.__x_list[0]))
        self.__y_list.append(deepcopy(self.__y_list[0]))

        self.iteration_results = []
        for i in range(len(self.__x_list)):
            iteration_result = IterationResult(None, deepcopy(self.__x_list[i]), deepcopy(self.__y_list[i]))
            self.iteration_results.append(iteration_result)

        # print(self.__x_list)
        return self.iteration_results

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

    def test_train_on_mocks(self, time_budget=22):

        steps_count = 3
        rows = 10000

        steps = []
        for i in range(steps_count):
            steps.append(StepMock([],sampling=True))

        p = Pipeline(steps, time_budget)

        data = list(range(rows, 0, -1))
        best_score = p.train(data)

        self.assertEqual(best_score, rows + len(steps))

        self.assertEqual(p.steps[0].scores[0], rows + 1)
        self.assertEqual(p.steps[1].scores[0], rows + 2)
        self.assertEqual(p.steps[2].scores[0], rows + 3)

        self.assertEqual(p.steps[0].best_score, rows + 1)
        self.assertEqual(p.steps[1].best_score, rows + 2)
        self.assertEqual(p.steps[2].best_score, rows + 3)

        self.assertEqual(p.steps[0].scores[0], rows + 1)
        self.assertEqual(p.steps[1].scores[0], rows + 2)
        self.assertEqual(p.steps[2].scores[0], rows + 3)

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