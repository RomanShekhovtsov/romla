import unittest
import numpy as np
from copy import deepcopy

from pipeline import *
from step import StepInstance
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
            iteration_result = StepInstance(instance, list(map(instance, x)), deepcopy(y_train))

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


class TransformMock(Model):

    def fit_transform(self, x, y):
        return np.add(x, 1)


class ModelMock(Model):

    def __init__(self, shift):
        super(ModelMock, self).__init__(None, {'shift': [0, 1, 2]})
        self.shift = shift

    def fit_transform(self, x, y=None):
        return np.mod(np.add(x, -self.shift), 2)


class ScorerMock:

    def score(self, x, y):
        return np.sum(x - y)


class PipelineTest(unittest.TestCase):

    def test_train_on_mocks(self, time_budget=22, rows=100):

        steps_count = 3

        x = np.arange(rows)
        y = x % 2

        transform_model = lambda x: x + 1
        steps = []
        for i in range(steps_count-1):
            steps.append(StepMock([transform_model]))

        predict_models = [lambda x: x % 2,
                          lambda x: (x - 1) % 2,
                          lambda x: (x - 2) % 2
                          ]
        scorer = lambda x, y: 1 if x == y else 0
        steps.append(StepMock(predict_models, scorer=scorer))

        p = Pipeline(steps, time_budget)
        best_score = p.train(x, y)

        self.assertEqual(best_score, rows * (1 - TEST_SIZE))
        #self.assertEqual(p.best_pipeline.score, best_score)

        return p

    def test_train(self, time_budget=22, rows=100):

        steps_count = 3

        x = np.arange(rows)
        y = x % 2

        steps = []
        for i in range(steps_count-1):
            steps.append(Step([TransformMock(None, {})]))

        predict_models = [ModelMock(0),
                          ModelMock(1),
                          ModelMock(2)
                          ]
        steps.append(Step(predict_models, scorer=ScorerMock()))

        p = Pipeline(steps, time_budget)
        best_score = p.train(x, y)

        self.assertEqual(best_score, rows * (1 - TEST_SIZE))
        #self.assertEqual(p.best_pipeline.score, best_score)

        return p

    @unittest.skip
    def test_zero_time_budget_success(self):
        self.test_train_on_mocks(0, 1000)

    @unittest.skip
    @unittest.expectedFailure
    # must fail after fist sample
    def test_zero_time_budget_exceed(self):
        self.test_train_on_mocks(0, INITIAL_SAMPLE_SIZE + 1)

    @unittest.skip
    def test_big(self):
        self.test_train_on_mocks(30, 100000)

    def test__iterate_step(self):

        p: Pipeline = self.test_train_on_mocks()

        #pipeline_instances = p.iterate_step(0, [p.best_pipeline], is_subsampling=False)
        #self.assertEqual(len(pipeline_instances), 1)

        #pipeline_instances = p.iterate_step(1, [p.best_pipeline], is_subsampling=False)
        #self.assertEqual(len(pipeline_instances), 1)

        #pipeline_instances = p.iterate_step(2, [p.best_pipeline], is_subsampling=False)
        #self.assertEqual(len(pipeline_instances), 2)


if __name__ == '__main__':
    unittest.main()