import unittest
import numpy as np
from copy import deepcopy

from pipeline import *
from step import StepResult
from stepTests import *


# class StepMock:
#
#     def __init__(self, models, scorer=None, elimination_policy='median'):
#
#         self.models = models
#         self.scorer = scorer
#         self.scoring = scorer is not None
#         self.elimination_policy = elimination_policy
#
#         self.step_results = []
#
#     def init_instances(self, max_instances):
#         self.step_results = self.models
#
#     def iterate(self, x_train, y_train, x_test=None, y_test=None, is_subsampling=False):
#
#         x = deepcopy(x_train)
#         iteration_results = []
#
#         for instance in self.step_results:
#             iteration_result = StepInstance(instance, list(map(instance, x)), deepcopy(y_train))
#
#             if self.scorer is not None:
#                 iteration_result.score = sum(list(map(self.scorer, iteration_result.x, y_train)))
#                 #print(iteration_result.x, y_train)
#             iteration_results.append(iteration_result)
#             #log(iteration_result.x)
#
#         return iteration_results
#
#     # def step_results(self):
#     #     n = 0
#     #     while n < 2:
#     #         yield InstanceMock()
#     #         n += 1


class TransformMock:

    def __init__(self):
        self.params = []

    def fit(self, x, y):
        print(x, y)
        return np.add(x, 1)

    def set_params(self, params):
        self.params = params


class EstimateMock:

    def __init__(self):
        self.params = []

    def fit(self, x, y=None):
        # print(x, y)
        return x

    def predict(self, x, y=None):
        shift = self.params['shift']
        print('shift:', shift)
        # print(x, y)
        return np.mod(np.add(x, -shift), 7)

    def set_params(self, params):
        self.params = params


class PipelineTest(unittest.TestCase):

    # def test_train_on_mocks(self, time_budget=22, rows=100):
    #
    #     steps_count = 3
    #
    #     x = np.arange(rows)
    #     y = x % 2
    #
    #     transform_model = lambda x: x + 1
    #     steps = []
    #     for i in range(steps_count-1):
    #         steps.append(StepMock([transform_model]))
    #
    #     predict_models = [lambda x: x % 2,
    #                       lambda x: (x - 1) % 2,
    #                       lambda x: (x - 2) % 2
    #                       ]
    #     scorer = lambda x, y: 1 if x == y else 0
    #     steps.append(StepMock(predict_models, scorer=scorer))
    #
    #     p = Pipeline(steps, time_budget)
    #     best_score = p.train(x, y)
    #
    #     self.assertEqual(rows * (1 - TEST_SIZE), best_score)
    #     #self.assertEqual(p.best_pipeline.score, best_score)
    #
    #     return p

    def test_train(self, time_budget=22, rows=100):

        steps_count = 3

        x = np.arange(rows)
        y = x % 7

        steps = []
        for i in range(steps_count-1):
            model = Model(TransformMock(), {}, {})
            steps.append(Step([model]))

        predict_models = []
        for shift in range(3):
            model = Model(EstimateMock(),
                          param_space={'shift': list(range(shift + 1))},
                          param_rules={'shift': ParamRuleType.SEQUENCE})
            predict_models.append(model)

        self.assertEqual(1, predict_models[0].param_space_cardinality())
        steps.append(Step(predict_models, scorer=neg_mean_squared_error))

        p = Pipeline(steps, time_budget)
        best_score = p.train(x, y)

        self.assertEqual(0, best_score)
        #self.assertEqual(p.best_pipeline.score, best_score)

        return p

    @unittest.skip
    def test_zero_time_budget_success(self):
        self.test_train(0, 1000)

    @unittest.skip
    @unittest.expectedFailure
    # must fail after fist sample
    def test_zero_time_budget_exceed(self):
        self.test_train(0, INITIAL_SAMPLE_SIZE + 1)

    @unittest.skip
    def test_big(self):
        self.test_train(30, 100000)

    def test__iterate_step(self):

        p: Pipeline = self.test_train()

        #pipeline_instances = p.iterate_step(0, [p.best_pipeline], is_subsampling=False)
        #self.assertEqual(len(pipeline_instances), 1)

        #pipeline_instances = p.iterate_step(1, [p.best_pipeline], is_subsampling=False)
        #self.assertEqual(len(pipeline_instances), 1)

        #pipeline_instances = p.iterate_step(2, [p.best_pipeline], is_subsampling=False)
        #self.assertEqual(len(pipeline_instances), 2)

    # @staticmethod
    # def test_csv_loader():
    #     model_config = []
    #     loader = CsvLoader(None, model_config)
    #     step = Step([loader])
    #     p = Pipeline([step], 22)
    #     p.train('E:\\ds\\sdsj\\examples\\sdsj2018_lightgbm_baseline\\res\\check_8_c\\train.csv')
    #
    #     print(step.step_results[0].x)


if __name__ == '__main__':
    unittest.main()