import unittest
from pipeline import *
from stepTests import *


class StepMock:

    def instances(self):
        n = 0
        while n < 2:
            yield InstanceMock()
            n += 1


class InstanceMock:

    def fit(self, data):
        return data + 1


class ScorerMock:

    def score(self, data):
        return data


class PipelineTest(unittest.TestCase):

    def test_run_on_mocks(self):
        steps = [StepMock()]*3
        p = Pipeline(steps, ScorerMock())

        data = 1
        res = p.run(data)
        # print(res)
        # self.assertIsNotNone(res[0])
        self.assertEqual(res, [3, 4, 3, 4])

    @unittest.expectedFailure
    def test_run(self):
        steps = [Step()]*3
        p = Pipeline(steps, ScorerMock())

        data = 1
        res = p.run(data)
        # print(res)
        # self.assertIsNotNone(res[0])
        self.assertEqual(res, [3, 4, 3, 4])


if __name__ == '__main__':
    unittest.main()