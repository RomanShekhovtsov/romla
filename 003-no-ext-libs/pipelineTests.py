import unittest
from pipeline import *
from stepTests import *


class PipelineTest(unittest.TestCase):

    def test_run(self):
        steps = [StepTest()]*10
        p = Pipeline(steps)

        data = [1]
        res = p.run(data)
        #self.assertIsNotNone(res[0])
        self.assertEqual(res[0], 10 + 1)


if __name__ == '__main__':
    unittest.main()