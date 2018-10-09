import unittest
from step import *


class StepTest(unittest.TestCase):

    step = None

    def setUp(self):
        self.step = Step()

    def test_run(self):
        data = [1]
        self.step.run(data)


if __name__ == '__main__':
    unittest.main()