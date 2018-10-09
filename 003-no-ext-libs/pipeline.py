

class Pipeline:

    steps = []

    def __init__(self, steps=[]):
        self.steps = steps

    def run(self, data):
        for step in self.steps:
            step.run(data)
        return data