import random


class Step:

    models = None
    scorer = None
    elimination_policy = None
    samples = None

    def __init__(self, models, scorer=None, elimination_policy=None, samples=None):

        # params validation
        if scorer is None and samples is not None:
            raise Exception('FOR SUBSAMPLING WE NEED SCORER')

        self.models = models
        self.scorer = scorer
        self.elimination_policy = elimination_policy
        self.samples = samples

    def instances(self):
        params = self.models.sample_param_space()
        instance = self.model.get_instance
        instance.set_params(params)
        yield instance

    def sample_param_space(self):
        param_sample = {}
        for key in self.model.param_space.keys():
            param_sample[key] = self.sample_param(self.param_space[key])
        yield param_sample

    def sample_param(self, param_distribution):
        # TODO: sampling types
        distribution_index = random.randint(1, len(param_distribution)) - 1
        return param_distribution[distribution_index]