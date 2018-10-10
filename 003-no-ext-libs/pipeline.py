import numpy as np


class Pipeline:

    steps = None

    def __init__(self, steps):
        self.steps = steps

    def run(self, data):
        return self.__iterate_steps(0, [data])

    def __iterate_steps(self, step_index, data_list):

        step_outputs = []
        step_scores = []
        step = self.steps[step_index]

        for sample_size in step.samples:
            for instance in step.instances():
                for data in range(len(data_list)):
                    instance_data = instance.fit(data)
                    if step.scorer is not None and sample_size < step.samples[-1]:
                        # for sub-sampling save only scores
                        step_scores.append(step.scorer.score(instance_data))
                    else:
                        # save output data
                        step_outputs.append(instance_data)

        if len(self.steps) > step_index + 1:
            step_data_list = self.__iterate_steps(step_index + 1, step_data_list)

        return step_data_list

    def __eliminate_by_score(self, step, step_data_list):

        scores = []

        for data in step_data_list:
            score = step.scorer.score(data)
            scores.append(score)

        if step.elimination_policy == 'median':
            median = np.median(scores)
            result_list = []
            result_instances = []

            for i in range(len(scores)):
                if scores[i] >= median:
                    result_list.append(step_data_list[i])
                    result_instances.append(step.instances[i])
        else:
            raise Exception('UNKNOWN ELIMINATION POLICY')

        return result_list
