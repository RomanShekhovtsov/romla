import random
import numpy as np
import math
from copy import deepcopy
from enum import Enum
from functools import partial

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from model import Model
from log import *


class EliminationPolicy(Enum):
    BEST_FRACTION = 'best_fraction'
    ONE_BEST = 'one_best'

# iteration results
class StepResult:

    def __init__(self, instance, x=None, y=None):
        self.base_instance: Model = instance
        self.instance: Model = None
        self.x = x
        self.y = y
        self.score = -math.inf

    def fit(self, x, y):
        return self.instance.fit(x, y=y), y

    def predict(self, x):
        return self.instance.predict(x)


# Abstract step in AutoML pipeline
# Responsibilities:
# 1. Train/test split for each input dataset
# 2. Initiate stepInstances step_results for iteration
# 2. Iterate all input datasets through all step_results
# 3. Instances elimination after each cycle.
class Step:

    def __init__(self, models, scorer=None, elimination_policy=EliminationPolicy.BEST_FRACTION):

        self.models: list[Model] = models
        self.elimination_policy = elimination_policy

        self.scoring = scorer is not None
        self.__scorer = scorer

        self.step_results: list[StepResult] = []

        # self.x_outputs = []
        # self.y_outputs = []
        # self.scores = []

        self.best_score = None
        self.best_model : Model = None

    # for given sample size, iterate all input datasets through all stepInstances
    # return output datasets (when sample size == dataset size) and save scores
    # TODO: clean data after iteration
    def iterate(self,
                x_train,
                y_train=None,
                x_test=None,
                y_test=None,
                is_subsampling=False,
                disable_elimination=False,
                time_budget=math.inf):

        start_time = time.time()

        # params validation
        if self.__scorer is None and is_subsampling:
            raise Exception('FOR SUBSAMPLING WE NEED SCORER')

        # iterate sample through step_results
        for index in range(len(self.step_results)):
            step_result = self.step_results[index]
            instance = deepcopy(step_result.base_instance)
            step_result.instance = instance

            if not is_subsampling:
                instance.set_final_params()

            x_output, y_output = instance.fit(x_train, y_train)

            if is_subsampling:
                # dataset not proceed, clear output
                x_output = None
                y_output = None

            step_result.x = x_output
            step_result.y = y_output

            if self.scoring and x_test is not None:
                # save scores
                prediction = instance.predict(x_test)
                step_result.score = self.__scorer(y_test, prediction)
                log('{}/{} fit {} {}, score: {}'.format(
                    index + 1,
                    len(self.step_results),
                    instance.get_name(),
                    instance.params,
                    step_result.score))
            else:
                log('fit {} {}'.format(instance.get_name(), instance.params))

            elimination_policy = self.elimination_policy
            work_time = time.time() - start_time
            if work_time > time_budget:
                log('stop iterating models (time budget {} exceed)'.format(time_budget))
                break  # will use previous iteration score for non-proceed instances

        # eliminate results
        if self.scoring and not disable_elimination:
            proceed_instances_fraction = index / len(self.step_results)
            survive_fraction = min(0.5, proceed_instances_fraction)
            self.eliminate_by_score(survive_fraction=survive_fraction)

        return self.step_results

    def init_instances_hyperopt(self,
                                x_train,
                                y_train,
                                x_test,
                                y_test,
                                time_budget):
        start_time = time.time()
        self.step_results = []

        def __hyperopt_objective(params):
            model.set_params(params)
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            loss = -self.__scorer(y_test, prediction)
            return {'loss': loss,
                    'status': STATUS_OK,
                    'params': params,
                    'estimator': model.wrapper.estimator}

        # iterate sample through step_results
        for index in range(len(self.models)):

            trials = Trials()
            log('{}/{} hyperopt {}'.format(index + 1, len(self.models), self.models[index].get_name()))

            #batch_size = 10
            #while True:
            for trial_index in range(150):
                # trial_index = len(trials.trials)

                model = deepcopy(self.models[index])
                fmin(fn=__hyperopt_objective,
                     space=model.wrapper.hyperopt_param_space,
                     algo=partial(tpe.suggest, n_startup_jobs=10),
                     max_evals=trial_index + 1,
                     trials=trials)

                result = trials.results[trial_index]
                model.set_params(result['params'])
                model.wrapper.estimator = result['estimator']
                step_result = StepResult(model)
                step_result.instance = model
                step_result.score = -result['loss']
                # TODO: step_result.x, step_result.y

                self.step_results.append(step_result)
                log('{}/{} hyperopt iteration {}, params: {}, score: {}'.format(
                    index + 1,
                    len(self.models),
                    int(trial_index + 1),
                    model.params,
                    step_result.score)
                )

                work_time = time.time() - start_time
                if work_time > time_budget * (index + 1) / len(self.models):
                    break

            if work_time > time_budget:
                break

        return self.step_results

    # initiate step_results
    def init_instances(self, max_instances):

        self.step_results = []
        instances_left = max_instances
        models_count = len(self.models)

        for i in range(models_count):

            model = self.models[i]

            if callable(getattr(model, 'param_space_cardinality')):

                # calculate step_results count for this model
                instances_for_model = int(instances_left / (models_count - i))
                instances_for_model = min(instances_for_model, model.param_space_cardinality())

                for j in range(instances_for_model):
                    instance = model.new_instance()
                    params = model.sample_param_space()
                    instance.set_params(params)
                    self.step_results.append(StepResult(instance))
                    instances_left -= 1
            else:
                self.step_results.append(StepResult(deepcopy(model)))

    # eliminate step_results by score
    def eliminate_by_score(self, survive_fraction=0.5, elimination_policy=None):

        if elimination_policy is None:
            elimination_policy = self.elimination_policy

        step_results = self.step_results
        # scores = list(map(lambda x: -np.inf if x.score is None else x.score, step_results))
        scores = list(map(lambda x: x.score, step_results))
        #log('scores:', scores)

        best_index = np.argmax(scores)
        self.best_model = step_results[best_index].instance
        self.best_score = step_results[best_index].score

        survived = []

        if elimination_policy == EliminationPolicy.BEST_FRACTION:
            new_len = int(len(step_results) * survive_fraction)
            new_len = max(new_len, 1)  # at least one dataset must survive
            survived = sorted(step_results, key=lambda x: -x.score)[:new_len]

        elif elimination_policy == EliminationPolicy.ONE_BEST:
            survived = [step_results[best_index]]

        else:
            raise Exception('UNKNOWN ELIMINATION POLICY: ' + elimination_policy)

        self.step_results = survived
        log('elimination: {} of {} step_results survived'.format(len(survived), len(scores)))

        return survived
