import time
from copy import deepcopy
from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

from utils import *


class ModelParamsSearchStrategy(Enum):
    GRID_SEARCH = 'random'
    FIRST_BEST = 'first_best'


class Model:

    base_estimator = None
    wrapper = None

    estimators = []

    def __init__(self, estimator, wrapper):
        self.base_estimators = estimator
        self.algorithm = wrapper

    def get_name(self):
        return self.base_estimator.__class__.__name__

    def iterate_models(models, X, y, scoring, min_train_rows=MIN_TRAIN_ROWS):

        scores = []
        speeds = []
        rows = y.shape[0]

        full_train = rows < min_train_rows

        for model in models:

            log_start()
            score, speed = model.evaluate(model, X, y, scoring, full_train)
            scores.append(score)
            speeds.append(speed)

            if scoring == NEG_MEAN_SQUARED_ERROR:
                print_score = (-score) ** 0.5
            else:
                print_score = score

            log_time('evaluate model, full train: {}; rows: {}; score:{};'.format(full_train, rows, print_score))
            log(model)

        return scores, speeds

    def params_search(self, X, y, scoring, speed):
        scores = []
        rows = y.shape[0]

        estimator_name = self.get_name()

        best_estimator = None

        if self.strategy == ModelParamsSearchStrategy.GRID_SEARCH:

            # n_iter = 0
            # for param_name in params.keys():
            #     param_values = params[param_name]
            #     n_iter += len(param_values)

            estimator = deepcopy(self.base_estimator)  # TODO: excessive copy for gridsearch?

            searcher = GridSearchCV(estimator,
                                    self.param_grid,
                                    scoring=scoring,
                                    n_jobs=4,
                                    cv=3,
                                    return_train_score=False)
            searcher.fit(X, y)
            best_estimator = searcher.best_estimator_

        elif self.strategy == ModelParamsSearchStrategy.FIRST_BEST:

            test_size = 0.25

            samples, time_to_fit_all = self.calc_sample_size(test_size, X.shape[0], speed['fit'], speed['predict'], n_iter)
            log('train fit estimation: {}; sample size: {}'.format( time_to_fit_all, samples))
            X_train, X_test, y_train, y_test = train_test_split(X[:samples], y[:samples], test_size=test_size)

            param_name = list(self.grid_params.keys())[0]  # only one param for this strategy
            param_values = self.grid_params[param_name]
            prev_score = None

            iteration_time = 0
            best_estimator = self.base_estimator
            for param_value in param_values:

                if time_left() < time_to_fit_all + iteration_time:
                    log('stop model params search due to time limit:',
                        'time left={}; time to fit all={}; last iteration time: {}'.format(
                        time_left(), time_to_fit_all, iteration_time))
                    break

                iteration_time = time.time()
                estimator = deepcopy(self.base_estimator)
                est_params = {param_name: param_value}
                log('estimate', est_params)
                estimator.set_params( **est_params)

                estimator.fit(X_train, y_train)
                fit_time = time.time() - iteration_time
                fit_speed = X_train.shape[0] / fit_time

                predict = estimator.predict(X_test)
                predict_time = time.time() - fit_time
                predict_speed = X_test.shape[0] / predict_time

                score = self.calc_score(scoring, y_test, predict)

                if (not prev_score is None) and (prev_score > score):
                    log('early finish at {}={}'.format(param_name, param_value))
                    break

                best_estimator = estimator
                prev_score = score
                iteration_time = time.time() - iteration_time

        return best_estimator

        # if model_name == 'GradientBoostingRegressor':
        #
        #     # def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
        #     #              subsample=1.0, criterion='friedman_mse', min_samples_split=2,
        #     #              min_samples_leaf=1, min_weight_fraction_leaf=0.,
        #     #              max_depth=3, min_impurity_decrease=0.,
        #     #              min_impurity_split=None, init=None, random_state=None,
        #     #              max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
        #     #              warm_start=False, presort='auto'):
        #     strategy = ModelParamsSearchStrategy.GRID_SEARCH
        #     # estimator = GradientBoostingRegressor()
        #     params = {'learning_rate': [0.05, 0.1, 0.15],
        #               'min_samples_split': [2, 4, 8],
        #               'min_samples_leaf': [1, 2, 4],
        #               'max_depth': [2, 3, 4]
        #               }
        #     const_params = {}
        #     # n_iter = 81

            # def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1,
            #              learning_rate=0.1, n_estimators=100,
            #              subsample_for_bin=200000, objective=None, class_weight=None,
            #              min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
            #              subsample=1., subsample_freq=0, colsample_bytree=1.,
            #              reg_alpha=0., reg_lambda=0., random_state=None,
            #              n_jobs=-1, silent=True, importance_type='split', **kwargs):
            #

        # elif model_name == 'RandomForestRegressor':
            # n_estimators = 10,
            # criterion = "mse",
            # max_depth = None,
            # min_samples_split = 2,
            # min_samples_leaf = 1,
            # min_weight_fraction_leaf = 0.,
            # max_features = "auto",
            # max_leaf_nodes = None,
            # min_impurity_decrease = 0.,
            # min_impurity_split = None,
            # bootstrap = True,
            # oob_score = False,
            # n_jobs = 1,
            # random_state = None,
            # verbose = 0,
            # warm_start = False):
            # strategy = ModelParamsSearchStrategy.GRID_SEARCH
            # params = {'min_samples_split': [2, 4, 8],
            #           'max_features': ['auto', 'sqrt', 'log2'],
            #           'min_samples_leaf': [1, 2, 4],
            #           'max_depth': [3, 5, 7]
            #           }
        #
        # elif model_name == 'ExtraTreesRegressor':
            # n_estimators = 10,
            # criterion = "mse",
            # max_depth = None,
            # min_samples_split = 2,
            # min_samples_leaf = 1,
            # min_weight_fraction_leaf = 0.,
            # max_features = "auto",
            # max_leaf_nodes = None,
            # min_impurity_decrease = 0.,
            # min_impurity_split = None,
            # bootstrap = False,
            # oob_score = False,
            # n_jobs = 1,
            # random_state = None,
            # verbose = 0,
            # warm_start = False):
            # strategy = ModelParamsSearchStrategy.GRID_SEARCH
            # estimator = ExtraTreesRegressor()
            # params = {'min_samples_split': [2, 4, 8],
            #           'max_features': ['auto', 'sqrt', 'log2'],
            #           'min_samples_leaf': [1, 2, 4],
            #           'max_depth': [3, 5, 7]
            #           }
            #
        # elif model_name == 'AdaBoostRegressor':
        # elif model_name == 'GradientBoostingClassifier':
        # elif model_name == 'RandomForestClassifier':
        # elif model_name == 'ExtraTreesClassifier':
        # elif model_name == 'AdaBoostClassifier':
        # elif model_name == 'LinearSVC':
        # elif model_name == 'SVC':

        # def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
        #              fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
        #              verbose=0, pre_dispatch='2*n_jobs', random_state=None,
        #              error_score='raise', return_train_score="warn"):