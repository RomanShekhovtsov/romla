from typing import Dict, Any, Union
from copy import deepcopy

import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier, FeaturesData, Pool
from hyperopt import hp

from model import Model, ParamRuleType


class CatBoostWrapper:

    params: Dict[Union[str, Any], Union[Union[str, float, int], Any]]

    def __init__(self):
        self.base_params = dict()
        self.base_params["iterations"] = 400
#        self.base_params["used_ram_limit"] = '512mb'
        self.base_params["one_hot_max_size"] = 10
        self.base_params["nan_mode"] = 'Min'
        self.base_params["depth"] = 5
        self.base_params["learning_rate"] = 0.01
        self.base_params["random_strength"] = 1.5
        self.base_params["bagging_temperature"] = 1.5

        self.params = deepcopy(self.base_params)

        max_depth_list = (list(range(2, 11)))

        self.param_space = {
            'depth': max_depth_list,
            'learning_rate': (0.005, 0.01, 0.05, 0.1, 0.3)
        }
        self.param_rules = {}

        self.hyperopt_param_space = {
            # 'num_leaves': hp.choice('num_leaves', [5,10,20,30,50,70,100]),
            # 'subsample': hp.choice('subsample', [0.7,0.8,0.9,1]),
            # 'colsample_bytree': hp.choice('colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1]),
            # 'min_child_weight': hp.choice('min_child_weight', [5,10,15,20,30,50]),
            # 'learning_rate': hp.choice('learning_rate', [0.02,0.03,0.05,0.07,0.1,0.2])

            'depth': hp.choice('depth', max_depth_list),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3))
        }

        self.estimator = None
        self.mode = None
        self.category_indices = None

    def get_regressor(self, category_indices):
        self.params["loss_function"] = "RMSE"
        self.mode = 'regression'
        self.estimator = CatBoostRegressor(**self.params)
        return self.get_model(category_indices)

    def get_classifier(self, category_indices):
        self.mode = 'classification'
        self.estimator = CatBoostClassifier(**self.params)
        return self.get_model(category_indices)

    def get_model(self, category_indices):
        # self.estimator.set_params(**params)
        self.category_indices = category_indices
        return Model(self, self.param_space, self.param_rules)

    def set_params(self, params):
        for key in params.keys():
            self.params[key] = params[key]
        self.estimator.set_params(**self.params)

    def set_final_params(self):
        pass
        # self.set_params({'learning_rate': 0.001})
        # self.num_iterations = 600
        # self.set_params({'learning_rate': 0.01})

    def fit(self, x, y=None):
        if self.mode == 'classification':
            pos_weight = x[y < 0.5].shape[0] / x[y > 0.5].shape[0]
            self.set_params({"scale_pos_weight": pos_weight})

        self.estimator.fit(x, y, logging_level='Silent', use_best_model=True)

    def predict(self, x):
        return self.estimator.predict(x)