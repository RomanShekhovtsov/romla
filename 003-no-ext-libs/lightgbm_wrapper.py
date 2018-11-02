from typing import Dict, Any, Union
from copy import deepcopy

import numpy as np
import lightgbm as lgb
from hyperopt import hp

from model import Model, ParamRuleType


class LightGBMWrapper:

    params: Dict[Union[str, Any], Union[Union[str, float, int], Any]]

    def __init__(self):
        self.base_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            "learning_rate": 0.01,
            "num_leaves": 200,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            'bagging_freq': 4,
            "max_depth": -1,
            "verbosity": -1,
            "reg_alpha": 0.3,
            "reg_lambda": 0.1,
            # "min_split_gain":0.2,
            "min_child_weight": 10,
            'zero_as_missing': True,
            'num_threads': 4,
            }
        self.params = deepcopy(self.base_params)
        self.params['seed'] = 1
        self.num_iterations = 600
        self.num_leaves_coeff = 0.6

        max_depth_list = [-1]
        max_depth_list.extend(list(range(2, 11)))

        self.param_space = {
            'max_depth': max_depth_list,
            'feature_fraction': np.arange(0.5, 1.0, 0.1),
            'reg_alpha': (0.2, 0.3, 0.4),
            'reg_lambda': (0.05, 0.1, 0.15)
        }

        self.hyperopt_param_space = {
            # 'num_leaves': hp.choice('num_leaves', [5,10,20,30,50,70,100]),
            # 'subsample': hp.choice('subsample', [0.7,0.8,0.9,1]),
            # 'colsample_bytree': hp.choice('colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1]),
            # 'min_child_weight': hp.choice('min_child_weight', [5,10,15,20,30,50]),
            # 'learning_rate': hp.choice('learning_rate', [0.02,0.03,0.05,0.07,0.1,0.2])

            #'max_depth': hp.choice('max_depth', max_depth_list),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
            'num_leaves_coeff': hp.uniform('num_leaves_coeff', 0.5, 0.7),
            'min_child_weight': hp.uniform('min_child_weight', 5, 15),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
        }

        self.param_rules = {'max_depth': ParamRuleType.SEQUENCE}
        self.estimator = None

    def get_regressor(self):
        self.params['objective'] = 'regression'
        return self.get_model()

    def get_classifier(self):
        self.params['objective'] = 'binary'  # TODO: multiclass support
        return self.get_model()

    def get_model(self):
        #self.estimator.set_params(**params)
        return Model(self, self.param_space, self.param_rules)

    def set_params(self, params):
        for key in params.keys():
            if key == 'num_leaves_coeff':
                self.num_leaves_coeff = params[key]
            else:
                self.params[key] = params[key]

        # with: 10.40833637689379, without: 10.470802953885036 on sdsj-1
        max_depth = params.get('max_depth', -1)
        if max_depth != -1:
            self.params['num_leaves'] = int(self.num_leaves_coeff * (2**max_depth))

    def set_final_params(self):
        pass
        # self.set_params({'learning_rate': 0.001})
        # self.num_iterations = 600
        # self.set_params({'learning_rate': 0.01})

    def fit(self, x, y=None):
        # self.set_params({'min_data_in_leaf': min_data_in_leaf})
        # min_data_in_leaf = len(x) / self.params['num_leaves'] * 2
        self.estimator = lgb.train(
            self.params,
            lgb.Dataset(x, label=y),
            num_boost_round=600
        )

    def predict(self, x):
        return self.estimator.predict(x)