from typing import Dict, Any, Union

import lightgbm as lgb

from model import Model, ParamRuleType


class LightGBMWrapper:

    params: Dict[Union[str, Any], Union[Union[str, float, int], Any]]

    def __init__(self):
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            "learning_rate": 0.01,
            "num_leaves": 200,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            'bagging_freq': 4,
            "max_depth": -1,
            "verbosity" : -1,
            "reg_alpha": 0.3,
            "reg_lambda": 0.1,
            #"min_split_gain":0.2,
            "min_child_weight":10,
            'zero_as_missing':True,
            'num_threads': 4,
            }
        self.params['seed'] = 1

        max_depth_list = [-1]
        #max_depth_list.extend(list(range(2, 16)))
        self.param_space = {'max_depth': max_depth_list}
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
            self.params[key] = params[key]

    def fit(self, x, y=None):
        self.estimator = lgb.train(self.params, lgb.Dataset(x, label=y), 600)
        #return self.estimator.fit(x, y=y, verbose=False), y

    def predict(self, x):
        return self.estimator.predict(x)