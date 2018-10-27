import lightgbm as lgb

from model import Model, ParamRuleType


class LightGBMWrapper:

    def __init__(self):
        self.const_params = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 200,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            'bagging_freq': 4,
            'max_depth': -1,
            'verbosity': 0,
            'verbose': 0,
            'reg_alpha': 0.3,
            'reg_lambda': 0.1,
            # "min_split_gain":0.2,
            "min_child_weight": 10,
            'zero_as_missing': True,
            'num_threads': 4,
            }

        max_depth_list = [-1]
        max_depth_list.extend(list(range(2, 16)))
        self.param_space = {'max_depth': max_depth_list}
        self.param_rules = {'max_depth': ParamRuleType.SEQUENCE}
        self.estimator = None

    def get_regressor(self):
        self.estimator = lgb.sklearn.LGBMRegressor(silent=True)
        params = self.const_params.copy()
        params['objective'] = 'regression'
        return self.get_model(params)

    def get_classifier(self):
        self.estimator = lgb.LGBMClassifier(silent=True)
        params = self.const_params.copy()
        params['objective'] = 'binary'  # TODO: multiclass support
        self.estimator.set_params(**params)
        return self.get_model(params)

    def get_model(self, params):
        self.estimator.set_params(**params)
        return Model(self, self.param_space, self.param_rules)

    def set_params(self, params):
        self.estimator.set_params(**params)

    def fit(self, x, y=None):
        return self.estimator.fit(x, y=y, verbose=False), y

    def predict(self, x):
        return self.estimator.predict(x)