import numpy as np
import xgboost as xgb
from hyperopt import hp

from model import Model, ParamRuleType


class XGBoostWrapper:

    def __init__(self):
        self.params = {'n_jobs': 4}

        self.param_space = {
            'learning_rate': np.arange(0.01, 0.2, 0.03),
            'colsample_bytree': np.arange(0.5, 1.0, 0.1)
        }

        self.hyperopt_param_space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'min_child_weight': hp.choice('min_child_weight', range(11)),
            'max_depth': hp.choice('max_depth', range(2, 11)),
            'gamma': hp.uniform('gamma', 0, 0.5),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
        }

        self.param_rules = {}  # 'max_depth': ParamRuleType.SEQUENCE}
        self.estimator = None

    def get_regressor(self):
        self.estimator = xgb.sklearn.XGBRegressor()
        return self.get_model()

    def get_classifier(self):
        self.estimator = xgb.sklearn.XGBClassifier()
        return self.get_model()

    def get_model(self):
        self.estimator.set_params(**self.params)
        return Model(self, self.param_space, self.param_rules)

    def set_params(self, params):
        for key in params.keys():
            self.params[key] = params[key]
        self.estimator.set_params(**self.params)

    def fit(self, x, y=None):
        return self.estimator.fit(x, y=y)

    def predict(self, x):
        return self.estimator.predict(x)


