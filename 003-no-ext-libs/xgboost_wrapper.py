import xgboost as xgb

from model import Model, ParamRuleType


class XGBoostWrapper:

    def __init__(self):
        self.const_params = {'n_jobs': 4}
        self.param_space = {'max_depth': list(range(2, 16))}
        self.param_rules = {'max_depth': ParamRuleType.SEQUENCE}
        self.estimator = None

    def get_regressor(self):
        self.estimator = xgb.sklearn.XGBRegressor()
        return self.get_model()

    def get_classifier(self):
        self.estimator = xgb.sklearn.XGBClassifier()
        return self.get_model()

    def get_model(self):
        self.estimator.set_params(**self.const_params)
        return Model(self, self.param_space, self.param_rules)

    def set_params(self, params):
        self.estimator.set_params(**params)

    def fit(self, x, y=None):
        return self.estimator.fit(x, y=y), y

    def predict(self, x):
        return self.estimator.predict(x)


