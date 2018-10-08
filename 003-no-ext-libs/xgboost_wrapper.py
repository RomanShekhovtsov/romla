import xgboost as xgb

from model import Model, ModelParamsSearchStrategy


class XGBoostWrapper:

    const_params = {'n_jobs': 4}
    strategy = ModelParamsSearchStrategy.FIRST_BEST
    param_grid = {'max_depth': list(range(2, 16))}
    n_iter = 15

    def get_regressor(self):
        est = xgb.sklearn.XGBRegressor()
        return self.get_model(est)

    def get_classifier(self):
        est = xgb.sklearn.XGBClassifier()
        return self.get_model(est)

    def get_model(self, est):
        est.set_params(**self.const_params)
        return Model(est, self.param_grid, self.strategy, self.n_iter)

