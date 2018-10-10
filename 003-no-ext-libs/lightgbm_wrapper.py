import lightgbm as lgb

from model import Model, ModelParamsSearchStrategy


class LightGBMWrapper:

    base_params = {'boosting_type': 'gbdt',
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    'num_leaves': 200,
                    "feature_fraction": 0.70,
                    "bagging_fraction": 0.70,
                    'bagging_freq': 4,
                    'max_depth': -1,
                    'verbosity': -1,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.1,
                    # "min_split_gain":0.2,
                    "min_child_weight": 10,
                    'zero_as_missing': True,
                    'num_threads': 4,
                    }
    params = {'max_depth': list(range(2, 16))}
    strategies = {'max_depth': ModelParamsSearchStrategy.FIRST_BEST}
    # n_iter = 15

    def get_regressor(self):
        est = lgb.sklearn.LGBMRegressor()
        params = self.base_params.copy()
        params['objective'] = 'regression'
        est.set_params(**params)
        return self.get_model(est)

    def get_classifier(self):
        est = lgb.LGBMClassifier()
        params = self.base_params.copy()
        params['objective'] = 'binary'  # TODO: multiclass support
        est.set_params(**params)
        return self.get_model(est)

    def get_model(self, est):
        return Model(est, self)
