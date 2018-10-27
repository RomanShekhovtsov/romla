import time

from sklearn.metrics import mean_squared_error, roc_auc_score

from utils import *

MIN_NUMBER = 1e-10  #small number to prevent division by zero
NEG_MEAN_SQUARED_ERROR = 'neg_mean_squared_error'


class Estimator():

    __estimator = None

    def __init__(self, estimator):
        self.__estimator = estimator

    def evaluate(self, X, y, scoring, full_train):
        speed = dict()

        if full_train:
            rows = X.shape[0]

            t = time.time()
            self.__estimator.fit(X, y=y)
            fit_speed = int(rows / (time.time() - t + MIN_NUMBER))

            t = time.time()
            prediction = self.__estimator.predict(X)
            predict_speed = int(rows / (time.time() - t + MIN_NUMBER))

            score = self.calc_score(scoring, y, prediction)

        # if rows < min_train_rows:
        #     cv = math.ceil((min_train_rows / rows)) + 1  # make X_train_rows >= rows * (nfolds - 1)
        #     cv = min(rows, cv)  # correction for extra-small datasets
        #     method = 'cross validation ' + str(cv) + '-folds'
        #     score = np.mean(cross_val_score(wrapper, X, y=y, scoring=scoring, cv=cv, n_jobs=N_JOBS))

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_TEST_SIZE)

            t = time.time()
            self.__estimator.fit(X_train, y=y_train)
            speed['fit'] = int(X_train.shape[0] / (time.time() - t))

            t = time.time()
            prediction = self.__estimator.predict(X_test)
            speed['predict'] = int(X_test.shape[0] / (time.time() - t))

            score = self.calc_score(scoring, y_test, prediction)

        return score, speed

    # calculate sample size (rows) to perform wrapper parameters search
    def calc_sample_size(self, test_size, total_rows, fit_speed, predict_speed, n_iter):

        time_to_fit_all = 3 * total_rows / fit_speed  # 3 - empirical coeff.
        time_to_search = time_left() - time_to_fit_all
        time_iteration = time_to_search / n_iter

        # equation for sample calculated from: max t_iteration = train_rows/Sfit + test_rows/Spredict
        # train_rows = (1-test_size)*sample
        # test_rows = test_size*sample
        sample_size = time_iteration / ((1 - test_size) / fit_speed + test_size / predict_speed)
        sample_size = min(sample_size, total_rows)

        return max(int(sample_size), 0), time_to_fit_all

    def calc_score(self, scoring, y_test, prediction):

        if scoring == NEG_MEAN_SQUARED_ERROR:
            score = -mean_squared_error(y_test, prediction)
        else:
            score = roc_auc_score(y_test, prediction)

        return score
