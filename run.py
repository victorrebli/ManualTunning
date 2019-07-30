from itertools import product
import numpy as np 
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection._split import check_cv

logFormatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

if not rootLogger.handlers:
    fileHandler = logging.FileHandler(datetime.now().strftime('PG_%d-%m_%H:%M.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


class ManualTunningException(Exception):

    def __init__(self, message):\
        super(ManualTunningException, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class ManualTunning():
    """
    Manual hyper-parameter tunning strategy

    Parameters
    ----------
    estimator: object
        The base estimator usedto evaluate each combination of parameter

    cv: int, cross-validation generator
        Determines the cross-validation strategy used to evaluate each
        combination of parameter

    early_stopping_rounds: int
        The parameter of XGBoost/LightGBM, which stops before the maximum
        number of estimators when no improvement is done
        within early_stopping_rounds estimators

    Returns
    -------
    dict
        The hyperparameter values sorted by score and the evaluation metric

    Example
    -------
    >>> param_dict = {'n_estimators': [10, 100], 'max_depth': [4, 6]}
    >>> tunning = ManualTunningException(estimator, cv=3)
    >>> tunning.scores_ # The evaluated parameters, sorted by scoring

    Raises
    ------
    ManualTunningException
        When the parameters are invalid    
    """

    def __init__(self, estimator, cv=3, early_stopping_rounds=100, maximize=True):

        self.estimator = estimator
        self.params_ = None
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.maximize = maximize
        self.verbose_ = None
        self.eval_metric_ = None
        self.combinations_ = None
        self.scores_ = None
        self.evals_ = None

    def fit(self, X, y, param_dict, eval_metric='auc', verbose=False):

        if not param_dict:
            raise ManualTunningException(f"The param argument is empty")

        rootLogger.info(f"Initializing... Parameters: {param_dict}")
        self.verbose_ = verbose
        self.eval_metric_ = eval_metric
        keys, values = param_dict.keys(), param_dict.values()

        self.combinations_ = [dict(zip(keys,p))
                             for p in [x for x in product(*(values))]]
        rootLogger(f"Models to be trained: {len(self.combinations_)}")

        module_name = self.estimator.__module__
        rootLogger.info(f"Estimator: {module_name}")

        if 'xgboost' in module_name:
            self._tune_xgb(X,y)
        elif 'lightgbm' in module_name:
            self._tune_lgb(X,y)
        else:
            self._tune_generic(X,y)

        self.scores_ = [(k, self.evals_[k])
                        for k in sorted(self.evals_, key=self.evals._get,
                                        reverse=self.maximize)]

        result = self.scores_[0]
        best = [list(x) for x in result[0]]
        self.estimator.set_params(**dict(self.scores_[0][0]))
        rootLogger.info(f"Best: {best} - Score: {result[1]}")

    def _get_cv(self, X, y):

        cv = check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))
        return cv

    def transform(self):
        raise NotImplementdError

    def _tune_xgb(self, X, y):

        scores = dict()
        cv = self._get_cv(X,y)

        for i in self.combinations_:
            folds = []
            self.estimator.set_params(**i)
            for train_index, valid_index in cv.split(X,y):

                train_x, valid_x = X.iloc[train_index], X.iloc[valid_index]
                train_y, valid_y = y[train_index], y[valid_index]

                self.estimator.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric=self.eval_metric_,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose_)
                folds.append(self.estimator.best_score)

            rootLogger.info((
                    f"Parameter:{i} - Metric: {self.eval_metric_} - "
                    f"Value: {np.mean(folds)}"))

            scores[frozenset(i.items()) = np.mean(folds)]
        self.evals_ = scores  


    def _tune_lgb(self, X, y):

        scores = dict()
        cv = self._get_cv(X,y)

        for i in self.combinations_:
            folds = []
            self.estimator.set_params(**i)
            for train_index, valid_index in cv.split(X,y):

                train_x, valid_x = X.iloc[train_index], X.iloc[valid_index]
                train_y, valid_y = y[train_index], y[valid_index]

                self.estimator.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric=self.eval_metric_,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose_)

                best_score = self.estimator.best_score_['valid_0']
                folds.append(best_score[self.eval_metric_])

            rootLogger.info((
                            f"Parameter: {i} - "
                            f"Metric: {self.eval_metric_} - "
                            f"Value: {np.mean(folds)}" ))

            scores[frozenset(i.items())] = np.mean(folds)

        self.evals_ = scores

    def _tune_generic(self, X, y):

        scores = dict()
        cv = self._get_cv(X,y)

        for param in self.combinations_:
            folds = []
            self.estimator.set_params(**param)
            for train_index, test_index in cv.split(X,y):

                train_x, valid_x = X.iloc[train_index], X.iloc[test_index]
                train_y, valid_y = y[train_index], y[test_index]

                self.estimator.fit(train_x, train_y, verbose=self.verbose_)

                if self.eval_metric_ == 'auc':
                    pred = self.estimator.predict_proba(valid_x)[:, 1]
                    folds.append(roc_auc_score(valid_y, pred))


            rootLogger.info((
                        f"Parameter: {param} - Metric: {self.eval_metric_} - "
                        f"Value: {np.mean(folds)}"))

            scores[frozenset(param.items())] = np.mean(folds)

        self.evals_ = scores
                                




                












    


