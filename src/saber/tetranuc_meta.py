from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import OneClassSVM


class MLModel:

    def __init__(self, model, scoreFun=metrics.make_scorer(metrics.accuracy_score)):
        '''Constructor function
        :param model: ML model to validate
        :param scoreFun : score function used to validate models, defaults to make_scorer(metrics.accuracy_score)
        '''
        self.model = model
        self.scoreFun = scoreFun
        return

    def validateRandom(self, X_train, params, y_train=None, n_jobs=1, n_iter=10, cv=5):
        '''Validate model with training data using RandomizedSearchCV
        :param X_train: X_train
        :param y_train: y_train
        :param params: parameter in a dictionary
        :param n_jobs: number of jobs to run in parallel, default to 1
        :param n_iter: Number of parameter settings that are sampled, default to 10
        :param cv: Determines the cross-validation splitting strategy, default to 5
        '''
        randomcv = RandomizedSearchCV(self.model, param_distributions=params, n_jobs=n_jobs, n_iter=n_iter, cv=cv,
                                      scoring=self.scoreFun)
        randomcv.fit(X_train, y_train)
        self.best_param = randomcv.best_params_
        self.best_estimator = randomcv.best_estimator_

        return randomcv.best_estimator_

    def validateGrid(self, X_train, y_train, params, n_jobs=1, cv=5):
        gridcv = GridSearchCV(self.model, param_grid=params, n_jobs=n_jobs, cv=cv, scoring=self.scoreFun)
        gridcv.fit(X_train, y_train)
        self.best_param = gridcv.best_params_
        self.best_estimator = gridcv.best_params_

        return gridcv.best_estimator_

    def store(self):
        '''
        Corce data into whatever file format
        '''
        pass
        return


if __name__ == "__main__":
    ## training set
    X_train = [[0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
    y_train = [-1, -1, 1, -1, 1]

    ## testing set
    X_test = [[1, 3, 4], [1, 0, 2], [3, 1, 4], [5, 1, 1], [2, 4, 3]]
    y_test = [-1, 1, 1, 1, 1]

    #### OCSVM validateRandom example ###

    ocsvm1 = MLModel(OneClassSVM())
    params = {'kernel': ['rbf', 'linear', 'poly'], 'nu': [0.3, 0.5]}
    best_estimator1 = ocsvm1.validateRandom(X_train, y_train, params, n_jobs=2, n_iter=10, cv=3)
    print(best_estimator1)

    #### OCSVM validateGrid example ###
    ocsvm2 = MLModel(OneClassSVM())
    best_estimator2 = ocsvm2.validateGrid(X_train, y_train, params, n_jobs=2, cv=3)
    print(best_estimator2)

    #### OCSVM with different scoring function example ###
    ocsvm3 = MLModel(OneClassSVM())
    ocsvm3.addScoreFun(metrics.make_scorer(metrics.mean_absolute_error))
    best_estimator3 = ocsvm3.validateGrid(X_train, y_train, params, n_jobs=2, cv=3)
    print(best_estimator3)
