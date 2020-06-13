import numpy as np
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import pandas as pd


class MLModel:

    def __init__(self, model):
        self.model = model
        self.scoreFun = metrics.make_scorer(metrics.accuracy_score)
        return

    def validateRandom(self, X_train, y_train, params):
        '''
        Validate model with training data using RandomizedSearchCV, return best_estimator_
        '''

        randomcv = RandomizedSearchCV(self.model, param_distributions= params, n_jobs= -1, n_iter = 10, cv = 5, scoring=self.scoreFun)
        randomcv.fit(X_train, y_train)
        self.best_param = randomcv.best_params_
        self.best_estimator = randomcv.best_estimator_

        return randomcv.best_estimator_

    def validateGrid(self, X_train, y_train, params):
        gridcv = GridSearchCV(self,model, param_grid= params, n_jobs = -1, n_iter = 10, cv = 5, scoring = self.scoreFun)
        gridcv.fit(X_train, y_train)
        self.best_param = gridcv.best_params_
        self.best_estimator = gridcv.best_params_

        return gridcv.best_estimator_
        

    def addScoreFun(self, scoreFun):
        '''
        Optionally create score functions
        '''
        self.scoreFun = scoreFun
        return


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

    ## score function
    score = metrics.make_scorer(metrics.accuracy_score)


    param_dis = {"n_estimators": np.arange(10, 110, step = 5)}
    c = MLModel(IsolationForest())
    c.addScoreFun(scoreFun= score) 
    best_estimator = c.validateRandom(X_train, y_train, param_dis)
    
    print(score(best_estimator, X_test, y_test))


