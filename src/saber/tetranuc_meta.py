import numpy as np
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics
import pandas as pd



## cross validation: Split training data into several sets, and use different parameters to fit the set into models to determine the best
##                   parameters. Then predict the test data

## data structures: X_train(cv), y_train(cv), X_test, y_test

class MLModel:

    def __init__(self, model, scoreFun = metrics.make_scorer(metrics.accuracy_score)):
        '''Constructor function
        :param model: ML model to validate
        :param scoreFun : score function used to validate models, defaults to make_scorer(metrics.accuracy_score)
        '''
        self.model = model
        self.scoreFun = scoreFun
        return

    def validateRandom(self, X_train, params, y_train = None, n_jobs = 1, n_iter = 10, cv = 5):
        '''Validate model with training data using RandomizedSearchCV
        :param X_train: X_train
        :param y_train: y_train
        :param params: parameter in a dictionary
        :param n_jobs: number of jobs to run in parallel, default to 1
        :param n_iter: Number of parameter settings that are sampled, default to 10
        :param cv: Determines the cross-validation splitting strategy, default to 5
        '''
        randomcv = RandomizedSearchCV(self.model, param_distributions= params, n_jobs = n_jobs, n_iter = n_iter, cv = cv, scoring= self.scoreFun)
        randomcv.fit(X_train, y_train)
        self.best_param = randomcv.best_params_
        self.best_estimator = randomcv.best_estimator_

        return randomcv.best_estimator_


    def validateGrid(self, X_train, y_train, params, n_jobs = 1, cv = 5):
        gridcv = GridSearchCV(self.model, param_grid= params, n_jobs = n_jobs, cv = cv, scoring = self.scoreFun)
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

    def precision_metric(self):


        return 


if __name__ == "__main__":
    
    ## training set
    # X_train = [[0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
    # y_train = [-1, -1, 1, -1, 1]

    ## testing set
    # X_test = [[1, 3, 4], [1, 0, 2], [3, 1, 4], [5, 1, 1], [2, 4, 3]]
    # y_test = [-1, 1, 1, 1, 1]


    X_train = np.random.rand(100,3)
    y_train = np.random.rand(100,1)

    X_test = np.random.rand(50,3)
    y_test = np.random.rand(50,1)


    ### GMM validateRandom example ###
    gmm = MLModel(GMM(), scoreFun= GMM().aic)
    gmm_param = {"n_components":range(1, 10, 2)}
    best = gmm.validateRandom(X_train= X_train, params = gmm_param)
    print(best)



    # #### OCSVM validateRandom example ###

    # ocsvm1 = MLModel(OneClassSVM())
    # params = {'kernel': ['rbf', 'linear', 'poly'], 'nu': [0.3, 0.5]}
    # best_estimator1 = ocsvm1.validateRandom(X_train, y_train, params, n_jobs = 2, n_iter = 10, cv = 5)
    # print(best_estimator1)



    # #### OCSVM validateGrid example ###
    # ocsvm2 = MLModel(OneClassSVM())
    # best_estimator2 = ocsvm2.validateGrid(X_train, y_train, params, n_jobs = 2, cv = 3)
    # print(best_estimator2)


    # #### OCSVM with different scoring function example ###
    # ocsvm3 = MLModel(OneClassSVM())
    # ocsvm3.addScoreFun(metrics.make_scorer(metrics.mean_absolute_error))
    # best_estimator3 = ocsvm3.validateGrid(X_train, y_train, params, n_jobs = 2, cv = 3)
    # print(best_estimator3)



    


