from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import sklearn
from sklearn.linear_model import Lasso, Ridge
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut


class linearReg(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.scaler = ut.XyScaler()
    
    def rmsle(self, actual, predictions):
        log_diff = np.log(np.abs(predictions)+1) - np.log(np.abs(actual)+1)
        return np.sqrt(np.mean(log_diff**2))

    def CV_kfold_ttsplit(self, base_estimator, alpha, random_seed=154):
        kf = KFold(n_splits=5, random_state=random_seed, shuffle=False)
        error_lst_test = []
        error_lst_train = []
        #split into test and train
        for train_index, test_index in kf.split(self.X):
            X_train_k, X_test_k = self.X[train_index], self.X[test_index]
            y_train_k, y_test_k = self.y[train_index], self.y[test_index]

            model = base_estimator(alpha)
            model.fit(X_train_k, y_train_k)

            test_predicted = model.predict(X_test_k)
            error_lst_test.append(self.rmsle(y_test_k, test_predicted))
            train_predicted = model.predict(X_train_k)
            error_lst_train.append(self.rmsle(y_train_k, train_predicted))

        cv_test_mean = np.mean(error_lst_test)
        cv_train_mean = np.mean(error_lst_train)

        return [cv_train_mean,cv_test_mean, alpha]
    
    def ridge(self, alpha, k):

        model_R = Ridge

        cols = ['CVtrain_mean_RMSLE','CVtest_mean_RMSLE', 'lambda']
        df = pd.DataFrame(columns = cols)

        for a in alpha:
            df.loc[len(df)] = self.CV_kfold_ttsplit(model_R, a)
        return df

    def lasso(self, alpha, k):

        model_L = Lasso

        cols = ['CVtrain_mean_RMSLE','CVtest_mean_RMSLE', 'lambda']
        df = pd.DataFrame(columns = cols)

        for a in alpha:
            df.loc[len(df)] = self.CV_kfold_ttsplit(model_L, a)
        return df