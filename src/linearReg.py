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

    def CV_kfold_ttsplit(self, base_estimator, alpha, n_folds = 5, random_seed=154):
        kf = KFold(n_splits=n_folds, random_state=random_seed, shuffle=False)
        error_lst_test = []
        error_lst_train = []
        #split into test and train
        for train_index, test_index in kf.split(self.X):
            X_train_k, X_test_k = self.X[train_index], self.X[test_index]
            y_train_k, y_test_k = self.y[train_index], self.y[test_index]

            # Standardize data,
            X_train_k, y_train_k = self.scaler.fit(X_train_k, y_train_k).transform(X_train_k, y_train_k)
            X_test_k, y_test_k = self.scaler.fit(X_test_k, y_test_k).transform(X_test_k, y_test_k)

            model = base_estimator(alpha)
            model.fit(X_train_k, y_train_k)

            test_predicted = model.predict(X_test_k)
            error_lst_test.append(rmsle(y_test_k, test_predicted))
            train_predicted = model.predict(X_train_k)
            error_lst_train.append(rmsle(y_train_k, train_predicted))

        cv_test_mean = np.mean(error_lst_test)
        cv_train_mean = np.mean(error_lst_train)

        return [cv_train_mean,cv_test_mean, alpha]
    
    def ridge(self, alpha, k):

        model_R = Ridge

        cols = ['CVtrain_mean_RMSLE','CVtest_mean_RMSLE', 'lambda']
        df = pd.DataFrame(columns = cols)

        for a in alpha:
            df.loc[len(df)] = self.CV_kfold_ttsplit(model_R, k, a)
        ridge_models = []
        
        for a in alpha:
            self.scaler.fit(self.X, self.y)
            X_train_std, y_train_std = self.scaler.transform(X, y)
            model_R = model_R(alpha=a)
            model_R.fit(X_train_std, y_train_std)
            ridge_models.append(model_R)
        return df

    def lasso(self, alpha, k):

        model_L = Lasso

        cols = ['CVtrain_mean_RMSLE','CVtest_mean_RMSLE', 'lambda']
        df = pd.DataFrame(columns = cols)

        for a in alpha:
            df.loc[len(df)] = self.CV_kfold_ttsplit(model_L, a)
        lasso_models = []
        
        for a in alpha:
            self.scaler.fit(self.X, self.y)
            X_train_std, y_train_std = self.scaler.transform(X, y)
            lasso = Lasso(alpha=a)
            lasso.fit(X_train_std, y_train_std)
            lasso_models.append(lasso)
        return df