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

    def CV_kfold_ttsplit(self, X, y, base_estimator, n_folds, alpha, random_seed=154):

        kf = KFold(n_splits=n_folds, random_state=random_seed, shuffle=False)
        error_lst_test = []
        error_lst_train = []
        #split into test and train
        for train_index, test_index in kf.split(X):
            X_train_k, X_test_k = X[train_index], X[test_index]
            y_train_k, y_test_k = np.log(y[train_index]), np.log(y[test_index])

            # Standardize data,
            X_train_k, y_train_k = scaler.fit(X_train_k, y_train_k).transform(X_train_k, y_train_k)
            X_test_k, y_test_k = scaler.fit(X_test_k, y_test_k).transform(X_test_k, y_test_k)

            model = base_estimator(alpha)
            model.fit(X_train_k, y_train_k)

            test_predicted = model.predict(X_test_k)
            error_lst_test.append(rmsle(y_test_k, test_predicted))
            train_predicted = model.predict(X_train_k)
            error_lst_train.append(rmsle(y_train_k, train_predicted))

        cv_test_mean = np.mean(error_lst_test)
        cv_train_mean = np.mean(error_lst_train)

        return [cv_train_mean,cv_test_mean, alpha]

    
    def ridge(self, alpha_lass, k):
        X = self.X.values
        y = self.y.values

        model_R = Ridge

        cols = ['CVtrain_mean_RMSLE','CVtest_mean_RMSLE', 'lambda']
        df = pd.DataFrame(columns = cols)

        for a in alpha_lass:
            df.loc[len(df)] = self.CV_kfold_ttsplit(X, y, model_R, k, a)
        lasso_models = []
        
        for a in alpha_lass:
            scaler.fit(X, y)
            X_train_std, y_train_std = scaler.transform(X, y)
            lasso = model_l(alpha=a)
            lasso.fit(X_train_std, y_train_std)
            lasso_models.append(lasso)

        fig, ax = plt.subplots(figsize=(14, 4))
        for column in X.columns:
            path = paths.loc[:, column]
            ax.plot(np.log10(alpha_lass), path, label=column)
        ax.axvline(np.log10(1e-3), color='grey')
        ax.legend(loc='lower right')
        ax.set_title("LASSO Regression, Standardized Coefficient Paths")
        ax.set_xlabel(r"$\log(\alpha)$")
        ax.set_ylabel("Standardized Coefficient")
        return df

    def lasso(self, alpha_lass, k):
        X = self.X.values
        y = self.y.values

        model_R = Lasso

        cols = ['CVtrain_mean_RMSLE','CVtest_mean_RMSLE', 'lambda']
        df = pd.DataFrame(columns = cols)

        for a in alpha_lass:
            df.loc[len(df)] = self.CV_kfold_ttsplit(X, y, model_R, k, a)
        lasso_models = []
        
        for a in alpha_lass:
            scaler.fit(X, y)
            X_train_std, y_train_std = scaler.transform(X, y)
            lasso = model_l(alpha=a)
            lasso.fit(X_train_std, y_train_std)
            lasso_models.append(lasso)

        fig, ax = plt.subplots(figsize=(14, 4))
        for column in X.columns:
            path = paths.loc[:, column]
            ax.plot(np.log10(alpha_lass), path, label=column)
        ax.axvline(np.log10(1e-3), color='grey')
        ax.legend(loc='lower right')
        ax.set_title("LASSO Regression, Standardized Coefficient Paths")
        ax.set_xlabel(r"$\log(\alpha)$")
        ax.set_ylabel("Standardized Coefficient")
        return df