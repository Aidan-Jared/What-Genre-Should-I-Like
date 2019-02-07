import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional

class Beta(object):
    '''
    object for doing quick beta functions
    '''
    def __init__(self, df1, df2, tag):
        '''
        df1: dataframe of one tag type
        df2: dataframe of another tag type
        '''
        self.df1 = df1
        self.df2 = df2
        self.tag = tag
    
    def avg_rating_by_user(self, df, tag):
        '''
        INPUT: dataframe, string
        OUTPUT: dataframe
        Groups dataframe by user id and averages each product over the given
        tag
        '''
        return df.groupby('user_id').mean()[self.tag].reset_index()

    def get_beta_params(self, df, tag):
        '''
        INPUT: dataframe, string
        OUTPUT: float, float
        Calculates alpha and beta for beta distribution using the mean and standard
        deviation
        '''
        df[self.tag] = df[self.tag]/df[self.tag].max()
        mu = np.mean(df.loc[:,self.tag])
        sd = np.std(df.loc[:,self.tag])
        alpha = (((1 - mu )/sd**2) - (1/mu)) * mu**2
        beta = alpha * ((1 / mu) - 1)
        return(alpha, beta)

    def beta_test(self, a1, a2, b1, b2):
        '''
        INPUT: float, float, float, float
        OUTPUT: float
        Runs a Baysian hypothesis test on two distributions with parameters a and b.
        Returns a probability
        '''
        count = 0
        for _ in range(10000):
            if np.random.beta(a1, b1) > np.random.beta(a2, b2):
                count +=1
        return count/10000

    def plot_distribution(self, df1_avg, df2_avg, tag, df1_name, df2_name):
        '''
        INPUT: dataframe, dataframe, string
        OUTPUT: graph
        Graphs the probability distribution of a column from each of two dataframes along with
        their fitted beta distribution
        '''
        df1_a, df1_b = self.get_beta_params(df1_avg, self.tag)
        df2_a, df2_b = self.get_beta_params(df2_avg, self.tag)

        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(2,2)
        ax = fig.add_subplot(gs[:,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[1,1])

        x = np.linspace(0, 1, 100)

        ax.plot(x, stats.beta.pdf(x, df1_a, df1_b),'r-', lw=5, alpha=0.7, label=df1_name)
        ax.plot(x, stats.beta.pdf(x, df2_a, df2_b),'b-', lw=5, alpha=0.7, label=df2_name)
        ax.legend()

        ax1.hist(df1_avg[tag], color='y', bins = 15, rwidth = 0.75, density=True, label= df1_name + " User Raitings")
        ax2.hist(df2_avg[tag], color = 'g',bins = 15, rwidth = 0.75, density=True, label= df2_name + " User Raitings")

        ax.set_ylabel('P(user_rating) = x)')
        ax.set_xlabel('x')
        ax.set_title('{0} avg rating vs {1} avg rating Beta Function'.format(df1_name, df2_name))
        ax1.legend()
        ax2.legend()


    def dfMerge(self, df1, df2, leftjoin = 'user_id', rightjoin = 'user_id', compile_A = True):
        '''
        Input: dataframe, dataframe, string, string
        Output: dataframe
        merges the two dataframes on userid
        '''
        if compile_A == True:
            df1_m_df2 = self.df1[['user_id','tag_name']].merge(self.df2, left_on=leftjoin, right_on=rightjoin)
            df2_m_df1 = self.df2[['user_id','tag_name']].merge(self.df1, left_on='user_id', right_on='user_id')
            return df1_m_df2, df2_m_df1
        elif compile_A == False:
            return self.df1.merge(self.df2, left_on=leftjoin, right_on=rightjoin)
    
    def compile_analysis(self, df1_name, df2_name):
        '''
        INPUT: dataframe, dataframe, string
        OUTPUT: float
        Calls above functions and returns results of hypothesis test.
        '''
        df1_m_df2, df2_m_df1 = self.dfMerge(self.df1, self.df2)

        df1_avgs = self.avg_rating_by_user(df1_m_df2, self.tag)
        df2_avgs = self.avg_rating_by_user(df2_m_df1, self.tag)

        self.plot_distribution(df1_avgs, df2_avgs, self.tag, df1_name, df2_name)

        df1_a, df1_b = self.get_beta_params(df1_avgs, self.tag)
        df2_a, df2_b = self.get_beta_params(df2_avgs, self.tag)

        return self.beta_test(df1_a, df2_a, df1_b, df2_b)