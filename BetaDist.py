import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    def avg_metric_by_product(self, df, tag):
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

    def plot_distribution(self, df1_avg, df2_avg, tag):
        '''
        INPUT: dataframe, dataframe, string
        OUTPUT: graph
        Graphs the probability distribution of a column from each of two dataframes along with
        their fitted beta distribution
        '''
        df1_a, df1_b = self.get_beta_params(df1_avg, self.tag)
        df2_a, df2_b = self.get_beta_params(df2_avg, self.tag)

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        x = np.linspace(0, 1, 100)

        ax.plot(x, stats.beta.pdf(x, df1_a, df1_b),'r-', lw=5, alpha=0.6, label='beta pdf')
        ax2.plot(x, stats.beta.pdf(x, df2_a, df2_b),'r-', lw=5, alpha=0.6, label='beta pdf')

        ax.hist(df1_avg[tag], bins = 15, rwidth = 0.75, density=True)
        ax2.hist(df2_avg[tag], bins = 15, rwidth = 0.75, density=True)

        ax.set_ylabel('P({self.tag} = x)')
        ax.set_xlabel('x')
        ax2.set_ylabel('P({self.tag} = x)')
        ax2.set_xlabel('x')
        ax.set_title('{self.df1.iloc[0,0]} Beta Function')
        ax2.set_title('{self.df2.iloc[0,0]} Beta Function')

        # fig2 = plt.figure(figsize=(16, 16))
        # bx = fig2.add_subplot(111)
        # bx.plot(x, stats.beta.pdf(x, df1_a, df1_b),'r-', lw=5, alpha=0.6, label='beta pdf')
        # bx.plot(x, stats.beta.pdf(x, df2_a, df2_b),'b-', lw=5, alpha=0.6, label='beta pdf')
        # bx.set_ylabel()
        # bx.set_xlabel()
        # bx.set_title('Vine reviews - ' + tag.replace('_', ' '))

    def compile_analysis(self):
        '''
        INPUT: dataframe, dataframe, string
        OUTPUT: float
        Calls above functions and returns results of hypothesis test.
        '''
        df1_avgs = self.avg_metric_by_product(self.df1, self.tag)
        df2_avgs = self.avg_metric_by_product(self.df2, self.tag)

        self.plot_distribution(df1_avgs, df2_avgs, self.tag)

        df1_a, df1_b = self.get_beta_params(df1_avgs, self.tag)
        df2_a, df2_b = self.get_beta_params(df2_avgs, self.tag)

        probability = self.beta_test(df1_a, df2_a, df1_b, df2_b)*100
        return probability