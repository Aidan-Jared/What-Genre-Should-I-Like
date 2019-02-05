import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def lang_select(df, lang):
    '''
    input: dataframe, string
    output: dataframe
    takes in a dataframe and removes all langs except for the one selected
    '''
    mask = df['language_code'] == lang
    return df[mask]

def merge_df(leftDF, rightDF, leftJoin, rightJoin):
    '''
    input: dataframe, dataframe, string, string
    output: dataframe
    merges the leftDF and rightDf on the leftJoin and rightJon
    '''
    return leftDF.merge(rightDF, left_on=leftJoin, right_on=rightJoin)


if __name__ == '__main__':
    #importing in the csv
    df_books = pd.read_csv('data/books.csv')
    df_book_tags = pd.read_csv('data/book_tags.csv')
    df_ratings = pd.read_csv('data/ratings.csv')
    df_tags = pd.read_csv('data/tags.csv')
    
    #selecting only the books with the eng lang code
    df_books = lang_select(df_books, 'eng')

    #merging dataframes
    df_tags_books = merge_df(df_book_tags, df_tags, 'tag_id', 'tag_id')
    df_rating_tags = merge_df(df_tags_books, df_ratings, 'goodreads_book_id', 'book_id')
    