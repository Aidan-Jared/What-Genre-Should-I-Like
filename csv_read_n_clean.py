import numpy as np
import pandas as pd
import scipy.stats as stats
import BetaDist as Beta
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

def isolate_tag(df, tag, books_read):
    '''
    input: dataframe, string, int
    output: dataframe
    selects only the books from the tag and users if they read more than the min books_read
    '''
    mask = df['tag_name'] == tag
    df_new = df[mask]
    test = df_new[['user_id','rating']].groupby(['user_id']).count()
    df_new = df_new.merge(test, left_on='user_id', right_index=True)
    df_new = df_new.rename(index=str, columns = {"rating_x": 'user_rating' , 'rating_y':'books_read'})
    mask2 = df_new['books_read'] >= books_read
    df_new = df_new[mask2]
    return df_new


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
    
    #Creating the first 2 data frames to compare
    df_fantasy = isolate_tag(df_rating_tags, 'fantasy', 4)
    df_scifi = isolate_tag(df_rating_tags, 'sci-fi', 4)

    #setting up Beta distributions and plots
    x = Beta.Beta(df_fantasy, df_scifi, 'user_rating').compile_analysis('Fantasy', 'Sci-Fi')
    plt.show()
    print(x)

    df_historyM = isolate_tag(df_rating_tags, "history-mystery", 4)
    y = Beta.Beta(df_historyM, df_fantasy, 'user_rating').compile_analysis('HistoryM', 'Fantasy')
    plt.show()