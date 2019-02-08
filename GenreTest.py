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

def isolate_tag(df, tag, books_read):
    '''
    input: dataframe, string, int
    output: dataframe
    selects only the books from the tag and users if they read more than the min books_read
    '''
    #need to change this area
    mask = df['tag_name'].str.contains(tag)
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
    df_tags_books = Beta.Beta(df_book_tags, df_tags).dfMerge('tag_id', 'tag_id', compile_A=False)
    df_rating_tags = Beta.Beta(df_tags_books, df_ratings).dfMerge('goodreads_book_id', 'book_id', compile_A=False)
    
    #Creating the first 2 data frames to compare
    df_fantasy = isolate_tag(df_rating_tags, 'fantasy', 4)
    df_scifi = isolate_tag(df_rating_tags, 'sci-fi', 4)

    #setting up Beta distributions and plots
    fantasy_scifi = Beta.Beta(df_fantasy, df_scifi, 'user_rating').compile_analysis('Fantasy', 'Sci-Fi')
    plt.show()
    print(fantasy_scifi)

    # y = Beta.Beta(df_historyH, df_fantasy, 'user_rating').compile_analysis('HistoryH', 'fantasy')
    # print(y)
    # plt.show()