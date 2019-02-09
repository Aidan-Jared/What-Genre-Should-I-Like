import numpy as np
import pandas as pd
import scipy.stats as stats
import BetaDist as Beta
import matplotlib.pyplot as plt
from itertools import combinations


def langSelect(df, lang):
    '''
    input: dataframe, string
    output: dataframe
    takes in a dataframe and removes all langs except for the one selected
    '''
    mask = df['language_code'] == lang
    return df[mask]

def dataClean(df):
    '''
    input: dataframe
    output: dataframe
    removes tags from dataframe 
    '''
    for i in ['read', 'book', 'favorite', 'own', 'audio', 'wish-list', '--', 'library','buy','kindle','finish','have','audi','borrowed','favourites','default']:
        mask1 = df['tag_name'].str.contains(i)
        df= df[~mask1]
    return df

def isolate_tag(df, tag, exculude, books_read):
    '''
    input: dataframe, string, list, int
    output: dataframe
    selects only the books from the tag and users if they read more than the min books_read
    '''
    #need to change this area
    mask = df['tag_name'].str.contains(tag)
    df_new = df[mask]
    for i in exculude:
        mask = df_new['tag_name'].str.contains(i)
        df_new = df_new[~mask]
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
    df_books = langSelect(df_books, 'eng')

    #merging dataframes
    df_tags_books = Beta.Beta(df_book_tags, df_tags).dfMerge('tag_id', 'tag_id', compile_A=False)
    df_rating_tags = Beta.Beta(df_tags_books, df_ratings).dfMerge('goodreads_book_id', 'book_id', compile_A=False)
    
    #cleaning dataframe
    df_rating_tags = dataClean(df_rating_tags)
    
    #Creating the first 2 data frames to compare
    df_fantasy = isolate_tag(df_rating_tags, 'fantasy', ['to-read','literature','sci-fi'],4)
    df_lit = isolate_tag(df_rating_tags, 'literature', ['to-read','fantasy'], 4)

    #setting up Beta distributions and plots
    fantasy_lit = Beta.Beta(df_fantasy, df_lit, 'user_rating').compile_analysis('Fantasy', 'Literature', Plot=True)
    plt.show()
    print(fantasy_lit)

    #history vs literature
    df_hist = isolate_tag(df_rating_tags, 'history', ['to-read','fantasy', 'fiction','sci-fi'], 4)
    lit_hist = Beta.Beta(df_lit, df_hist, 'user_rating').compile_analysis('Literature','History', Plot=True)
    plt.show()
    print(lit_hist)

    #science vs relgion
    df_sci = isolate_tag(df_rating_tags, 'science', ['to-read','fantasy', 'fiction','sci-fi'], 4)
    df_relig = isolate_tag(df_rating_tags, 'religion', ['to-read','fantasy', 'fiction','sci-fi'], 4)
    sci_relig = Beta.Beta(df_sci, df_relig, 'user_rating').compile_analysis('Science','Religion', Plot=True)
    plt.show()
    print(sci_relig)

    #creat combinations of top 10 tags
    tag_comb = combinations(tag_10, 2)