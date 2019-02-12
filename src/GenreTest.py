import numpy as np
import pandas as pd
import scipy.stats as stats
import BetaDist as Beta
import linearReg as LR
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
    removes common tags from dataframe 
    '''
    for i in ['read', 'book', 'favorite', 'own', 'audio', 'wish-list', '--', 'library','buy','kindle','finish','have','audi','borrowed','favourites','default']:
        mask1 = df['tag_name'].str.contains(i)
        df= df[~mask1]
    return df

def isolate_tag(df, tag, books_read):
    '''
    input: dataframe, string, list, int
    output: dataframe
    selects only the books from the tag and users if they read more than the min books_read
    '''
    #need to change this area
    mask = df['tag_name'] == tag
    df_new = df[mask]
    df_new = Beta.Beta(df_new, df_ratings).dfMerge('goodreads_book_id', 'book_id', compile_A=False)
    test = df_new[['user_id','rating']].groupby(['user_id']).count()
    df_new = df_new.merge(test, left_on='user_id', right_index=True)
    df_new = df_new.rename(index=str, columns = {"rating_x": 'user_rating' , 'rating_y':'books_read'})
    mask2 = df_new['books_read'] >= books_read
    df_new = df_new[mask2]
    df_new = df_new[['user_id', 'user_rating']]
    return df_new.groupby(['user_id']).mean().reset_index()

def massCompare(tag_comb, tag_10):
        res = []
        for i in tag_comb:
            df1 = isolate_tag(df_tags_books, i[0], 4)
            df2 = isolate_tag(df_tags_books, i[1], 4)
            tags = Beta.Beta(df1, df2, 'user_rating').compile_analysis(i[0],i[1], Plot=False)
            res.append(tags)
        return res

def massTagComb(df, num_tag):
        df_user = df[["tag_name"]].sort_values(by=['tag_name'])
        tag_10 = df_user['tag_name'].value_counts().reset_index()
        tag_10 = tag_10.iloc[:num_tag,0]
        tag_comb = combinations(tag_10, 2)
        return tag_comb, tag_10.tolist()

def dataCleanAdvanced(df):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(df)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, df)
    doc_new = 'fantasy'
    X_new_count = count_vect.transform(doc_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_count)
    pred = clf.predict(X_new_tfidf)
    print(df[pred])
    return clf


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
    df_tags_books = dataClean(df_tags_books)
    df_tags_books = df_tags_books.sort_values('count', ascending=False).drop_duplicates(['goodreads_book_id'])
    print(df_tags_books['tag_name'].value_counts())

    #Creating the first 2 data frames to compare
    df_fantasy = isolate_tag(df_tags_books, 'fantasy', 4)
    df_fic = isolate_tag(df_tags_books, 'fiction', 4)

    #setting up Beta distributions and plots
    fantasy_fic = Beta.Beta(df_fantasy, df_fic, 'user_rating').compile_analysis('Fantasy', 'Fiction', Plot=True)
    plt.show()
    print(fantasy_fic)

    #history vs literature
    df_nfic = isolate_tag(df_tags_books, 'non-fiction', 4)
    fic_nfic = Beta.Beta(df_fic, df_nfic, 'user_rating').compile_analysis('Fiction','Non-Fiction', Plot=True)
    plt.show()
    print(fic_nfic), 

    #science vs relgion
    df_sci = isolate_tag(df_tags_books, 'science-fiction', 4)
    df_class = isolate_tag(df_tags_books, 'classics', 4)
    sci_class = Beta.Beta(df_sci, df_class, 'user_rating').compile_analysis('Science-Fiction','Classics', Plot=True)
    plt.show()
    print(sci_class)

    #create and test combinations of top 10 tags
    tag_comb, tag_10 = massTagComb(df_tags_books,10)
    print(massCompare(tag_comb,tag_10))