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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import utils as ut
plt.style.use('ggplot')

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

def ratingMatrix(df):
    '''
    input: dataframe, string, list, int
    output: dataframe
    makes a matrix of users, books, and their raitings
    '''
    #need to change this area
    df_new = df
    df_new = Beta.Beta(df_new, df_ratings).dfMerge('goodreads_book_id', 'book_id', compile_A=False)
    df_new = df_new[['book_id', 'user_id', 'rating']].reset_index()
    df_new = pd.pivot_table(df_new, values='rating', index=['user_id'], columns=['book_id'], aggfunc=np.sum, fill_value=0)
    return df_new

def massCompare(tag_comb, tag_10):
        columns = ['Combinations', 'Bleed_Over', "Avg_Raiting_diff"]
        df = pd.DataFrame(columns = columns)
        for i in tag_comb:
            df1 = isolate_tag(df_tags_books, i[0], 4)
            df2 = isolate_tag(df_tags_books, i[1], 4)
            tags = Beta.Beta(df1, df2, 'user_rating').compile_analysis(i[0],i[1], Plot=False, dataframe=False)
            df.loc[len(df)] = i, tags[0], tags[1]
        return df

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

def Ridge_model(df, df1,df2, name):
        X = df['user_rating_x'].values.reshape(-1,1)
        y = df['user_rating_y'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=10)
        LR_R = LR.linearReg(X_train, y_train)
        alpha = [.00001,.0001,.001,.01,.1,1,10,100]
        ridge = LR_R.ridge(alpha,5)
        index = ridge['CVtest_mean_RMSLE'].idxmin()
        a = ridge['lambda'][index]
        pred = Ridge(alpha=a).fit(X_train, y_train).predict(X_test)
        print(LR_R.rmsle(y_test, pred))
        plt.scatter(X_test, y_test)
        plt.plot(X_test, pred)
        plt.ylabel(df2)
        plt.xlabel(df1)
        plt.title('Ridge Regression for {}'.format(name))
        plt.savefig('images/{}_ridge_model'.format(name))
        plt.show()

def Lasso_model(df, df1, df2, name):
    X = df['user_rating_x'].values.reshape(-1,1)
    y = df['user_rating_y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=10)
    LR_L = LR.linearReg(X_train, y_train)
    alpha = [.00001,.0001,.001,.01,.1,1,10,100]
    lasso = LR_L.lasso(alpha,5)
    index = lasso['CVtest_mean_RMSLE'].idxmin()
    a = lasso['lambda'][index]
    pred = Lasso(alpha=a).fit(X_train, y_train).predict(X_test)
    print(LR_L.rmsle(y_test, pred))
    plt.plot(X_test, pred)
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_test)
    plt.plot(X_test, pred)
    plt.ylabel(df2)
    plt.xlabel(df1)
    plt.title('Lasso Regression for {}'.format(name))
    plt.savefig('images/{}_lasso_model'.format(name))
    plt.show()

def Linear_Regression(df, df1, df2, name):
    X = df['user_rating_x'].values.reshape(-1,1)
    y = df['user_rating_y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=10)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    print(LR.linearReg(X_test,y_test).rmsle(y_test, pred))
    plt.plot(X_test, pred)
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_test)
    plt.plot(X_test, pred)
    plt.ylabel(df2)
    plt.xlabel(df1)
    plt.title('Linear Regression for {}'.format(name))
    plt.savefig('images/{}_linear_model'.format(name))
    plt.show()

if __name__ == '__main__':
    #settings
    min_books_read = 1
    plot = False
    # plot = input("plot True or False: ")
    # if plot == 'True':
    #     plot = True
    # else:
    #    plot == False
    
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
    bar_df = df_tags_books['tag_name'].value_counts()
    bar_df = bar_df.iloc[:50]
    fig = plt.figure(figsize=(8,8))
    bar_df.plot(kind='bar')
    fig.savefig('images/EDA_Bar.png')
    plt.show()

    #Creating the first 2 data frames to compare
    df_fantasy= isolate_tag(df_tags_books, 'fantasy', min_books_read)
    fig = plt.figure(figsize=(7,7))
    df_fantasy['user_rating'].hist(bins=20)
    plt.title("Fantasy Mean User Ratings")
    plt.xlabel('mean user ratings')
    fig.savefig('images/fantasy_hist.png')
    plt.show()

    df_fic = isolate_tag(df_tags_books, 'fiction', min_books_read)
    fig = plt.figure(figsize=(7,7))
    df_fic['user_rating'].hist(bins=20)
    plt.title("Fiction Mean User Ratings")
    plt.xlabel('mean user ratings')
    fig.savefig('images/fiction_hist.png')
    plt.show()

    #setting up Beta distributions and plots
    fantasy_fic, fantasy_fic_df = Beta.Beta(df_fantasy, df_fic, 'user_rating').compile_analysis('Fantasy', 'Fiction', Plot=plot)
    plt.show()
    print(fantasy_fic)

    #history vs literature
    df_vamp = isolate_tag(df_tags_books, 'vampires', min_books_read)
    fig = plt.figure(figsize=(7,7))
    df_vamp['user_rating'].hist(bins=20)
    plt.title("Vampire Mean User Ratings")
    plt.xlabel('mean user ratings')
    fig.savefig('images/Vampire_hist.png')
    plt.show()
    
    fic_vamp, fic_vamp_df = Beta.Beta(df_fic, df_vamp, 'user_rating').compile_analysis('Fiction','Vampire', Plot=plot)
    plt.show()
    print(fic_vamp), 

    #science vs relgion
    df_sci = isolate_tag(df_tags_books, 'science-fiction', min_books_read)
    fig = plt.figure(figsize=(7,7))
    df_sci['user_rating'].hist(bins=20)
    plt.title("Science-Fiction Mean User Ratings")
    plt.xlabel('mean user ratings')
    fig.savefig('images/sci-fi_hist.png')
    plt.show()
    
    sci_fant, sci_fant_df = Beta.Beta(df_sci, df_fantasy, 'user_rating').compile_analysis('Science-Fiction','Fantasy', Plot=plot)
    plt.show()
    print(sci_fant)

    #create and test combinations of top 10 tags
    # tag_comb, tag_10 = massTagComb(df_tags_books,10)
    # df_10_comb = massCompare(tag_comb,tag_10)
    # print(df_10_comb)

    #linear models
    Ridge_model(fantasy_fic_df,'Fantasy','Fiction','Fantasy_and_Fiction')
    Lasso_model(fantasy_fic_df,'Fantasy','Fiction','Fantasy_and_Fiction')
    Linear_Regression(fantasy_fic_df,'Fantasy','Fiction','Fantasy_and_Fiction')

    #building a pivot table
    # Rating_Matrix = ratingMatrix(df_tags_books)
    # Rating_Matrix = Rating_Matrix.values
    # w, v = np.linalg.eig(Rating_Matrix)