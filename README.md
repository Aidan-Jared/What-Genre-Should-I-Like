# Galvanize Capstone 1

## Table of Contents

- [Project Explanation and Goals](#project-explanation-and-goals)    
- [Baysian A/B testing](#baysian-ab-testing)        
    - [Fantasy and Fiction](#fantasy-and-fiction)        
    - [Literature and History](#literature-and-history)        
    - [Science and Religion](#science-and-religion)        
    - [Top 10 tags](#top-10-tags)    
- [Linear Regresion](#linear-regresion)        
    - [Ridge](#ridge)        
    - [Lasso](#lasso)

## Project Explanation and Goals

For this poject I decided to look at book genres and the "bleeding" between genres (the enjoyment of one genre given the enjoyment of another genre). The Goodreads 10k book dataset from kaggle seemed perfect because it countained 10k books, user reviews and user given tags. The data sets was made of four csv files, books, book_tags, ratings, and tags. My end plan is to be able to predict the raitings that someone might give any genre given their enjoyment of one genre. 

My first step was to fiugre out how to put all of the data together so I could search through the data frame. The main problem though is the identifyer I am using are the user generated tags. (tag csv file) When just loading up theses tags, the first thing I noticed was that the majority of the tags are worthless such as to-read, favorite, --31-, and some tags writen in japanese. To deal with this I just did a quick visual search and made a function that removed the most common words I did not care about and would have messed with my analysis.
```python
def dataClean(df):
    for i in ['read', 'book', 'favorite', 'own', 'audio', 'wish-list', '--', 
    'library','buy','kindle','finish','have','audi','borrowed',
    'favourites','default']:
        mask1 = df['tag_name'].str.contains(i)
        df= df[~mask1]
    return df
```

After this I needed to find out how group the tags, books, and users so I could select specific genres and mean users reviews. My solution was to join the book_tags, and tag dataframes so that I would have tag names tied with book_ids. I then put this dataframe through my cleaning function and removed all repeate values so that I would only have one instance of every book and the tag that most people games this book. Then end result looked soemthing like this:

| Genre        | Books in Genre           |
| ------------- |:-------------:|
|fiction            |      1639|
|fantasy            |      1104|
|young-adult        |       695|
|mystery            |       640|
|non-fiction        |       597|
|romance            |       464|
|historical-fiction |       428|
|classics           |       419|
|childrens          |       294|
|science-fiction    |       269|
|horror             |       201|
|graphic-novels     |       180|

After finishing all this up it was off to the races.

## Baysian A/B testing

After cleaning and orginizing my dataset, I decided that the beset way to compare genres would be through Baysian A/B testing because it can show the mean rating that users give to other genres and can give me a numerical values for how much better does one genre fan like another genre. For these experments my Null is that individuals give the same rating to any genre and my Alt is that they give differnt values.

### Fantasy and Fiction
The first two genres I compared were fantasy and fiction being the two most common book types.

```python
df_fantasy = isolate_tag(df_tags_books, 'fantasy', min_books_read)
df_fic = isolate_tag(df_tags_books, 'fiction', min_books_read)
fantasy_fic = Beta.Beta(df_fantasy, df_fic, 'user_rating').compile_analysis('Fantasy', 'Fiction', Plot=True)
plt.show()
print('A/B test, Difference in Mean Raiting', fantasy_fic)
'A/B test, Difference in Mean Raiting' [0.5153, -0.012692577030812365]
```

![alt text](images/fig_Fantasy.png)

From running the A/B test and the Beta function produced I found that acording to the data there is almost no difference in the ratings that fans of fantasy give to fiction and vice-versa. I ended up getting that model B was better than model A about 50% of the time which once again shows no difference between the two models which makes it so I can't reject the null. This does make some sense because these two genres tend to have a lot of overlap in intrests and some individuals might consider a fantasy novel to be fiction which would definetly skew the data one way or another.

To double check the distribution of user raitings I took the difference between the average fantasy rating and the average fiction and found that on average, there is only a -.01 point difference between how individuals rate fantasy and fiction. To me this shows that if you like fiction or fantasy, you will mostlikly like the other genre equally as much.

### Literature and History

I then compared were literature and history.
![alt text](images/fig_Fiction.png)
This seems very simular to 51%


### Science and Religion

I then compared were science and religion.
![alt text](images/fig_Science-Fiction.png)
These results suprised me 49%

### Top 10 tags

After having this relisations, I decided to look at how all the top 10 used tags compare against eachother

## Linear Regresion

### Ridge

### Lasso