# Galvanize Capstone 1

## Table of Contents

- [Project Explanation and Goals](#project-explanation-and-goals)    
- [Baysian A/B testing](#baysian-ab-testing)        
    - [Fantasy and Literature](#fantasy-and-literature)        
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

After cleaning and orginizing my dataset, I decided that the beset way to compare genres would be through Baysian A/B testing because it can show the mean rating that users give to other genres and can give me a numerical values for how much better does one genre fan like another genre.

### Fantasy and Literature
The first two genres I compared were fantasy and literature.
![alt text](images/fig_Fantasy.png)

Fantasy tended to only raited literture higher than literature raited fantasy 48% of the time which combined with the graph shows to me that if you are a fan of fantasy or literature you are very likely to enjoy the other genre.

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