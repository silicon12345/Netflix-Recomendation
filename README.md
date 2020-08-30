# Netflix-Recomendation
project
## **Pre-Processing**

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()


import gc
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
%matplotlib inline

ratings = pd.read_csv('ratings.csv')

movies = pd.read_csv('movies.csv')

total=ratings.merge(movies,on='movieId',how='inner')
total['genres']=total['genres'].apply(lambda x: x.split('|'))

test=pd.DataFrame(total.groupby(['userId']).agg(movie_count=pd.NamedAgg(column='movieId', aggfunc='count')))

test

test = test.sort_values('movie_count',ascending=False)

test.describe()

test = test[test['movie_count'] > 200]

test = test.reset_index()

test

total=total.merge(test,on='userId',how='inner')

total.drop(columns = ['movie_count'], inplace = True)

total

total.describe()['timestamp']['75%']

temp = total[['userId','movieId','timestamp']]

temp1 = temp.copy()

temp1

temp1['class'] = temp1['timestamp'].apply(lambda x: 1 if x < 1455488882.0 else 2)

temp1

del ratings

gc.collect()

temp1 = temp1.groupby(['userId','class'])['movieId'].count()

temp1

temp1 = temp1.reset_index(level=[1])

temp1 = temp1.groupby(level = 0).count()

active_users = temp1[temp1['class'] == 2].index.tolist()

del test,temp1
gc.collect()

gc.collect()

## **User Movie Matrix**

active_users = pd.read_csv('/content/drive/My Drive/DS Project 20M/ActiveUsers.csv')

total = total[total.userId.isin(active_users['userId'].tolist())]

total

movie_ids = pd.read_csv('/content/drive/My Drive/DS Project 20M/ActiveMovies.csv')

movies_temp = movies[movies.movieId.isin(movie_ids['movieId'].tolist())]

movies_temp['genres']=movies_temp['genres'].apply(lambda x: x.split('|'))

movies_temp

import itertools
genres_count = list(itertools.chain(*movies_temp.genres.tolist()))

import collections
genres_dict = collections.Counter(genres_count)

genres_dict

user_movie = pd.merge(active_users, ratings, left_on = 'userId', right_on = 'userId')

movie_matrix = user_movie.pivot_table(index='userId',columns='movieId',values='rating')

movie_matrix = movie_matrix.fillna(0.0)

X = movie_matrix.iloc[:, :].values

## **Hierarchical Clustering**

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X)

y_hc

xyz = pd.concat([active_users, pd.DataFrame(y_hc)], axis = 1)
xyz = xyz.rename(columns = {0: 'Cluster'})
xy = xyz.iloc[:, :].values

import collections
collections.Counter(y_hc)

## **Cluster 1**

temp1 = xyz[xyz['Cluster'] == 0]
temp1

cluster_1 = pd.merge(ratings, temp1, left_on = 'userId', right_on = 'userId')

cluster_1 = pd.merge(cluster_1, movies, on = 'movieId')

cluster_1.drop(columns = ['Cluster', 'timestamp', 'userId'], inplace = True)

cluster_1['genres'] = cluster_1['genres'].apply(lambda x: x.split('|'))

cluster_1

cluster_1.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)

cluster_1.movieId.nunique()

import itertools
genres_1 = list(itertools.chain(*cluster_1.genres.tolist()))

myDictionary1 = collections.Counter(genres_1)

for key in myDictionary1:
  myDictionary1[key] = float(myDictionary1[key])/float(genres_dict[key])

bars1 = list(myDictionary1.keys())

height1 = list(myDictionary1.values())

s1 = pd.Series(data = height1, index = bars1)

s1.plot.bar(figsize = (15, 8))

## **Cluster 2**

temp2 = xyz[xyz['Cluster'] == 1]
temp2

cluster_2 = pd.merge(ratings, temp2, left_on = 'userId', right_on = 'userId')

cluster_2 = pd.merge(cluster_2, movies, on = 'movieId')

cluster_2.drop(columns = ['Cluster', 'timestamp', 'userId'], inplace = True)

cluster_2['genres']=cluster_2['genres'].apply(lambda x: x.split('|'))

cluster_2.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)

import itertools
genres_2 = list(itertools.chain(*cluster_2.genres.tolist()))

myDictionary2 = collections.Counter(genres_2)

for key in myDictionary2:
  myDictionary2[key] = float(myDictionary2[key])/float(genres_dict[key])

bars2 = list(myDictionary2.keys())

height2 = list(myDictionary2.values())

s2 = pd.Series(data = height2, index = bars2)

s2.plot.bar(figsize = (15, 8))

## **Cluster 3**

temp3 = xyz[xyz['Cluster'] == 2]
temp3

cluster_3 = pd.merge(ratings, temp3, left_on = 'userId', right_on = 'userId')

cluster_3 = pd.merge(cluster_3, movies, on = 'movieId')

cluster_3.drop(columns = ['Cluster', 'timestamp', 'userId'], inplace = True)

cluster_3['genres']=cluster_3['genres'].apply(lambda x: x.split('|'))

cluster_3.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)

import itertools
genres_3 = list(itertools.chain(*cluster_3.genres.tolist()))

myDictionary3 = collections.Counter(genres_3)

for key in myDictionary3:
  myDictionary3[key] = float(myDictionary3[key])/float(genres_dict[key])

bars3 = list(myDictionary3.keys())

height3 = list(myDictionary3.values())

s3 = pd.Series(data = height3, index = bars3)

s3.plot.bar(figsize = (15, 8))

## **Cluster 4**

temp4 = xyz[xyz['Cluster'] == 3]
temp4

cluster_4 = pd.merge(ratings, temp4, left_on = 'userId', right_on = 'userId')

cluster_4 = pd.merge(cluster_4, movies, on = 'movieId')

cluster_4.drop(columns = ['Cluster', 'timestamp', 'userId'], inplace = True)

cluster_4['genres']=cluster_4['genres'].apply(lambda x: x.split('|'))

cluster_4.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)

import itertools
genres_4 = list(itertools.chain(*cluster_4.genres.tolist()))

myDictionary4 = collections.Counter(genres_4)

for key in myDictionary4:
  myDictionary4[key] = float(myDictionary4[key])/float(genres_dict[key])

bars4 = list(myDictionary4.keys())

height4 = list(myDictionary4.values())

s4 = pd.Series(data = height4, index = bars4)

s4.plot.bar(figsize = (15, 8))

## **Cluster 5**

temp5 = xyz[xyz['Cluster'] == 4]
temp5

cluster_5 = pd.merge(ratings, temp5, left_on = 'userId', right_on = 'userId')

cluster_5 = pd.merge(cluster_5, movies, on = 'movieId')

cluster_5.drop(columns = ['Cluster', 'timestamp', 'userId'], inplace = True)

cluster_5['genres']=cluster_5['genres'].apply(lambda x: x.split('|'))

cluster_5.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)

import itertools
genres_5 = list(itertools.chain(*cluster_5.genres.tolist()))

myDictionary5= collections.Counter(genres_5)

for key in myDictionary5:
  myDictionary5[key] = float(myDictionary5[key])/float(genres_dict[key])

bars5 = list(myDictionary5.keys())

height5 = list(myDictionary5.values())

s5 = pd.Series(data = height5, index = bars5)

s5.plot.bar(figsize = (15, 8))

## **Cluster 6**

temp6 = xyz[xyz['Cluster'] == 5]
temp6

cluster_6 = pd.merge(ratings, temp6, left_on = 'userId', right_on = 'userId')

cluster_6 = pd.merge(cluster_6, movies, on = 'movieId')

cluster_6.drop(columns = ['Cluster', 'timestamp', 'userId'], inplace = True)

cluster_6['genres']=cluster_6['genres'].apply(lambda x: x.split('|'))

cluster_6.drop_duplicates(subset = 'movieId', keep = 'first', inplace = True)

import itertools
genres_6 = list(itertools.chain(*cluster_6.genres.tolist()))

myDictionary6 = collections.Counter(genres_6)

for key in myDictionary6:
  myDictionary6[key] = float(myDictionary6[key])/float(genres_dict[key])

bars6 = list(myDictionary6.keys())

height6 = list(myDictionary6.values())

s6 = pd.Series(data = height6, index = bars6)

s6.plot.bar(figsize = (15, 8))

## **Density-based spatial clustering of applications with noise (DBSCAN)**

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

db = DBSCAN(eps= 100, algorithm='ball_tree', metric='minkowski', p=2, n_jobs = -1)

arr = db.fit_predict(X)

labels = db.labels_

import collections
len(collections.Counter(labels))

collections.Counter(labels)

grid_search_dbscan = {}

def dbscan_grid_search(X_data, eps_space):
  for eps_val in eps_space:
    dbscan_grid = DBSCAN(eps = eps_val, n_jobs = -1).fit(X_data)
    cluster_count = collections.Counter(dbscan_grid.labels_)
    grid_search_dbscan[eps_val] = cluster_count
    print(eps_val)

dbscan_grid_search(X_data = X,eps_space = [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250])

grid_search_dbscan

## **Elbow Method**

from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist

Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
    print(k)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

## ***MOVIE GENRE MATRIX***

genres = []
for i in movies.index:
    genres.extend(movies['genres'][i].split('|'))

genres = list(set(genres))
genres.remove('(no genres listed)')
len(genres)

movies_temp = movies.copy()

movies_temp = movies_temp.set_index('movieId')

movie_genre = pd.DataFrame(columns = genres, index = movie_ids)

movie_genre[genres] = 0.0

for i in movie_ids:
    for j in movies_temp['genres'][i].split('|'):
        try:
            movie_genre[j][i] = 1
        except KeyError:
            pass

movie_genre

## ***Timestamp***

total.describe()['timestamp']['75%']

test_75 = total[total['timestamp'] <= 1454200478.0]

test_25 = total[total['timestamp'] > 1454200478.0]

len(test_75['userId'].unique())

len(test_25['userId'].unique())

common = list(set(test_75['userId'].unique()).intersection(set(test_25['userId'].unique())))

common

test_75

## **Predicting From TimeStamp**



test75_groupby = pd.DataFrame(test_75.groupby(['userId', 'timestamp'])['movieId'].unique())

identify = pd.DataFrame(movie_genre.index.tolist())
identify = identify.rename(columns = {0 : 'movieId'})

identify[identify.index.isin([12404, 18238, 24501, 38769, 38770])].movieId.tolist()

user_movie_pred2 = {}
for i in common:
    user_movie_pred2[i] = []

count_user = 1
for user in common:
    num = 5
    print(count_user)
    count_user += 1
    count = 0
    try:
      while len(user_movie_pred2[user]) != 15:
          movie_id = (test75_groupby.loc[user])[::-1].iloc[count]['movieId']
          for i in movie_id:
              array = (cosine_similarity(np.array(movie_genre.loc[i]).reshape(1, 19), movie_genre))[0].argsort()[-num:][::-1]
              user_movie_pred2[user].extend(identify[identify.index.isin(array.tolist())].movieId.tolist())
              if num > 1:
                  num -= 1
          user_movie_pred2[user] = list(set(user_movie_pred2[user]).difference(set(test_75[test_75['userId'] == user].movieId.tolist())))
          count += 1
    except IndexError:
      pass

for user in user_movie_pred2:
    match = set(user_movie_pred2[user]).intersection(set(test_25[test_25['userId'] == user].movieId.tolist()))
    if len(match) > 0:
      print(user)
      print(list(match), len(match))

count = 0
for i in user_movie_pred2:
  if len(user_movie_pred2[i]) > 0:
    count += 1
print(count)

## **Predicting From Clusters**

test75_matrix = test_75.pivot_table(index='userId',columns='movieId',values='rating')

test75_matrix = test75_matrix.fillna(0.0)

X = test75_matrix.iloc[:, :].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Users')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(X)

y_hc

import collections
collections.Counter(y_hc)

xyz = pd.concat([pd.DataFrame(test75_matrix.index.tolist(), columns = ['userId']), pd.DataFrame(y_hc, columns = ['Cluster'])], axis = 1)

xyz

ratings = pd.merge(ratings, ratings.groupby('movieId', as_index = False)['rating'].count(), on = 'movieId')

ratings = ratings.rename(columns = {'rating_x': 'rating', 'rating_y': 'count_num'})

gc.collect()

## **Cluster 1**

temp1 = xyz[xyz['Cluster'] == 0]
temp1

cluster_1 = pd.merge(ratings, temp1, left_on = 'userId', right_on = 'userId')

cluster_1.drop(columns = ['Cluster', 'timestamp'], inplace = True)

cluster_1=pd.DataFrame(cluster_1.groupby(['movieId', 'count_num']).agg(count_rate=pd.NamedAgg(column='rating', aggfunc='mean')))

cluster_1 = cluster_1.reset_index()

cluster_1

## **Cluster 2**

temp2 = xyz[xyz['Cluster'] == 1]
temp2

cluster_2 = pd.merge(ratings, temp2, left_on = 'userId', right_on = 'userId')

cluster_2.drop(columns = ['Cluster', 'timestamp'], inplace = True)

cluster_2=pd.DataFrame(cluster_2.groupby(['movieId', 'count_num']).agg(count_rate=pd.NamedAgg(column='rating', aggfunc='mean')))

cluster_2 = cluster_2.reset_index()

cluster_2

## **Cluster 3**

temp3 = xyz[xyz['Cluster'] == 2]
temp3

cluster_3 = pd.merge(ratings, temp3, left_on = 'userId', right_on = 'userId')

cluster_3.drop(columns = ['Cluster', 'timestamp'], inplace = True)

cluster_3=pd.DataFrame(cluster_3.groupby(['movieId', 'count_num']).agg(count_rate=pd.NamedAgg(column='rating', aggfunc='mean')))

cluster_3 = cluster_3.reset_index()

cluster_3

## **Cluster 4**

temp4 = xyz[xyz['Cluster'] == 3]
temp4

cluster_4 = pd.merge(ratings, temp4, left_on = 'userId', right_on = 'userId')

cluster_4.drop(columns = ['Cluster', 'timestamp'], inplace = True)

cluster_4=pd.DataFrame(cluster_4.groupby(['movieId', 'count_num']).agg(count_rate=pd.NamedAgg(column='rating', aggfunc='mean')))

cluster_4 = cluster_4.reset_index()

cluster_4

# **Cross Validation & Conclusion**

test_25 = test_25[test_25['userId'].isin(xyz.userId.tolist())]

test_25 = pd.DataFrame(test_25.groupby(['userId'])['movieId'].unique())

test_25 = test_25.reset_index()

test_25 = pd.merge(test_25, xyz, on = 'userId')

mean_dict = {}
for k in range(2000, 4501, 200):
  for j in range(2000, 3001, 100):
    cluster_1_movies = (cluster_1[cluster_1['count_num'] > k].nlargest(j, 'count_rate')).movieId.tolist()
    cluster_2_movies = (cluster_2[cluster_2['count_num'] > k].nlargest(j, 'count_rate')).movieId.tolist()
    cluster_3_movies = (cluster_3[cluster_3['count_num'] > k].nlargest(j, 'count_rate')).movieId.tolist()
    cluster_4_movies = (cluster_4[cluster_4['count_num'] > k].nlargest(j, 'count_rate')).movieId.tolist()
    movie_dict = {0 : cluster_1_movies, 1 : cluster_2_movies, 2 : cluster_3_movies, 3 : cluster_4_movies}
    test_25['moviesPredicted'] = test_25['Cluster'].apply(lambda x : movie_dict[x])
    test_25['correctPredictions'] = 0
    test_25['recall'] = 0.0
    for i in test_25.index:
      test_25.loc[i, 'correctPredictions'] = len(set(test_25.iloc[i]['movieId']).intersection(set(test_25.iloc[i]['moviesPredicted'])))
    for i in test_25.index:
      test_25.loc[i, 'recall'] = round(float(test_25.iloc[i]['correctPredictions'])/float(len(test_25.iloc[i]['movieId'])), 2)
    mean_dict[(k, j)] = test_25.describe()['recall']['mean']

mean_dict

  cluster_1_movies = (cluster_1[cluster_1['count_num'] > 2000].nlargest(2500, 'count_rate')).movieId.tolist()
  cluster_2_movies = (cluster_2[cluster_2['count_num'] > 2000].nlargest(2500, 'count_rate')).movieId.tolist()
  cluster_3_movies = (cluster_3[cluster_3['count_num'] > 2000].nlargest(2500, 'count_rate')).movieId.tolist()
  cluster_4_movies = (cluster_4[cluster_4['count_num'] > 2000].nlargest(2500, 'count_rate')).movieId.tolist()
  movie_dict = {0 : cluster_1_movies, 1 : cluster_2_movies, 2 : cluster_3_movies, 3 : cluster_4_movies}
  test_25['moviesPredicted'] = test_25['Cluster'].apply(lambda x : movie_dict[x])
  test_25['correctPredictions'] = 0
  test_25['recall'] = 0.0
  for i in test_25.index:
    test_25.loc[i, 'correctPredictions'] = len(set(test_25.iloc[i]['movieId']).intersection(set(test_25.iloc[i]['moviesPredicted'])))
  for i in test_25.index:
    test_25.loc[i, 'recall'] = round(float(test_25.iloc[i]['correctPredictions'])/float(len(test_25.iloc[i]['movieId'])), 2)

test_25

test_25.describe()

test_25[['userId', 'recall']]

test_25[['userId', 'recall']].to_csv("/content/drive/My Drive/DS Project 20M/predictions1.csv", index = False)

# **Recommending Similar Movies**

moviemat = total.pivot_table(index='userId',columns='title',values='rating')

moviemat = moviemat.fillna(0.0)

ratings_movies = pd.DataFrame(total.groupby('title')['rating'].mean())

ratings_movies['num of ratings'] = pd.DataFrame(total.groupby('title')['rating'].count())

ratings_movies.sort_values('num of ratings',ascending=False).head(10)

MOVIES = (ratings_movies.sort_values('num of ratings',ascending=False).head(10)).drop(columns = ['rating', 'num of ratings'])

lords_user_ratings = moviemat['Lord of the Rings: The Fellowship of the Ring, The (2001)']
the_matrix_user_ratings = moviemat['Dark Knight, The (2008)']

similar_to_lords = moviemat.corrwith(lords_user_ratings)
similar_to_the_matrix = moviemat.corrwith(the_matrix_user_ratings)

corr_lords = pd.DataFrame(similar_to_lords,columns=['Correlation'])
corr_lords.dropna(inplace=True)
corr_lords = corr_lords.join(ratings_movies['num of ratings'])
corr_lords[corr_lords['num of ratings']>100].sort_values('Correlation',ascending=False).head(6)

corr_the_matrix = pd.DataFrame(similar_to_the_matrix,columns=['Correlation'])
corr_the_matrix.dropna(inplace=True)
corr_the_matrix = corr_the_matrix.join(ratings_movies['num of ratings'])
corr_the_matrix[corr_the_matrix['num of ratings']>100].sort_values('Correlation',ascending=False).head(6)

def magic_function(s):
  movie_user_ratings = moviemat[s]
  similar_to_movie = moviemat.corrwith(movie_user_ratings)
  corr_movies = pd.DataFrame(similar_to_movie,columns=['Correlation'])
  corr_movies.dropna(inplace=True)
  corr_movies = corr_movies.join(ratings_movies['num of ratings'])
  return ((corr_movies[corr_movies['num of ratings']>100].sort_values('Correlation',ascending=False)).head(6)).drop(columns = ['Correlation',	'num of ratings'])

# **MAGIC FUNCTION**

MOVIES

magic_function('Inception (2010)')

