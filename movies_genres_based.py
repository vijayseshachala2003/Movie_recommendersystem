#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

import numpy as np 


# In[2]:




# In[3]:



# In[ ]:





# In[5]:


#'genre', 'id', 'imdb_id', 'overview', 'title'


# In[6]:




# In[7]:


# # creating dataframe 
# data = pd.DataFrame()
# data['genres'] = dictio_new
# data['id'] = movies_metadata.id
# data['title'] = movies_metadata.title
# data['adult'] = movies_metadata.adult 
# data['spoken_languages'] = movies_metadata.spoken_languages


# In[8]:



# In[9]:




# In[10]:


# unique_genres = []
# for i in genre_names:
#     for j in i:
#         if j not in unique_genres:
#             unique_genres.append(j)
#     else:
#         pass


# # In[11]:


# unique_genres  # all the collection of various genres in movies_metadata


# In[12]:


# intialize MultiLabelBinazier 
from sklearn.preprocessing import MultiLabelBinarizer 
# one_hot_df = pd.DataFrame(one_hot_encoded,columns = mlb.classes_)
# print(one_hot_df)


# In[13]:


 


# In[14]:


from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# In[15]:



# In[16]:




# In[17]:



# def give_recommendations_genres(target_genre):
    
# # filter 1 --> selecting the required columns of metadata 

#     sele_cols = ['adult','genres','orginal_language','overview','title','id']


#     movies_metadata = pd.read_csv(r"D:\vijju\recommender_system\recommender_system\movies_metadata.csv",low_memory = False)
    
    
#     dictio = movies_metadata.genres

#     # the content of the above is in str format & hence converting it into dictionary 
#     # first checking for none or null strings 
#     # found no null strings
#     # let the fun begin !!!! 
#     import json
#     dictio_new = []
#     for i in dictio:
#         dictio_new.append(json.loads(i.replace("'","\"")))


        
#     # extracting genre names from the dictionaries 
#     genre_names = [[item['name'] for item in sublist] for sublist in dictio_new]
#     mlb = MultiLabelBinarizer()
#     one_hot_encoded = mlb.fit_transform(genre_names)
    
#     movie_titles = movies_metadata.title # collecting movie titles from dataset
    
#     # using SVD for developing recommendations 
#     np_data = np.array(one_hot_encoded) # converting into numpy array for easier manipulation

#     # performing SVD 
#     n_components = min(np_data.shape) - 1  # Choose the number of components for SVD
#     svd = TruncatedSVD(n_components=n_components)
#     svd_matrix = svd.fit_transform(np_data)
#     genres = target_genre
#     target_genre_idx = genres.index(target_genre)
#     similarity_scores = cosine_similarity(np_data)
#     # get the similarity factor using cosine similarity 
#     similar_movies = sorted(list(enumerate(similarity_scores[target_genre_idx])),key = lambda x:x[1],reverse = True)
#     similar_movies = [movie for movie in similar_movies if np_data[movie[0], target_genre_idx] != 1]
#     # Get recommended movies and their similarity scores
#     recommendations = [{'movie_title': movie_titles[idx], 'similarity_score': score} for idx, score in similar_movies]
#     max_similarity_scores = max(rec['similarity_score'] for rec in recommendations)
#     print(f"Recommended movies for the genre '{target_genre}':")
#     count = 0
#     dummy = []
#     for rec in recommendations:
#         count += 1
        
#         if rec['similarity_score'] == max_similarity_scores:
#             if count<11:
#                 rec['movie_title']
#             else:
#                 break
#     print("done")
    
#     #return dummy





def give_recommendations_genres(target_genre):
    
    # filter 1 --> selecting the required columns of metadata 

    sele_cols = ['adult','genres','orginal_language','overview','title','id']

    movies_metadata = pd.read_csv(r"D:\vijju\recommender_system\recommender_system\movies_metadata.csv", low_memory=False)

    dictio = movies_metadata.genres

    # the content of the above is in str format & hence converting it into a dictionary 
    # first checking for none or null strings 
    # found no null strings
    # let the fun begin !!!
    import json
    dictio_new = []
    for i in dictio:
        dictio_new.append(json.loads(i.replace("'", "\"")))

    # extracting genre names from the dictionaries
    genre_names = [[item['name'] for item in sublist] for sublist in dictio_new]
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(genre_names)

    movie_titles = movies_metadata.title  # collecting movie titles from the dataset

    # using SVD for developing recommendations
    np_data = np.array(one_hot_encoded)  # converting into a numpy array for easier manipulation

    # performing SVD
    n_components = min(np_data.shape) - 1  # Choose the number of components for SVD
    svd = TruncatedSVD(n_components=n_components)
    svd_matrix = svd.fit_transform(np_data)
    genres = target_genre
    target_genre_idx = genres.index(target_genre)
    similarity_scores = cosine_similarity(np_data)

    # get the similarity factor using cosine similarity
    similar_movies = sorted(enumerate(similarity_scores[target_genre_idx]), key=lambda x: x[1], reverse=True)
    similar_movies = [movie for movie in similar_movies if np_data[movie[0], target_genre_idx] != 1]

    # Get recommended movies and their similarity scores
    recommendations = [{'movie_title': movie_titles[idx], 'similarity_score': score} for idx, score in similar_movies]
    max_similarity_scores = max(rec['similarity_score'] for rec in recommendations)
    
    # Collect the top 10 movie titles in a list
    dummy = []
    for rec in recommendations:
        if rec['similarity_score'] == max_similarity_scores and len(dummy) < 10:
            dummy.append(rec['movie_title'])
    
    return dummy



# In[18]:

# from flask import request
# # Test the recommender system using genres 
# target_genre = 'Animation' # Change this to the genre you want recommendations for
# recommendations = give_recommendations_genres(target_genre)
# print(recommendations)

print("trial worked but didnt get results")


# In[24]:




# In[ ]:





# In[21]:





# In[ ]:




