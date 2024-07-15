from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import request




# # Define the function to give recommendations
# def give_recommendations(movie_title):
#     print("execution intialized !!!!")
#     print(movie_title)
#     # Load your data
#     sele_cols = ['adult', 'genres', 'overview', 'title', 'id']
#     data = pd.read_csv(r"D:\vijju\recommender_system\recommender_system\movies_metadata.csv", usecols=sele_cols)

#     # Fill NaN values in the 'overview' column with an empty string
#     data['overview'] = data['overview'].fillna('')

#     # Overview corpus for similarity
#     corpus = data['overview'].tolist()

#     tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
#     def get_movie_index_by_title(movie_title):
#          data[data['title'] == movie_title].index[0]

#     # Calculate the cosine similarity for the overviews
#     cos_sim_data = pd.DataFrame(cosine_similarity(tfidf_matrix))
#     index = get_movie_index_by_title(movie_title)
#     index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:11]
#     movies_recomm = data['title'].loc[index_recomm].values
#     result = {'Movies': movies_recomm, 'Index': index_recomm}
#     print(movies_recomm)
    
#     # print(movies_recomm)
#     # if print_recommendation:
#     #     print('The watched movie is: %s\n' % data['title'].loc[index])
#     #     k = 1
#     #     for movie in movies_recomm:
#     #         print('The number %i recommended movie is: %s\n' % (k, movie))
#     #         k += 1
#     # if print_recommendation_plots:
#     #     print('The plot of the watched movie is:\n %s\n' % data['overview'].loc[index])
#     #     k = 1
#     #     for q in range(len(movies_recomm)):
#     #         plot_q = data['overview'].loc[index_recomm[q]]
#     #         print('The plot of the number %i recommended movie is:\n %s\n' % (k, plot_q))
#     #         k += 1
#     return 


def give_recommendations(movie_title):
    

    # Load your data
    sele_cols = ['adult', 'genres', 'overview', 'title', 'id']
    data = pd.read_csv(r"D:\vijju\recommender_system\recommender_system\movies_metadata.csv", usecols=sele_cols)

    # Fill NaN values in the 'overview' column with an empty string
    data['overview'] = data['overview'].fillna('')

    # Check if the movie_title exists in the DataFrame
    if movie_title not in data['title'].values:
        return ["Movie not found in the database"]

    # Overview corpus for similarity
    corpus = data['overview'].tolist()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    def get_movie_index_by_title(title):
        return data[data['title'] == title].index[0]

    # Calculate the cosine similarity for the overviews
    cos_sim_data = pd.DataFrame(cosine_similarity(tfidf_matrix))

    index = get_movie_index_by_title(movie_title)
    index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:11]
    movies_recomm = data['title'].loc[index_recomm].values
    result = {'Movies': movies_recomm, 'Index': index_recomm}
    print(movies_recomm)
    return movies_recomm



# # # # Example usage: give recommendations for a movie title
# movie_title = ""
# recommendations = give_recommendations(movie_title)

