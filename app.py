
from flask import Flask, request, jsonify, render_template,request,redirect,url_for


app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')
# start with genre based recommender 
@app.route('/genre_recommender/',methods=['POST',"GET"])
def genre_recommender():
    # Your route handling code here


    if request.method == 'POST':
        target_genre = request.form.get('genre')
        print("The target_genre is",target_genre)
        import movies_genres_based
        recommendations = movies_genres_based.give_recommendations_genres(target_genre)
        print(type(recommendations))
        return render_template('recommender.html',recommendations = recommendations)
    return render_template('genre_recommender.html')

# emotion based recommender system 

@app.route('/emotion_recommender/',methods=['POST','GET'])
def emotion_recommender():
    if request.method == "POST":
        movie_title = request.form.get('movie_title')
        import movie_emotion_based
        recommendations = movie_emotion_based.emotion_based_recommender(movie_title)
        print(recommendations)
        return render_template('recommender1.html',recommendations = recommendations)
    return render_template('emotion_recommender.html')


# content based recommender system 

@app.route('/content_recommender/',methods = ['POST',"GET"])
def content_recommender():
    if request.method == "POST":
        movie_title = request.form.get('movie_title')
        import content_based 
        recommendations = content_based.give_recommendations(movie_title)
        print(recommendations)
        return render_template('recommender2.html',recommendations = recommendations)
    return render_template('content_recommender.html')

if __name__ == "__main__":
    app.run(debug=True)
# # Load your data
# sele_cols = ['adult', 'genres', 'overview', 'title', 'id']
# data = pd.read_csv("movies_metadata.csv", usecols=sele_cols)

# # Data preprocessing
# data['overview'] = data['overview'].fillna('')  # Replace NaN with empty strings

# # Overview corpus for similarity
# corpus = data['overview'].tolist()

# # Load the SentenceTransformer model
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# # load sentence transformer embeddings from the  saved file 
# with open('overview_embeddings.pkl', 'rb') as f:
#     overview_embeddings = pickle.load(f)

# # Encode movie overviews into embeddings
# #overview_embeddings = model.encode(corpus, show_progress_bar=True)

# # Calculate cosine similarity based on embeddings
# cos_sim_data = pd.DataFrame(cosine_similarity(overview_embeddings))

# # Define a function to get the movie index by title
# def get_movie_index_by_title(movie_title):
#     movie = data[data['title'] == movie_title]
#     if not movie.empty:
#         return movie.index[0]
#     else:
#         return None

# # Define the function to give recommendations
# def give_recommendations(movie_title):
#     index = get_movie_index_by_title(movie_title)

#     if index is not None:
#         index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:11]
#         movies_recomm = data['title'].loc[index_recomm].values
#         return {'Movies': movies_recomm}
#     else:
#         return "Movie not found in the dataset."


# @app.route('/content_recommend', methods=['POST'])
# def content_recommend():
#     movie_title = request.form.get('movie_title')
#     recommendations = give_recommendations(movie_title)
#     return render_template('content_recommender.html',movie_title = movie_title,recommendations = recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)

# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import pickle

# # Load your data
# sele_cols = ['adult', 'genres', 'overview', 'title', 'id']
# data = pd.read_csv("D:/vijju/recommender_system/movies_dataset/movies_metadata.csv", usecols=sele_cols)

# # Data preprocessing
# data['overview'] = data['overview'].fillna('')  # Replace NaN with empty strings

# # Overview corpus for similarity
# corpus = data['overview'].tolist()

# # Load the SentenceTransformer model
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# # Encode movie overviews into embeddings
# overview_embeddings = model.encode(corpus, show_progress_bar=True)

# # Save the embeddings to a file using pickle
# with open('overview_embeddings.pkl', 'wb') as f:
#     pickle.dump(overview_embeddings, f)
#                                                                                                                                                                                 # 
                                                                                                                                                                                       