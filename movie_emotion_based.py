# # #!/usr/bin/env python
# # # coding: utf-8

# # # In[1]:




# # # In[3]:


# # #!pip install text2emotion 
# # #!pip install clean-text
# # # !pip install --upgrade emoji 
# # # !pip install --upgrade clean-text


# # # In[4]:


# overview = movies_metadata.overview
# # emotion = [] # for storing emotion dictionaries 
# # print(type(overview))


# # # In[5]:



# # # In[7]:


# # overview


# # # 

# # # In[8]:


# # # step 1 : checking for emojis and removing if any 
# # # step 2 : cleaning the data 
# # # step 3:  generate sentiment/emotinos using nltk packages for entire dataset 
# # # step 4: form a recommender system based on acquired attributes 

# # emotions = ['happy','sad','excitement','anger','love','fear','surprise','hope','humor'] # deciding the possible emotions


# # # In[9]:


# # # cleaning emojis if any from overview 




# # # In[11]:


# # # checking and dropping NA values 
# # data.isna().sum()


# # # In[12]:


# # # downloading required libraries 
# # import pandas as pd
# # import nltk
# # from nltk.sentiment.vader import SentimentIntensityAnalyzer
# # from nltk.tokenize import sent_tokenize, word_tokenize
# # from nltk.corpus import stopwords
# # # Download necessary NLTK resources
# # nltk.download('vader_lexicon')
# # nltk.download('punkt')
# # nltk.download('stopwords')


# # # In[13]:



# # # Function to preprocess text
# def preprocess_text(text):
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text.lower())
#     words = [word for word in words if word.isalpha() and word not in stop_words]
#     return ' '.join(words)


# # # In[14]:



# # # # Function to get emotion labels based on sentiment scores
# # # def get_emotion_label(sentiment_scores):
# # #     # Extract sentiment scores
# # #     pos_score = sentiment_scores['pos']
# # #     neg_score = sentiment_scores['neg']
# # #     neu_score = sentiment_scores['neu']
    
# # #     # Define emotion labels and their corresponding ranges for sentiment scores
# # #     emotion_labels = {
# # #         'happy': (0.4, 1.0),
# # #         'sad': (-0.1, -0.4),
# # #         'excitement': (0.1, 0.4),
# # #         'anger': (-1.0, -0.4),
# # #         'love': (0.4, 1.0),
# # #         'fear': (-1.0, 0.0),
# # #         'surprise': (0.0, 0.25),
# # #         'hope': (0.25, 0.5),
# # #         'humor': (0.4, 1.0)
# # #     }
    
# # #     # Determine the emotion based on sentiment scores
# # #     for emotion, (min_range, max_range) in emotion_labels.items():
# # #         if min_range <= pos_score <= max_range:
# # #             return emotion
    
# # #     # Default to neutral if no specific emotion is detected
# # #     return 'neutral'


# # # In[14]:


# # overall_emotion = {item:0 for item in emotions}
# # print(overall_emotion)


# # # In[ ]:





# # # In[15]:


# # data = data.dropna()


# # # In[17]:


# # data


# # # In[24]:


# # # # creating a place holder for emotions

# # # emotion_h = []

# # # # Analyze each movie overview and recommend a movie based on emotion
# # # for i, row in data.iterrows():
# # #     overview = row['overview']
# # #     preprocessed_overview = preprocess_text(overview)
    
# # #     # Initialize the SentimentIntensityAnalyzer
# # #     analyzer = SentimentIntensityAnalyzer()
    
# # #     sentences = sent_tokenize(preprocessed_overview)

    
# # #     for sentence in sentences:
# # #         sentiment_scores = analyzer.polarity_scores(sentence)
# # #         emotion_label = get_emotion_label(sentiment_scores)
# # #         if len(emotion_label) == 0:
# # #             print(1)
# # #         overall_emotion[emotion_label] += 1
# # #         emotion_h.append(emotion_label)
    
# # #     # Get the dominant emotion for the overview
# # #     dominant_emotion = max(overall_emotion, key=overall_emotion.get)
    
   


# # # In[23]:


# # pip install textblob


# # # In[24]:
# import re

#     def remove_emojis(text):
#         # checking and type casting into string if any 
#         if type(text) is not str:
#             text = str(text)
            
#         # Remove emojis using a regular expression
#         emoji_pattern = re.compile(
#             pattern="["
#                     "\U0001F600-\U0001F64F"  # emoticons
#                     "\U0001F300-\U0001F5FF"  # symbols & pictographs
#                     "\U0001F680-\U0001F6FF"  # transport & map symbols
#                     "\U0001F700-\U0001F77F"  # alchemical symbols
#                     "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
#                     "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
#                     "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
#                     "\U0001FA00-\U0001FA6F"  # Chess Symbols
#                     "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
#                     "\U0001F004-\U0001F0CF"  # Additional emoticons
#                     "\U0001F170-\U0001F251"  # Enclosed characters
#                     "]+",
#             flags=re.UNICODE
#         )
        
#         # Remove emojis from the text
#         return emoji_pattern.sub(r'', text)




    # clean_text = overview.apply(remove_emojis)
    # print("Cleaned text:", clean_text)


    # # In[10]:




# # In[25]:


# data_emotions.value_counts()


# In[28]:




# In[33]:


# OBJECTIVE FUNCTION OF EMOTION RECOMMENDER SYSTEM 

def emotion_based_recommender(movie_title):
    import pandas as pd 

    import numpy as np 


    # # In[2]:
    print("execution initialized")

    # # filter 1 --> selecting the required columns of metadata 

    sele_cols = ['adult','orginal_language','overview','title','id'] # filter 2 not importing genres 


    movies_metadata = pd.read_csv(r"D:\vijju\recommender_system\recommender_system\movies_metadata.csv",low_memory=False)


    overview = movies_metadata.overview

    

    # # creating dataframe 
    data = pd.DataFrame()
    data['overview'] = overview
    data['title'] = movies_metadata.title


    # converting overview into string if any
    data['overview'] = data['overview'].astype(str)


    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import pandas as pd
    from textblob import TextBlob


    # Function to get the dominant emotion from a text
    def get_dominant_emotion(text):
        if isinstance(text, float):
            return 'Unknown'  # Handle cases where 'overview' is NaN (float)
        analysis = TextBlob(str(text))  # Convert to string
        polarity = analysis.sentiment.polarity

        if polarity > 0.5:
            return 'Happy'
        elif polarity < -0.5:
            return 'Sad'
        elif polarity > 0:
            return 'Neutral'
        elif polarity < 0 and polarity > -0.5:
            return 'Angry'
        else:
            return 'Surprised'


    # Sample data (a Pandas Series)
    # Apply emotion analysis to th(9e series
    data_emotions = overview.apply(get_dominant_emotion)


    data['emotions'] = data_emotions


    # In[30]:


    # intially one hot encode the emotions for calculating cosine_similarity 

    one_hot_encoded = pd.get_dummies(data_emotions)


    # In[31]:


    # calculate cosine similarity for emotions 

    from sklearn.metrics.pairwise import cosine_similarity 

    cos_sim_data = cosine_similarity(one_hot_encoded)
        # finding if we have title in our dataframe 
    if movie_title not in data['title'].values:
        return "Movie is not available at the moment "
    else:
        # finding the movie index in the dataframe 
        
        movie_index = data[data['title'] == movie_title].index[0]    
        
        # get the emotion of input movie 
        
        movie_emotion = data[data['title'] == movie_title].iloc[0]['emotions']
        
        print(" The emotion of selected movie is " + str(movie_emotion))

        # calculating the similarity scores for the specified movie 

        similar_scores = list(enumerate(cos_sim_data[movie_index]))
        
        similar_scores = sorted(similar_scores,key=lambda x:x[1], reverse = True )
        
        # get the top 10 similar movies 
        
        top_similar_movies = similar_scores[1:10]
        
        # Get the titles of the top similar movies
        similar_movie_titles = [data.iloc[score[0]]['title'] for score in top_similar_movies]      
        
        return similar_movie_titles
          


# In[34]:


# # Example usage: recommend movies similar to 'Movie A' in terms of emotion
# movie_title = 'Toy Story'
# recommended_movies = emotion_based_recommender(movie_title)

# print(f"Recommended movies for '{movie_title}':")
# for movie in recommended_movies:
#     print(movie)

