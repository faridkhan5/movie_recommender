import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_pickle('movies.pkl')
movies_lst = movies_df['title'].values

###Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
Mat = cv.fit_transform(movies_df['tags']).toarray()


###Similarity Mat
similarityMat = cosine_similarity(Mat)
similarityMat_df = pd.DataFrame(similarityMat)


###Recommender Model
def recommend(movie):
  #fetching idx of a movie
  movie_idx = movies_df[movies_df['title'] == movie].index[0]
  dist = similarityMat_df[movie_idx]
  #vec of similarity of 1 movie with the rest
  movies_lst = sorted(list(enumerate(dist)), reverse=True, key=lambda x:x[1])[1:6]
  #getting the top 5 similar movies
  movies_lst_title = []
  #(idx, cosine) -> 'title'
  for i in  movies_lst:
    movies_lst_title.append(movies_df.iloc[i[0]].title)
  return movies_lst_title



st.title("Movie Recommender")

selected_movie = st.selectbox("how would you like to be contacted",
                      movies_lst)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)