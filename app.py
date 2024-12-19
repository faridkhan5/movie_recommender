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


# Streamlit App UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("Movie Recommender")
st.markdown("### Get movie recommendations based on your selected movie")

selected_movie = st.selectbox("Select movie name",
                              movies_lst)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)

    st.markdown(f"### Top 5 Movies Similar to: *{selected_movie}*")

    for movie in recommendations:
        st.markdown(f"""
            <div class="movie-box">
                <h3>{movie}</h3>
            </div>
        """, unsafe_allow_html=True)


st.markdown("""
        <style>
            .movie-box {
              width: 50%;
              border: 2px solid green;
              border-radius: 10px;
              padding: 10px;
              margin-bottom: 20px;
              box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
              color: 
            }

            .stButton>button {
              color: white;
              font-size: 16px;
              padding: 10px 20px;
              border: 2px solid green;
            }

            .stButton>button:hover {
              color: white;
              border: 2px solid green;
              background-color: green;
            }
        </style>
    """, unsafe_allow_html=True)