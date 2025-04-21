
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

@st.cache_data
def load_data():
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                          sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                         sep='|', encoding='latin-1',
                         names=['movieId', 'title', 'release_date', 'video_release_date',
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
                                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                                'War', 'Western'])
    data = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
    return data, movies

@st.cache_resource
def compute_models(data, movies):
    user_movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
    user_sim_matrix = cosine_similarity(user_movie_matrix.fillna(0))
    user_sim_df = pd.DataFrame(user_sim_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    movie_titles = movies[['movieId', 'title']].copy()
    movie_titles['title_cleaned'] = movie_titles['title'].str.lower()
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_titles['title_cleaned'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    return user_movie_matrix, user_sim_df, movie_titles, cosine_sim

def hybrid_recommendations(user_id, data, movies, user_sim_df, user_movie_matrix, movie_titles, cosine_sim, top_n=10, cf_weight=0.6, genre_filter=None):
    rated_movies = data[data['userId'] == user_id]['movieId'].tolist()
    if genre_filter:
        movies = movies[movies[genre_filter] == 1]
    unrated_movies = movies[~movies['movieId'].isin(rated_movies)].copy()

    sim_users = user_sim_df.loc[user_id]
    cf_scores = user_movie_matrix.T.dot(sim_users)
    valid_mask = user_movie_matrix.notna()
    denom = valid_mask.mul(sim_users, axis=0).sum()
    cf_scores = cf_scores / denom

    scores = []
    for movie_id in unrated_movies['movieId']:
        cf_score = cf_scores.get(movie_id, 2.5)
        try:
            idx1 = movie_titles.index[movie_titles['movieId'] == movie_id][0]
            idxs_rated = movie_titles[movie_titles['movieId'].isin(rated_movies)].index
            cb_score = cosine_sim[idx1, idxs_rated].mean()
        except:
            cb_score = 0
        hybrid_score = cf_weight * cf_score + (1 - cf_weight) * cb_score
        scores.append((movie_id, hybrid_score))

    top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    result = movies[movies['movieId'].isin([m[0] for m in top_movies])][['title']]
    result['Score'] = [m[1] for m in top_movies]
    return result

# Streamlit UI
st.set_page_config(page_title="CineMatch", layout="wide")
st.title("🎬 CineMatch: Hybrid Movie Recommendation System")

data, movies = load_data()
user_movie_matrix, user_sim_df, movie_titles, cosine_sim = compute_models(data, movies)

col1, col2 = st.columns([2, 1])
with col1:
    user_id = st.slider("Select User ID", min_value=1, max_value=943, value=10)
    top_n = st.slider("Number of Recommendations", 5, 20, 10)
    cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6)

with col2:
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                  'Sci-Fi', 'Thriller', 'War', 'Western']
    genre_filter = st.selectbox("Filter by Genre (optional)", ["None"] + genre_cols)
    genre_filter = None if genre_filter == "None" else genre_filter

recommendations = hybrid_recommendations(user_id, data, movies, user_sim_df, user_movie_matrix, movie_titles, cosine_sim, top_n, cf_weight, genre_filter)

st.markdown("### 🎥 Recommended Movies")
st.dataframe(recommendations, use_container_width=True)

chart = alt.Chart(recommendations.reset_index()).mark_bar().encode(
    x=alt.X('Score:Q', title='Hybrid Score'),
    y=alt.Y('title:N', sort='-x', title='Movie Title'),
    color=alt.value('#1f77b4')
).properties(width=700, height=400)

st.altair_chart(chart)
