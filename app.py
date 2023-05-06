from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Create Flask application
app = Flask(__name__)

# Load movie and rating data
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
# Extract all unique genres
unique_genres = set()
for index, row in movies.iterrows():
    genres = row['genres'].split('|')
    unique_genres.update(genres)

unique_genres = sorted(list(unique_genres))
# Convert ratings to a pivot table
ratings_matrix = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')

import json

# Create a JSON file of movie titles
@app.route('/movies.json')
def movies_json():
    movie_titles = movies[['movieId', 'title']].to_dict(orient='records')
    return json.dumps(movie_titles)

@app.route('/')
def home():
    return render_template('index.html', unique_genres=unique_genres)


@app.route('/recommend_genre', methods=['POST'])
def recommend_genre():
    genre = request.form['genre']
    recommended_movies = recommend_movies_by_genre(genre)
    return render_template('recommend.html', movie_title=f"Top {genre} Movies", recommended_movies=recommended_movies.to_dict('records'))

def recommend_movies_by_genre(genre):
    # Filter movies by the selected genre
    genre_movies = movies[movies['genres'].str.contains(genre)]

    # Get the top 10 highest-rated movies in the selected genre
    recommended_movies = genre_movies.merge(ratings.groupby('movieId')['rating'].mean().reset_index(),
                                            on='movieId').nlargest(10, 'rating')

    return recommended_movies[['title', 'genres', 'rating']]


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_title = request.form['movie_title']

    # Get the movie ID for the given movie title
    movie_id = movies.loc[movies['title'] == movie_title, 'movieId'].values[0]

    # Calculate the similarity matrix between movies
    movie_similarity = cosine_similarity(ratings_matrix.fillna(0).T)

    # Set the index according to the movie ID
    movie_similarity = pd.DataFrame(movie_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

    # Get the top 10 most similar movies
    similar_movies = movie_similarity.loc[movie_id].nlargest(11)[1:].index

    # Find the common movie IDs in similar_movies and ratings_matrix
    common_movie_ids = set(similar_movies).intersection(ratings_matrix.columns)

    # Get the ratings for the similar movies that are present in ratings_matrix
    ratings_for_similar_movies = ratings_matrix[list(common_movie_ids)]

    # Get the users who rated the similar movies highly
    users_rated_similar_movies_highly = ratings_for_similar_movies.stack().reset_index().rename(columns={0: 'rating'}) \
        .loc[lambda df: df['movieId'].isin(common_movie_ids)] \
        .nlargest(10, 'rating')['userId']

    # Get the movies highly rated by these users
    recommended_movies = ratings.loc[lambda df: (df['userId'].isin(users_rated_similar_movies_highly)) &
                                                (~df['movieId'].isin(ratings_matrix[movie_id].dropna().index))] \
        .groupby('movieId')['rating'].mean().reset_index() \
        .nlargest(10, 'rating').merge(movies, on='movieId')

    return render_template('recommend.html', movie_title=movie_title, recommended_movies=recommended_movies.to_dict('records'))


if __name__ == '__main__':
    app.run(debug=True)
