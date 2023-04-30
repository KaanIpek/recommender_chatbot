
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load the movie and rating data into dataframes
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# create a pivot table of ratings
ratings_matrix = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')


def recommend_movies(movie_title):
    # get the movie ID for the given movie title
    movie_id = movies.loc[movies['title'] == movie_title, 'movieId'].values[0]

    # calculate the similarity matrix between movies
    movie_similarity = cosine_similarity(ratings_matrix.fillna(0).T)

    # get the top 10 most similar movies
    similar_movies = pd.DataFrame(movie_similarity).iloc[movie_id - 1].nlargest(11)[1:].index

    # find the common movie IDs in similar_movies and ratings_matrix
    common_movie_ids = set(similar_movies).intersection(ratings_matrix.columns)

    # get the ratings for the similar movies that are present in ratings_matrix
    ratings_for_similar_movies = ratings_matrix[list(common_movie_ids)]

    # get the users who rated the similar movies highly
    users_rated_similar_movies_highly = ratings_for_similar_movies.stack().reset_index().rename(columns={0: 'rating'}) \
        .loc[lambda df: df['movieId'].isin(common_movie_ids)] \
        .nlargest(10, 'rating')['userId']

    # get the movies highly rated by these users
    recommended_movies = ratings.loc[lambda df: (df['userId'].isin(users_rated_similar_movies_highly)) &
                                                (~df['movieId'].isin(ratings_matrix[movie_id].dropna().index))] \
        .groupby('movieId')['rating'].mean().reset_index() \
        .nlargest(10, 'rating').merge(movies, on='movieId')

    # display the top 5 recommended movies
    return recommended_movies[['title', 'genres', 'rating']]


def movie_recommender_bot():
    while True:
        print("Hello! I am a recommender bot. Currently I just have movie option. Please enter a movie title.")
        movie_title = input()
        recommended_movies = recommend_movies(movie_title)
        if recommended_movies.empty:
            print("I'm sorry, I could not find any movies to recommend for you. But you can try again!")
        else:
            print("Here are some movies you might like:")
            for index, row in recommended_movies.iterrows():
                print(row['title'], "(" + row['genres'] + ")", "-", row['rating'])
        print("Enjoy your movie!")

        print("Do you want more movie recommendations? (yes/no)")
        answer = input().lower()
        if answer == "yes":
            continue
        elif answer == "no":
            print("Thank you for using our recommender bot. Goodbye!")
            break
        else:
            print("I'm sorry, I did not understand. Please enter 'yes' or 'no'.")


# start the bot
movie_recommender_bot()
