from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

from scipy.sparse import csr_matrix

app = Flask(__name__)

movies = pd.read_csv('ml-latest/movies.csv')
ratings = pd.read_csv('ml-latest/ratings.csv')
links = pd.read_csv('ml-latest/links.csv')
imdb_crew = pd.read_csv('ml-latest-small/title_crew.tsv', delimiter='\t')
imdb_names = pd.read_csv('ml-latest-small/name_basic.tsv', delimiter='\t')
# Load principal cast data
chunksize = 10 ** 6  # adjust this value depending on your system's memory

# Create an empty DataFrame to store the processed chunks
principals_df = pd.DataFrame()

for chunk in pd.read_csv('ml-latest-small/title_principals.tsv', delimiter='\t', chunksize=chunksize):
    # Remove 'tt' prefix and convert 'tconst' to int
    chunk['tconst'] = chunk['tconst'].str[2:].astype(int)

    # Merge with links to get the movieId
    merged_principals = pd.merge(links, chunk, left_on='imdbId', right_on='tconst')

    # 'Explode' the principals so there's one row per movieId-principal pair
    merged_principals['nconst'] = merged_principals['nconst'].str.split(',')
    chunk_df = merged_principals.explode('nconst')

    # Merge with imdb_names to get the actor names
    actors_chunk_df = pd.merge(chunk_df, imdb_names, left_on='nconst', right_on='nconst')

    # Append the processed chunk to the main DataFrame
    principals_df = pd.concat([principals_df, actors_chunk_df])

imdb_crew['tconst'] = imdb_crew['tconst'].str[2:].astype(int)  # remove 'tt' prefix before converting
merged_df = pd.merge(links, imdb_crew, left_on='imdbId', right_on='tconst')

merged_df['directors'] = merged_df['directors'].astype(str)
merged_df['directors'] = merged_df['directors'].str.split(',')

directors_df = merged_df.explode('directors')
directors_with_names_df = pd.merge(directors_df, imdb_names, left_on='directors', right_on='nconst')
directors = directors_with_names_df[['movieId', 'primaryName']].drop_duplicates()

# Create a sparse matrix
sparse_ratings = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))


@app.route('/')
def home():
    return render_template('index2.html')


@app.route('/chat', methods=['POST'])
def handle_chat():
    user_message = request.json['message']
    bot_message = get_chatbot_response(user_message)
    return jsonify(bot_message=bot_message)


def recommend_movies_by_genre(genre, num_recommendations=10):
    genre = genre.lower()
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    recommended_movies = genre_movies.merge(ratings.groupby('movieId')['rating'].mean().reset_index(),
                                            on='movieId').nlargest(num_recommendations, 'rating')
    return recommended_movies[['title', 'genres', 'rating']]
previous_movie_title = ""
previous_movie_recommendations = set()
previous_genre = ""
previous_genre_recommendations = set()
previous_recommendation_count = 0
movie_id = 0
def get_chatbot_response(user_input):
    global previous_recommendation_count
    global previous_recommendations
    global previous_movie_title
    global previous_movie_recommendations
    global previous_genre
    global previous_genre_recommendations
    global movie_id
    user_input_lower = user_input.lower()

    search_pattern_num = re.compile(r'(\d+)\s+(?:more\s+)?(?:recommendations?|movies?)', re.IGNORECASE)
    num_recommendations = search_pattern_num.search(user_input_lower)
    num_recommendations = int(num_recommendations.group(1)) if num_recommendations else 10

    if "recommend" in user_input_lower or "i watch" in user_input_lower:

        if "more" in user_input_lower:
            previous_recommendation_count += 20
            recommended_movies = recommend_movies_based_on_title(movie_id, previous_recommendation_count)

            # Yeni önerileri alın ve önceki önerileri çıkarın
            new_recommendations = recommended_movies[~recommended_movies['title'].isin(previous_movie_recommendations)]

            # Yeni önerileri önceki önerilere ekleyin
            previous_movie_recommendations.update(new_recommendations['title'])

            return "\n".join(new_recommendations['title'].tolist())

        search_pattern = re.compile(r'\"(.*?)\"', re.IGNORECASE)
        movie_title = search_pattern.search(user_input)

        if movie_title:
            movie_title = movie_title.group(1).lower()
        else:
            return "Please specify the movie you want recommendations for in double quotes. For example: \"Forrest Gump\""

        movie_title_pattern = re.compile(r'(.+?)(?:\s\((\d{4})\))?')
        user_title_match = movie_title_pattern.match(movie_title)
        if user_title_match:
            user_title, user_year = user_title_match.groups()
            user_year = int(user_year) if user_year else None
        else:
            return "I couldn't understand the movie title format. Please use a format like \"Movie Title (Year)\" or \"Movie Title\"."

        def match_movie_title(row):
            row_title_match = movie_title_pattern.match(row['title'].lower())
            row_title, row_year = row_title_match.groups()
            row_year = int(row_year) if row_year else None

            if user_year:
                return user_title == row_title and user_year == row_year
            else:
                return user_title == row_title

        matched_movie = movies[movies.apply(match_movie_title, axis=1)]

        if not matched_movie.empty:
            movie_id = matched_movie.iloc[0]['movieId']
            previous_movie_title = movie_title

            recommended_movies = recommend_movies_based_on_title(movie_id, num_recommendations)
            previous_movie_recommendations = recommended_movies
            if not recommended_movies.empty:
                return "\n".join(recommended_movies['title'].tolist())
            else:
                return "I couldn't find any similar movies."
        else:
            return "I couldn't find the movie you're looking for."
    elif "director" in user_input_lower:
            search_pattern = re.compile(r'\"(.*?)\"', re.IGNORECASE)
            director = search_pattern.search(user_input)

            if director:
                director = director.group(1)
                recommended_movies = recommend_movies_by_director(director, num_recommendations)

                if not recommended_movies.empty:
                    return "\n".join(recommended_movies['title'].tolist())
                else:
                    return "I couldn't find any movies for the specified director."
            else:
                return "Please specify the director you want recommendations for in double quotes. For example: \"Christopher Nolan\""
    elif "actor" in user_input_lower:
        search_pattern = re.compile(r'\"(.*?)\"', re.IGNORECASE)
        actor = search_pattern.search(user_input)

        if actor:
            actor = actor.group(1)
            recommended_movies = recommend_movies_by_actor(actor, num_recommendations)

            if not recommended_movies.empty:
                return "\n".join(recommended_movies['title'].tolist())
            else:
                return "I couldn't find any movies with the specified actor."
        else:
            return "Please specify the actor you want recommendations for in double quotes. For example: \"Brad Pitt\""
    elif "this genre" in user_input_lower:
        search_pattern = re.compile(r'\"(.*?)\"', re.IGNORECASE)
        genre = search_pattern.search(user_input)

        if genre:
            genre = genre.group(1)
            if previous_genre != genre.lower():
                previous_genre = genre.lower()
                previous_genre_recommendations.clear()
            recommended_movies = recommend_movies_by_genre(genre, num_recommendations)

            if not recommended_movies.empty:
                return "\n".join(recommended_movies['title'].tolist())
            else:
                return "I couldn't find any movies for the specified genre."
        else:
            return "Please specify the genre you want recommendations for in double quotes. For example: \"Action\""
    elif "hello" in user_input_lower or "hi" in user_input_lower or "hey" in user_input_lower:
        return "Hello! How can I help you?"
    elif "how are you" in user_input_lower:
        return "I'm a chatbot, I don't have feelings. How can I help you?"
    else:
        return "I didn't understand your message. Please try again."
def recommend_movies_by_director(director, num_recommendations=10):
    print(directors_with_names_df['primaryName'].unique())  # Print all unique director names
    director = director.lower()
    director_movies = directors_with_names_df[directors_with_names_df['primaryName'].str.lower() == director]  # Also changed merged_df to directors_with_names_df
    recommended_movies = director_movies.merge(ratings.groupby('movieId')['rating'].mean().reset_index(),
                                               on='movieId').nlargest(num_recommendations, 'rating')
    recommended_movies = pd.merge(recommended_movies, movies, on='movieId', how='left')
    return recommended_movies[['title', 'genres', 'rating']]
def recommend_movies_by_actor(actor, num_recommendations=10):
    actor = actor.lower()
    actor_movies = principals_df[principals_df['primaryName'].str.lower() == actor]  # Changed from actors_chunk_df to principals_df
    recommended_movies = actor_movies.merge(ratings.groupby('movieId')['rating'].mean().reset_index(),
                                            on='movieId').nlargest(num_recommendations, 'rating')
    recommended_movies = pd.merge(recommended_movies, movies, on='movieId', how='left')
    return recommended_movies[['title', 'genres', 'rating']]

def recommend_movies_based_on_title(movie_id, num_recommendations=10):
    movie_similarity = cosine_similarity(sparse_ratings.fillna(0).T)
    movie_similarity = pd.DataFrame(movie_similarity, index=sparse_ratings.columns, columns=sparse_ratings.columns)
    similar_movies = movie_similarity.loc[movie_id].nlargest(num_recommendations + 1)[1:].index
    common_movie_ids = set(similar_movies).intersection(sparse_ratings.columns)
    ratings_for_similar_movies = sparse_ratings[list(common_movie_ids)]
    users_rated_similar_movies_highly = ratings_for_similar_movies.stack().reset_index().rename(columns={0: 'rating'}) \
        .loc[lambda df: df['movieId'].isin(common_movie_ids)] \
        .nlargest(num_recommendations, 'rating')['userId']
    recommended_movies = ratings.loc[lambda df: (df['userId'].isin(users_rated_similar_movies_highly)) &
                                            (~df['movieId'].isin(sparse_ratings[movie_id].dropna().index))].groupby('movieId')['rating'].mean().reset_index().nlargest(num_recommendations, 'rating').merge(movies, on='movieId')
    return recommended_movies

if __name__ == '__main__':
    app.run(debug=True)
