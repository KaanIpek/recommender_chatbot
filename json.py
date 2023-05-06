import pandas as pd

movies = pd.read_csv('ml-latest-small/movies.csv')
movies[['movieId', 'title']].to_json('movies.json', orient='records')
