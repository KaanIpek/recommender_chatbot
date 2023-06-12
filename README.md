Movie Recommender Chatbot
This chatbot application is a movie recommendation system built using the Flask framework, Pandas, scikit-learn, and OpenAI's GPT-3.5-turbo. It can recommend movies based on a given movie title, director, or genre.

Installation
First, you need to install the necessary Python libraries for this project. You can install them using pip:


pip install flask pandas sklearn openai

Dataset
The chatbot uses the MovieLens Latest Small dataset for recommendations. You will need to download and extract the dataset to the ml-latest-small/ directory in your project.

The dataset files used are:

movies.csv
ratings.csv
links.csv
title_crew.tsv
name_basic.tsv
Running the Application
To run the application, use the following command:

python app2.py


Available Commands Usage
Recommend movies based on a title: "I watched "Forrest Gump""
Recommend more movies: "5 more recommendations"
Recommend movies by a director: "Director "Christopher Nolan""
Recommend movies of a genre: "This genre "Action""
Random chat: Any other input will be processed by GPT-3.5-turbo for a random chat.
Please note that for title, director, and genre, the input should be enclosed in double quotes.

License
This project is licensed under the MIT License.