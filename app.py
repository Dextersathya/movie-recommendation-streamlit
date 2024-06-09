import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
df = pd.read_csv("./model/tmdb.csv")

# Create the count matrix and compute the cosine similarity matrix
count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df["soup"])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset the dataframe index
df = df.reset_index()


# Function to get recommendations
def get_recommendations(title):
    idx = df.loc[df["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][["title", "release_date"]]


# Streamlit app
def main():
    # Set the title and background image
    st.title("Movie Recommender System")
    st.markdown(
        """
    <style>
    body {
        background-image: url("https://your-background-image-url.jpg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.write("Select a movie to get recommendations:")

    # Add an empty option at the beginning of the list
    all_titles_with_empty = [""] + df["title"].tolist()

    # Create a form for the input and submit button
    movie_name = st.selectbox("Choose a movie", all_titles_with_empty)

    # Show recommendations if a movie is selected
    if movie_name:
        result_final = get_recommendations(movie_name)
        st.write(f"Recommendations for '{movie_name}':")
        for index, row in result_final.iterrows():
            st.write(f"{row['title']} ({row['release_date']})")


if __name__ == "__main__":
    main()
