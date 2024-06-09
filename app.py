import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df2 = pd.read_csv("./model/tmdb.csv")

# Create the count matrix and compute the cosine similarity matrix
count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df2["soup"])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset the dataframe index and create a series for the indices
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2["title"])
all_titles = [df2["title"][i] for i in range(len(df2["title"]))]


# Function to get recommendations
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2["title"].iloc[movie_indices]
    dat = df2["release_date"].iloc[movie_indices]
    return_df = pd.DataFrame(columns=["Title", "Year"])
    return_df["Title"] = tit
    return_df["Year"] = dat
    return return_df


# Streamlit app
def main():
    # Set the title and background image
    st.title("Movie Recommender System")

    # CSS to set background image
    page_bg_img = """
    <style>
    body {
        background-image: url("templates/back.jpeg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.write("Select a movie to get recommendations:")

    # Add an empty option at the beginning of the list
    all_titles_with_empty = [""] + all_titles

    # Create a form for the input and submit button
    with st.form(key="movie_form"):
        movie_name = st.selectbox("Choose a movie", all_titles_with_empty)
        submit_button = st.form_submit_button(label="Get Recommendations")

    # Show recommendations if the form is submitted and a movie is selected
    if submit_button and movie_name:
        result_final = get_recommendations(movie_name)
        st.write(f"Recommendations for '{movie_name}':")
        for i in range(len(result_final)):
            st.write(
                f"{result_final.iloc[i]['Title']} ({result_final.iloc[i]['Year']})"
            )


if __name__ == "__main__":
    main()
