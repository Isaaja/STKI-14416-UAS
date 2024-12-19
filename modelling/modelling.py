# %% Imports and Initial Setup
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import joblib
import re
from rapidfuzz import process, fuzz

# %% Load Data
fulldata = pd.read_csv('C:/SEMESTER 5/STKI/Tugas 3 STKI/data/fulldata.csv')
anime = pd.read_csv('C:/SEMESTER 5/STKI/Tugas 3 STKI/data/anime.csv')

# %% Preprocessing Data
data = fulldata.copy()
data["user_rating"].replace(to_replace=-1, value=np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

# Filter users with at least 50 ratings
selected_users = data["user_id"].value_counts()
data = data[data["user_id"].isin(selected_users[selected_users >= 50].index)]

# Clean anime names
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    return text

data["name"] = data["name"].apply(text_cleaning)

# Create pivot table
data_pivot = data.pivot_table(index="name", columns="user_id", values="user_rating").fillna(0)

# %% Collaborative Filtering (KNN)
data_matrix = csr_matrix(data_pivot.values)

# Train KNN Model
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(data_matrix)

# Function to get KNN recommendations
def knn_recommendation(title, n_neighbors=6):
    # Search title with fuzzy matching
    title_match = process.extractOne(title, data_pivot.index, scorer=fuzz.ratio)
    if not title_match or title_match[1] < 70:  # Adjust threshold as needed
        print("Title not found or match too low in dataset!")
        return None

    matched_title = title_match[0]
    idx = data_pivot.index.get_loc(matched_title)
    distances, indices = model_knn.kneighbors(
        data_pivot.iloc[idx, :].values.reshape(1, -1), n_neighbors=n_neighbors
    )

    # Build recommendations DataFrame
    no = []
    name = []
    rating = []
    for i in range(1, len(indices.flatten())):
        no.append(i)
        anime_name = data_pivot.index[indices.flatten()[i]]
        name.append(anime_name)
        rating.append(*anime[anime["name"] == anime_name]["rating"].values)

    rec_dic = {"No": no, "Anime Name": name, "Rating": rating}
    recommendation = pd.DataFrame(data=rec_dic)
    recommendation.set_index("No", inplace=True)

    print(f"Recommendations for {matched_title} viewers:\n")
    print(recommendation)
    return recommendation

# %% Content-Based Filtering
# Process genres with TF-IDF
tfv = TfidfVectorizer(min_df=3, strip_accents="unicode", analyzer="word",
                      token_pattern=r"\w{1,}", ngram_range=(1, 3), stop_words="english")

rec_data = fulldata.copy()
rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
rec_data.reset_index(drop=True, inplace=True)

genres = rec_data["genre"].str.split(", | , | ,").astype(str)
tfv_matrix = tfv.fit_transform(genres)

# Compute sigmoid kernel
sig_file_path = 'C:/SEMESTER 5/STKI/Tugas 3 STKI/data/sig_kernel.pkl'
try:
    # Load precomputed sigmoid kernel if available
    sig = joblib.load(sig_file_path)
    print("Loaded precomputed sigmoid kernel.")
except FileNotFoundError:
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    joblib.dump(sig, sig_file_path)
    print("Computed and saved sigmoid kernel.")

# Map titles to indices
rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()

# Function to get Content-Based recommendations
def give_recommendation(title, sig=sig):
    # Search title with fuzzy matching
    title_match = process.extractOne(title, rec_indices.index, scorer=fuzz.ratio)
    if not title_match or title_match[1] < 70:  # Adjust threshold as needed
        print("Title not found or match too low in dataset!")
        return None

    matched_title = title_match[0]
    idx = rec_indices[matched_title]
    sig_score = list(enumerate(sig[idx]))
    sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)[1:11]
    anime_indices = [i[0] for i in sig_score]

    # Build recommendations DataFrame
    rec_dic = {"No": range(1, 11),
               "Anime Name": anime["name"].iloc[anime_indices].values,
               "Rating": anime["rating"].iloc[anime_indices].values,
               "Genre": anime["genre"].iloc[anime_indices].values}
    dataframe = pd.DataFrame(data=rec_dic)
    dataframe.set_index("No", inplace=True)

    print(f"Recommendations for {matched_title} viewers:\n")
    print(dataframe)
    return dataframe
