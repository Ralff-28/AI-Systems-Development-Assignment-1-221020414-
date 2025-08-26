# AI-Systems-Development-Assignment-1-221020414-
Assignment-1
# =========================
# 1. Install Dependencies
# =========================
!pip install sentence-transformers pandas scikit-learn

# =========================
# 2. Import Libraries
# =========================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 3. Load Dataset
# =========================
# Option 1: Upload movies.csv from your local system
from google.colab import files
uploaded = files.upload()

movies = pd.read_csv("/content/movies.csv")
print("Dataset Loaded. Shape:", movies.shape)
movies.head()

# =========================
# 4. Load Model and Encode Plots
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding movie plots... this may take a moment.")
embeddings = model.encode(movies["plot"].tolist(), convert_to_tensor=True)
print("Embeddings created with shape:", embeddings.shape)

# =========================
# 5. Define Search Function
# =========================
def search_movies(query: str, top_n: int = 5):
    """
    Search for movies most relevant to the query based on semantic similarity.
    Returns a DataFrame with top_n results and similarity scores.
    """
    query_embedding = model.encode([query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = movies.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]
    return results[["title", "plot", "similarity"]]

# =========================
# 6. Test the Function
# =========================
search_movies("spy thriller in Paris", top_n=3)
