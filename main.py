from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuration
MOVIE_DATA_PATH = 'movie_data.pkl'
VECTORS_PATH = 'movie_vectors.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

app = FastAPI(title="Cinephile AI Backend", version="1.0")

# Global State Loading
print("Initializing Server Artifacts...")

# 1. Load DataFrame
if os.path.exists(MOVIE_DATA_PATH):
    with open(MOVIE_DATA_PATH, 'rb') as f:
        df = pickle.load(f)
    print(f"Loaded Movie DB: {len(df)} records.")
else:
    print(f"Critical: {MOVIE_DATA_PATH} not found.")
    df = None

# 2. Load Pre-computed Vectors
if os.path.exists(VECTORS_PATH):
    with open(VECTORS_PATH, 'rb') as f:
        vectors = pickle.load(f)
    print("Loaded Vectors.")
else:
    print(f"Critical: {VECTORS_PATH} not found.")
    vectors = None

# 3. Load ML Model
print(f"Loading Transformer Model ({MODEL_NAME})...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("Model Ready.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

print("Server Ready.")


# Endpoints

@app.get("/")
def health_check():
# Health check endpoint to verify server status.
    return {
        "status": "active",
        "service": "Cinephile AI Backend",
        "version": "1.0",
        "movies_loaded": len(df) if df is not None else 0
    }

@app.get("/search")
def search_movies(q: str):
    """
    Semantic search endpoint.
    Takes a query 'q', converts it to a vector, and performs cosine similarity
    search against the movie database.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if model is None or vectors is None or df is None:
        raise HTTPException(status_code=500, detail="Server is not fully initialized.")

    # 1. Vectorize Query
    query_vector = model.encode([q])

    # 2. Calculate Similarity
    similarity_scores = cosine_similarity(query_vector, vectors).flatten()

    # 3. Get Top 5 Results
    top_indices = np.argsort(similarity_scores)[::-1][:5]

    results = []
    for idx in top_indices:
        movie = df.iloc[idx]
        score = similarity_scores[idx]

        # Data Cleaning & Formatting
        poster_path = movie.get('poster_path', None) if 'poster_path' in df.columns else None
        
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}" 
            if poster_path and str(poster_path) != 'nan' 
            else "https://via.placeholder.com/500x750?text=No+Image"
        )

        overview = movie['overview'] if str(movie['overview']) != 'nan' else "No description available."
        
        try:
            rating = float(movie['vote_average'])
        except (ValueError, KeyError):
            rating = 0.0

        release_date = str(movie['release_date'])
        year = release_date.split("-")[0] if release_date != 'nan' else "Unknown"

        results.append({
            "id": int(movie['movie_id']),
            "title": movie['title'],
            "score": round(float(score), 2),
            "poster_path": poster_url,
            "overview": overview,
            "rating": rating,
            "year": year
        })

    return {"query": q, "results": results}