import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD THE SAVED DATA
# This is instant because we are just reading files, not calculating. 
with open('movie_data.pkl', 'rb') as f:
    df = pickle.load(f)
with open('movie_vectors.pkl', 'rb') as f:
    vectors = pickle.load(f)
print("Loading database...")

# 2. LOAD THE MODEL
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. DEFINE THE SEARCH FUNCTION
def search(query, top_k=5):

    
    # Step A: Vectorize the user's query
    query_vector = model.encode([query])

    # Step B: Calculate Similarity
    
    similarity_scores = cosine_similarity(query_vector, vectors)

    # Step C: Sort the results
    scores = similarity_scores.flatten()
    sorted_indices = np.argsort(scores)[::-1]

    # Step D: Print top K results
    print(f"\nTop matches for: '{query}'")
    print("-" * 30)
    
    for i in range(top_k):
        idx = sorted_indices[i] 
        score = scores[idx]     
        title = df.iloc[idx]['original_title'] 
        
        # Only show matches that are somewhat relevant (score > 0.1)
        if score > 0.1:
            print(f"{i+1}. {title} (Score: {score:.4f})")

# 4. INTERACTIVE LOOP/Search PROMPT
print("Welcome to the Movie Search App!")
while True:
    user_input = input("\nEnter a movie description (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    search(user_input)

