import pandas as pd
from sentence_transformers import SentenceTransformer 
import pickle  
import gc

print("Starting vectorization process...")

# Load cleaned data and model
df=pd.read_csv('cleaned_movies_data.csv')
model=SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Generating vectors...")
vectors=model.encode(df['tags'].tolist(),show_progress_bar=True)
print("Vectors generated.")

# Save vectors and dataframe(for easy access later) using pickle
with open('movie_vectors.pkl','wb') as f:
    pickle.dump(vectors,f)

del model
del vectors
gc.collect()  # Forces Python to release the memory back to the OS
print("RAM cleared.")

with open('movie_data.pkl','wb') as f:
    pickle.dump(df,f)

print("Vectorization complete. Vectors and data saved.")


