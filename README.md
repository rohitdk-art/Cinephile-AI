# Cinephile-AI
Context-Aware Movie Discovery Engine | MDG Season of Code 2025

# Week 1: Core Logic & Vector Engine

## Overview <br>
Week 1 focuses on building the "Brain" of the recommendation system. Unlike traditional keyword search (which only matches exact words), this engine uses Semantic Search. It converts movie plots and metadata into high-dimensional vectors, allowing users to search by "vibe" or abstract description (e.g., "A dystopian future where machines take over" matches The Matrix).



## Tech Stack <br>
Language: Python <br>
Data Manipulation: Pandas, NumPy <br>
Machine Learning: Scikit-Learn (Cosine Similarity) <br>
NLP & Embeddings: Sentence-Transformers (all-MiniLM-L6-v2) <br>
Serialization: Pickle

## Project Structure
Cinephile-AI\  
├── tmdb_5000_movies.csv       (Raw Dataset) <br>
├── tmdb_5000_credits.csv      (Raw Dataset) <br> 
├── clean_data.py              (Script 1: ETL & Data Cleaning) <br>
├── vectorize.py               (Script 2: Embedding Generation) <br>
├── search.py                  (Script 3: CLI Search Engine) <br>
├── cleaned_movies_data.csv    (Intermediate Processed Data) <br>
├── movie_data.pkl             (Serialized DataFrame) <br>
└── movie_vectors.pkl         # Serialized Vector Matrix <br>
└── movie_vectors.pkl         # Serialized Vector Matrix <br>

## Work Progress:  
(Day 1 - 15/12/25) Setup & Installation: Downloaded and explored TMDB 5000 Movie Datasets,Searched about & installed libraries & dependencies, Set up the Development Environment    

(Day 2) Data Cleaning & ETL: Wrote clean_data.py to merge the Movies and Credits datasets. Implemented JSON parsing logic to extract readable names from the raw cast and genres stringified lists. Handled missing data by dropping null values to ensure data integrity.  

(Day 3) Data Cleaning & ETL: Created the a keyword "tags" column by combining Overview, Genres, Keywords, Cast, and Crew into a single text column. Removed spaces from proper names (e.g., "James Cameron" to "JamesCameron") to create unique semantic tokens. 

(Day 4) AI Model Integration: Integrated the HuggingFace sentence-transformers library. Selected and downloaded the all-MiniLM-L6-v2 model for its balance of speed and accuracy. Wrote the logic to convert the pre-processed text tags into 384-dimensional vectors. 

(Day 5) Serialization & Storage: Implemented pickle to serialize the DataFrame and Vector Matrix into binary files (.pkl). Validated that loading data from binary files is significantly faster (<1s) than reprocessing raw CSVs. Tried to Understand the neural network & all-MiniLM-L6-v2 model.

(Day 6) Search Engine Logic: Developed search.py to implement the Cosine Similarity algorithm. Built the function to convert user queries into vectors and compare them against the movie database.   

(Day 7) Search Engine Logic: Implemented the sorting logic to retrieve and display the top 5 most relevant results. Understood how search.py works and looked for some modifications. Did some Manual Testing to see how it works with abstract, detailed queries etc.  

(Day 8) Testing & Documentation: Finetuned & finalized the code structure (added comments for better understanding) and wrote this README.md documentation.  


## Results:  
The system identifies movies based on abstract descriptions.
VS Code Terminal at the end of Day 8 (Search Engine Finally Working!! )
<img width="1199" height="1042" alt="Screenshot 2025-12-23 135923" src="https://github.com/user-attachments/assets/d577e73f-ab0b-4e82-aef3-1c9eaccce943" />



