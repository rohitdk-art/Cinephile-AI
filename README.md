# Cinephile-AI
Context-Aware Movie Discovery Engine | MDG Season of Code 2025



---

##  Setup & Installation

### **1. Backend Setup (Python)**

Prerequisites: Python 3.x installed.

1. **Clone the repository:**
```bash
git clone https://github.com/rohitdk-art/Cinephile-AI.git
cd Cinephile-AI/backend

```


2. **Install Dependencies:**
```bash
pip install fastapi uvicorn pandas numpy scikit-learn sentence-transformers

```


3. **Initialize Data (First Run Only):**
Run the data processing scripts to generate the vector database.
```bash
python clean_data.py
python vectorize.py
python search.py
```


4. **Start the Server:**
To run the server in development mode:
```bash
uvicorn main:app --reload

```


* **Local Access:** `http://127.0.0.1:8000`
* **Network Access (for Mobile):** `uvicorn main:app --host 0.0.0.0 --port 8000`



## Testing the API

You can test the backend logic directly from your browser without opening the mobile app.

### **1. Health Check**

Visit `http://127.0.0.1:8000/` to confirm the server is running and the database is loaded.

* **Expected Output:** `{"status": "active", "movies_loaded": 4799}`

### **2. Search Test**

Visit `http://127.0.0.1:8000/search?q=iron man` to see the JSON response.

* **Expected Output:** A list of movies including "Iron Man", "Iron Man 2", etc., with similarity scores.

