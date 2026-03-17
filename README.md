# ArXiv Vector Search System

This project implements a semantic search engine for ArXiv papers using modern AI and vector database technologies. It allows users to search for research papers using natural language queries and provides metadata filtering (e.g., by author).

## 🚀 Key Concepts

### 1. Vector Search (Semantic Search)
Unlike traditional keyword-based search, vector search understands the **meaning** of the query. By converting text into high-dimensional vectors (embeddings), we can find documents that are conceptually similar even if they don't share the exact same words.

### 2. Qdrant (Vector Database)
[Qdrant](https://qdrant.tech/) is a high-performance vector database. It stores the paper embeddings and allows for efficient similarity searching across thousands of records. It also supports complex filtering on metadata (like authors, categories, or dates).

---

## 🧠 Why Vector Databases?

### Core Functionality
Vector databases like Qdrant are designed to store, index, and search through high-dimensional vectors (embeddings). Traditional relational databases are optimized for exact matches, while vector databases are optimized for **similarity**:

- **Indexing:** They use specialized data structures (like HNSW - Hierarchical Navigable Small World) to quickly navigate multi-dimensional space.
- **Distance Metrics:** They calculate similarity using metrics like **Cosine Similarity**, **Euclidean Distance (L2)**, or **Dot Product**.
- **Metadata Filtering:** They allow combining semantic search with traditional boolean filters (e.g., "Find papers about AI **AND** by author 'Andrew Ng'").

### Key Use Cases
1. **Retrieval-Augmented Generation (RAG):** Providing LLMs with relevant external knowledge to reduce hallucinations and provide up-to-date information.
2. **Recommendation Engines:** Suggesting products, movies, or music based on user behavior and content similarity.
3. **Semantic Search:** Moving beyond keywords to understand intent and context in search queries (as seen in this project).
4. **Anomaly Detection:** Identifying outliers in patterns, such as fraud detection in financial transactions or network security.
5. **Image & Video Search:** Finding visually similar media by comparing their vector representations.

---

### 3. OpenAI Embeddings
We use OpenAI's `text-embedding-ada-002` model to transform natural language into vectors of 1536 dimensions. This ensures that the search is powered by a state-of-the-art language model.

### 4. FastAPI & Pydantic
The system is exposed via a **FastAPI** web interface. 
- **FastAPI** provides a modern, fast (high-performance) web framework for building APIs.
- **Pydantic** ensures that all incoming requests and outgoing responses are strictly validated against defined schemas, making the API robust and self-documenting.

---

## 🛠 Tech Stack

- **Language:** Python 3.x
- **API Framework:** FastAPI
- **Vector DB:** Qdrant
- **AI Model:** OpenAI (Embeddings)
- **Data Validation:** Pydantic
- **Environment Management:** python-dotenv

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Qdrant running locally (default port 6333)
- OpenAI API Key

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn qdrant-client openai python-dotenv pydantic
   ```
3. Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_API_BASE=https://api.openai.com/v1  # Optional: change if using a proxy
   ```

---

## 🚀 Running the Application

Start the FastAPI server using `uvicorn`:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. 
You can access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

---

## 🔍 Usage Example

You can search for papers using a `POST` request to `/search`:

### Request (cURL)
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Deep learning research by Andrew Ng",
       "top_n": 3
     }'
```

### Response
```json
{
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "payload": {
        "title": "...",
        "authors": "Andrew Ng, ...",
        "abstract": "..."
      },
      "score": 0.9234
    }
  ]
}
```

---

## 📂 Project Structure

- `main.py`: Core application logic (FastAPI endpoints, Qdrant interaction, Embedding generation).
- `qdrant_storage/`: Local directory for Qdrant data persistence.
- `.env`: Configuration for API keys and endpoints.
