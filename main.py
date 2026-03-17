"""
ArXiv Paper Vector Search System.

This module implements a vector search system for ArXiv papers using:
- FastAPI for the web interface.
- Qdrant for vector database storage and similarity search.
- OpenAI for generating text embeddings.
- Pydantic for data validation and schema definition.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, MatchText
import re
import json
import uuid
import os
from typing import Any, Generator, List, Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables (OPENAI_API_KEY, OPENAI_API_BASE)
load_dotenv()

# Initialize Qdrant Client - assumed to be running locally or in a container
client = QdrantClient(host="localhost", port=6333)

# Initialize OpenAI Client for embedding generation
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# Initialize FastAPI application
app = FastAPI(
    title="ArXiv Vector Search API",
    description="Search for ArXiv papers using semantic similarity and metadata filtering.",
    version="1.0.0"
)


class SearchRequest(BaseModel):
    """
    Model representing a search request sent by a client.

    Attributes:
        query (str): The search query string provided by the client.
        top_n (Optional[int]): The number of top search results to return.
            Defaults to 5 if not provided.
    """
    query: str
    top_n: Optional[int] = 5


class SearchResult(BaseModel):
    """
    Model representing an individual search result.

    Attributes:
        id (str): Unique identifier for the search result item.
        payload (Dict): A dictionary containing additional data or metadata about the item.
        score (float): The relevance score of the search result, typically used for ranking.
    """
    id: str
    payload: Dict
    score: float


class SearchResponse(BaseModel):
    """
    Model representing the response returned to a client after a search.

    Attributes:
        results (List[SearchResult]): A list containing the top matching search results.
    """
    results: List[SearchResult]


COLLECTION_NAME = 'arxiv_papers'
# Embedding size for 'text-embedding-ada-002' is 1536
vectors_config = {"size": 1536, "distance": "Cosine"}

# Ensure the Qdrant collection exists on startup
if not client.collection_exists(collection_name=f"{COLLECTION_NAME}"):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vectors_config["size"],
            distance=Distance.COSINE
        )
    )


def stream_json(file_path: str) -> Generator[Any, Any, None]:
    """
    Streams JSON objects from a file where each line is a valid JSON.
    
    Args:
        file_path: Path to the .jsonl file.
        
    Yields:
        Parsed JSON object for each line.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_vectors_to_qdrant(file_path: str, batch_size: int = 500) -> None:
    """
    Loads pre-computed embeddings and metadata from a file into Qdrant.
    
    Args:
        file_path: Path to the file containing paper data and embeddings.
        batch_size: Number of points to upsert in a single request.
    """
    batch = []
    total_records = 0

    for record in stream_json(file_path):
        embedding = record.get("embedding")
        if embedding is None:
            continue

        # Exclude embedding from payload to save space and avoid redundancy
        payload = {k: v for k, v in record.items() if k != "embedding"}

        # Use deterministic UUID based on paper ID for idempotency
        point = PointStruct(
            id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=record["id"])),
            vector=embedding,
            payload=payload,
        )
        batch.append(point)

        if len(batch) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_records += len(batch)
            print(f"Uploaded {total_records} records...")
            batch = []

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        total_records += len(batch)
        print(f"Uploaded {total_records} records...")


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Generates a vector embedding for the given text using OpenAI API.
    
    Args:
        text: Input string to embed.
        model: OpenAI model name.
        
    Returns:
        List of floats representing the embedding vector.
    """
    # Remove newlines for better embedding quality as per OpenAI recommendation
    text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def extract_author(query: str) -> Optional[str]:
    """
    Attempts to extract an author's name from a natural language query.
    Looks for the pattern 'by [Author Name]'.
    
    Args:
        query: User's search query.
        
    Returns:
        Extracted author name or None.
    """
    match = re.search(r"by\s+([A-Za-z\s\-]+)", query)
    if match:
        return match.group(1).strip()
    return None


def find_similar_papers(query_vector: List[float], author_name: Optional[str] = None, top_k: int = 3) -> List[Any]:
    """
    Searches Qdrant for papers similar to the query vector, with optional author filtering.
    
    Args:
        query_vector: Vector representation of the search query.
        author_name: Optional author name to filter by.
        top_k: Number of results to return.
        
    Returns:
        List of Qdrant ScoredPoint objects.
    """
    query_filter = None
    if author_name:
        # MatchText performs a full-text search on the 'authors' field
        query_filter = Filter(
            must=[
                FieldCondition(key="authors", match=MatchText(text=author_name))
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points
    return results


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Main endpoint for semantic search.
    Processes the query, extracts filters, generates embeddings, and queries Qdrant.
    """
    # 1. Extract potential filters (author name)
    author = extract_author(request.query)

    # 2. Convert query text to vector
    query_embedding = get_embedding(request.query)

    # 3. Perform vector search in Qdrant
    similar_results = find_similar_papers(query_embedding, author, request.top_n)

    # 4. Map results to the API response model
    results = [
        SearchResult(
            id=str(point.id),
            payload=point.payload,
            score=point.score
        )
        for point in similar_results
    ]

    return SearchResponse(results=results)


if __name__ == "__main__":
    # The input query
    input_query = "Mentions of point clouds by Tian-Xing Xu"
    
    # Extract the author name
    author = extract_author(input_query)
    
    # Generate embedding for the query
    query_embedding = get_embedding(input_query)
    
    # Find 3 most similar results with author filter if applicable
    similar_results = find_similar_papers(query_embedding, author, 3)
    
    # Extract the paper IDs and append them to a new list
    similar_paper_ids = [point.payload["id"] for point in similar_results]
    
    # Print the result
    print(similar_paper_ids)