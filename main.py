from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, MatchText
import re
import json
import uuid
import os
from typing import Any, Generator, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

COLLECTION_NAME = 'arxiv_papers'
vectors_config = {"size": 1536, "distance": "Cosine"}

if not client.collection_exists(collection_name=f"{COLLECTION_NAME}"):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vectors_config["size"],
            distance=Distance.COSINE
        )
    )


def stream_json(file_path) -> Generator[Any, Any, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_vectors_to_qdrant(file_path, batch_size=500) -> None:
    batch = []
    total_records = 0

    for record in stream_json(file_path):
        embedding = record.get("embedding")
        if embedding is None:
            continue

        payload = {k: v for k, v in record.items() if k != "embedding"}

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
    # Remove newlines from the query
    text = text.replace("\n", " ")
    # Make a call to OpenAI's embedding API
    response = openai_client.embeddings.create(input=[text], model=model)
    # Return the embedding as a list of floats
    return response.data[0].embedding


def extract_author(query: str) -> Optional[str]:
    # Regex to extract author's name after 'by'
    match = re.search(r"by\s+([A-Za-z\s\-]+)", query)
    if match:
        return match.group(1).strip()
    return None


def find_similar_papers(query_vector: List[float], author_name: Optional[str] = None, top_k: int = 3) -> List[Any]:
    query_filter = None
    if author_name:
        query_filter = Filter(
            must=[
                FieldCondition(key="authors", match=MatchText(text=author_name))
            ]
        )

    # Find the top-k most similar entries with the .query_points() method
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points
    return results


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