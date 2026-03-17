from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import json
import uuid
import os
from typing import Any, Generator, List
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


def find_similar_papers(query_vector: List[float], top_k: int = 5) -> List[Any]:
    # Find the top-5 most similar entries with the .query_points() method
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


if __name__ == "__main__":
    # The input query
    input_query = "the attention mechanism in deep learning"
    
    # Generate embedding for the query
    query_embedding = get_embedding(input_query)
    
    # Find 5 most similar results
    similar_results = find_similar_papers(query_embedding, 5)
    
    # Extract the paper IDs and append them to a new list
    similar_paper_ids = [point.payload["id"] for point in similar_results]
    
    # Print the result
    print(similar_paper_ids)