from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import json
import uuid
from typing import Any, Generator

client = QdrantClient(host="localhost", port=6333)

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


if __name__ == "__main__":
    DATA_PATH = "/Users/elinor/Downloads/ml-arxiv-embeddings.json"
    load_vectors_to_qdrant(DATA_PATH)