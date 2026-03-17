from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import json
import uuid
from typing import Any, Generator, List

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


def search_similar_papers(collection_name: str, paper_id: str, top_k: int = 5) -> List[str]:
    # Query the collection and retrieve the vector associated with the specified paper ID.
    scroll_filter = Filter(
        must=[
            FieldCondition(key="id", match=MatchValue(value=paper_id))
        ]
    )

    response, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        with_vectors=True,
    )

    # Check if the point with the given paper ID exists in the collection.
    if not response:
        raise Exception(f"Point with paper ID {paper_id} not found.")

    # Once you've retrieved the point, extract its embedding.
    query_vector = response[0].vector

    # Use the client to perform a search in the collection using the query vector.
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    # After obtaining the search results, extract the paper IDs from the results list.
    paper_ids = [point.payload["id"] for point in results]
    return paper_ids


if __name__ == "__main__":
    # DATA_PATH = "/Users/elinor/Downloads/ml-arxiv-embeddings.json"
    # load_vectors_to_qdrant(DATA_PATH)
    similar_paper_ids = search_similar_papers(COLLECTION_NAME, "1311.5068", 5)
    print(similar_paper_ids)