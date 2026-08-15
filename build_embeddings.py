"""Build hosted embeddings from new_jsons. Run once before deployment."""

import json
from pathlib import Path

import joblib
import pandas as pd

from Processing_user_query import BASE_DIR, EMBEDDINGS_PATH, create_embedding

CHUNK_SIZE = 100


def main():
    chunks = []
    chunk_id = 0
    for json_path in (BASE_DIR / "new_jsons").glob("*.json"):
        with json_path.open(encoding="utf-8") as file:
            content = json.load(file)
        for chunk in content["chunks"]:
            chunk["chunk_id"] = chunk_id
            chunks.append(chunk)
            chunk_id += 1

    for start in range(0, len(chunks), CHUNK_SIZE):
        batch = chunks[start : start + CHUNK_SIZE]
        embeddings = create_embedding([chunk["text"] for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
        print(f"Embedded {min(start + CHUNK_SIZE, len(chunks))}/{len(chunks)} chunks")

    joblib.dump(pd.DataFrame.from_records(chunks), EMBEDDINGS_PATH)
    print(f"Saved hosted embeddings to {EMBEDDINGS_PATH.name}")


if __name__ == "__main__":
    main()
