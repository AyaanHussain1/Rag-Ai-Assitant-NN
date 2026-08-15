import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
import time 
from sklearn.metrics.pairwise import cosine_similarity
def create_embedding(text_list):
    try:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text_list},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not get embeddings from Ollama. Make sure Ollama is running "
            "and the bge-m3 model is installed (`ollama pull bge-m3`)."
        ) from exc

    if "embeddings" not in data:
        raise RuntimeError(f"Ollama embedding request failed: {data.get('error', data)}")
    return data["embeddings"]
# print(create_embedding("Ayaan is good"))
jsons = os.listdir("new_jsons")
chunk_list = [] 
chunk_id = 0
# a = time.time()
for json_file in jsons:

    with open(f"new_jsons/{json_file}") as f:
        content = json.load(f)
    embeddings = create_embedding([c["text"] for c in content["chunks"]])
    print(f"Create embeddings for {json_file}")
    for i,chunk in enumerate(content["chunks"]):

        chunk["chunk_id"] = chunk_id
        chunk_id +=1
        
        chunk["embedding"] = embeddings[i]

        chunk_list.append(chunk)
        print(chunk)
# b = time.time()
# print(chunk_list)
# print(f"Total Time: {a-b}")
df = pd.DataFrame.from_records(chunk_list)
joblib.dump(df,"embeddings.joblib")
# print(df)
