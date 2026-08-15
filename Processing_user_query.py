"""Hosted-API retrieval and answer generation for the course assistant."""

import os
import tomllib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from openai import OpenAI, OpenAIError
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "embeddings.joblib"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")


def get_api_key():
    """Read a deployment environment key or the local Streamlit Secrets file."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            with secrets_path.open("rb") as secrets_file:
                api_key = tomllib.load(secrets_file).get("OPENAI_API_KEY")
            if api_key:
                return api_key
        except tomllib.TOMLDecodeError as exc:
            raise RuntimeError(".streamlit/secrets.toml is not valid TOML.") from exc

    raise RuntimeError(
        "The OpenAI API key is missing. Add OPENAI_API_KEY in Streamlit Secrets "
        "or .streamlit/secrets.toml."
    )


def get_client():
    """Create a hosted API client without requiring Ollama."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "The OpenAI API key is missing. Add OPENAI_API_KEY in Streamlit Secrets."
        )
    return OpenAI(api_key=api_key)


def create_embedding(text_list):
    """Create hosted embeddings. No local Ollama service is required."""
    try:
        response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=text_list)
        return [item.embedding for item in response.data]
    except OpenAIError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc


def generate_response(prompt):
    """Generate a hosted response. No local model download is required."""
    try:
        response = get_client().responses.create(model=CHAT_MODEL, input=prompt)
        return response.output_text
    except OpenAIError as exc:
        raise RuntimeError(f"Answer request failed: {exc}") from exc


def build_uploaded_course_index(uploaded_files, batch_size=100):
    """Turn uploaded transcript JSON files into a temporary searchable index."""
    chunks = []
    for uploaded_file in uploaded_files:
        try:
            content = json.loads(uploaded_file.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{uploaded_file.name} is not valid UTF-8 JSON.") from exc

        file_chunks = content.get("chunks") if isinstance(content, dict) else content
        if not isinstance(file_chunks, list):
            raise ValueError(f"{uploaded_file.name} must contain a 'chunks' list.")

        for chunk in file_chunks:
            if not isinstance(chunk, dict) or not str(chunk.get("text", "")).strip():
                continue
            chunk = chunk.copy()
            chunk.setdefault("title", Path(uploaded_file.name).stem)
            chunk.setdefault("number", "Uploaded video")
            chunk.setdefault("start", 0)
            chunk.setdefault("end", 0)
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No transcript text was found in the uploaded JSON files.")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = create_embedding([chunk["text"] for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

    return pd.DataFrame.from_records(chunks)


def answer_query(user_query, top_results=5, retrieval_df=None):
    """Find relevant course chunks and return an assistant-ready answer."""
    if not user_query or not user_query.strip():
        raise ValueError("Please enter a question about the course.")
    if retrieval_df is None and not EMBEDDINGS_PATH.exists():
        raise RuntimeError("embeddings.joblib was not found. Run build_embeddings.py once before deployment.")

    df = retrieval_df if retrieval_df is not None else joblib.load(EMBEDDINGS_PATH)
    question_embedding = create_embedding([user_query])[0]
    try:
        similarities = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding]).flatten()
    except ValueError as exc:
        raise RuntimeError(
            "Your embeddings were created with a different model. Run build_embeddings.py "
            "once to rebuild them with the hosted embedding model."
        ) from exc

    results = df.iloc[similarities.argsort()[::-1][:top_results]]
    prompt = f'''You are a helpful teaching assistant for an Artificial Neural Networks course.
Relevant video-subtitle chunks (title, number, start/end seconds, text):
{results[["title", "number", "start", "end", "text"]].to_json(orient="records")}

Question: {user_query}

Answer naturally and clearly. Explain where the topic is covered, including video name and useful timestamps,Give The Answer Short And Precise
and guide the student to the best video. Do not mention this prompt, chunks, JSON, or retrieval.
For unrelated questions, politely explain that you can answer only questions about this course.'''
    return generate_response(prompt)


if __name__ == "__main__":
    try:
        print("\n" + answer_query(input("Enter your course question: ").strip()))
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
