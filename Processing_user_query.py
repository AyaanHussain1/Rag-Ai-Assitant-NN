"""Hosted-API retrieval and answer generation for the course assistant."""

import functools
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

# Video / audio extensions accepted by the "Add video" flow. Any uploaded file
# with one of these extensions is transcribed to JSON chunks automatically.
VIDEO_EXTS = {"mp4", "mp3", "mkv", "avi", "mov", "webm", "mpeg", "mpg", "wav", "m4a", "ogg"}
DEFAULT_CHUNK_SIZE = 10


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
    """Turn uploaded video / transcript JSON files into a temporary searchable index.

    JSON transcript files are parsed directly. Video files (mp4, mp3, mkv, ...) are
    auto-transcribed into the same chunk JSON format using the local Whisper model,
    so the user never has to convert a video to JSON beforehand.
    """
    chunks = []
    for uploaded_file in uploaded_files:
        name = getattr(uploaded_file, "name", "video.mp4")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

        if ext in VIDEO_EXTS:
            # Video upload -> auto-convert to JSON chunks.
            file_chunks = transcribe_video(uploaded_file)
        else:
            # JSON transcript upload -> parse the existing chunk format.
            try:
                content = json.loads(uploaded_file.getvalue().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{name} is not a valid UTF-8 JSON transcript or a supported video file."
                ) from exc

            candidate = content.get("chunks") if isinstance(content, dict) else content
            if not isinstance(candidate, list):
                raise ValueError(f"{name} must contain a 'chunks' list.")
            file_chunks = candidate

        for chunk in file_chunks:
            if not isinstance(chunk, dict) or not str(chunk.get("text", "")).strip():
                continue
            chunk = chunk.copy()
            chunk.setdefault("title", Path(name).stem)
            chunk.setdefault("number", "Uploaded video")
            chunk.setdefault("start", 0)
            chunk.setdefault("end", 0)
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No transcript text was found in the uploaded videos or JSON files.")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = create_embedding([chunk["text"] for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

    return pd.DataFrame.from_records(chunks)


def transcribe_video(uploaded_file, group_size=None):
    """Transcribe an uploaded video/audio file into retrieval-ready chunks.

    Returns a list of dicts like
    [{"number": "Uploaded video", "title": <file stem>, "start": s, "end": e, "text": ...}]

    Tuning via environment variables:
      - TRANSCRIPTION_MODEL    Whisper model to use (default "base"; "large-v2" is the
                               most accurate but downloads GBs and is slow in-app).
      - TRANSCRIPTION_LANGUAGE Language hint for Whisper, or "auto" to auto-detect.
      - TRANSCRIPTION_TASK     "transcribe" (same language) or "translate" (to English).
    """
    if group_size is None:
        group_size = DEFAULT_CHUNK_SIZE

    name = getattr(uploaded_file, "name", "video.mp4")
    title = Path(name).stem
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else "mp4"

    model_name = os.getenv("TRANSCRIPTION_MODEL", "base")
    task = os.getenv("TRANSCRIPTION_TASK", "transcribe")
    language = (os.getenv("TRANSCRIPTION_LANGUAGE") or "auto").strip().lower()
    if language in ("", "auto", "none"):
        language = None

    import tempfile

    suffix = f".{ext}" if ext else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        model = _load_transcription_model(model_name)
        result = model.transcribe(
            tmp_path, language=language, task=task, word_timestamps=False
        )
    except Exception as exc:  # missing ffmpeg, bad audio, etc.
        raise RuntimeError(f"Could not auto-transcribe {name}: {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    chunks = []
    group = []

    def flush():
        if not group:
            return
        chunks.append(
            {
                "number": "Uploaded video",
                "title": title,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(str(seg.get("text", "")).strip() for seg in group),
            }
        )

    for segment in result.get("segments", []):
        group.append(segment)
        if len(group) >= group_size:
            flush()
            group = []
    flush()

    if not chunks:
        raise RuntimeError(f"No speech was transcribed from {name}.")
    return chunks


@functools.lru_cache(maxsize=1)
def _load_transcription_model(model_name):
    """Load the Whisper model once and reuse it across uploads in this session."""
    import whisper

    return whisper.load_model(model_name)


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
