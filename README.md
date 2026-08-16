# Course Companion — AI Video Course Assistant

A Streamlit app that helps learners find and understand content inside course videos. Upload a course video and it's **auto-transcribed into a searchable index**, or upload transcript JSON files directly. Ask questions in plain English and the assistant points you to the right video with the exact timestamps.

## Live App
```
https://rag-ai-bot-nn.streamlit.app/
```

## Features

- **Ask questions about any course** — get natural-language answers that cite the exact video and timestamps where the topic is covered.
- **Auto-transcription of videos** — upload an `.mp4`, `.mp3`, `.mkv`, etc. and it is automatically converted to a JSON transcript and embedded (powered by local Whisper). No manual "convert to JSON" step needed.
- **JSON transcript upload** — for users who already have transcript JSON files.
- **Semantic search** — embeddings powered by OpenAI `text-embedding-3-small` and cosine similarity for accurate chunk retrieval.
- **Dark, clean UI** — simple chat-style interface built with Streamlit.

## How It Works

1. **Upload** a course video or transcript JSON file in the app.
2. **Auto-transcribe** — Whisper converts the video into time-stamped text segments and groups them into chunks.
3. **Embed** — each chunk is embedded using the OpenAI embedding API.
4. **Ask & find** — your question is embedded and matched against all chunks via cosine similarity; the top matches are fed to a chat model that produces the answer with references.

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit UI |
| `Processing_user_query.py` | Embedding, retrieval, answer generation, and auto-transcription logic |
| `build_embeddings.py` | Build the static hosted `embeddings.joblib` index from `new_jsons/` |
| `create_chunks.py` / `merge_chunks.py` / `read_chunks.py` | Offline pipeline to build a persistent course index from a folder of videos |
| `video_download_script.py` | Download course videos from YouTube (`yt-dlp`) |
| `convet_mp4-mp3.py` | Convert videos to audio (`ffmpeg`) |
| `embeddings.joblib` | Pre-built embeddings used by the default course |
| `requirements.txt` | Python dependencies |

## Installation

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- For **video auto-transcription**: `ffmpeg` on your PATH and the `openai-whisper` package (installed automatically — see below)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/course-companion.git
cd course-companion

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
#    Either set the OPENAI_API_KEY environment variable, or create .streamlit/secrets.toml:
#    OPENAI_API_KEY = "your key here"
```

### First run

```bash
# Optional: build the static index for the bundled course before the first run
python build_embeddings.py

# Launch the app
streamlit run app.py
```

Open the printed `http://localhost:8501` URL in your browser.

## Usage

1. Click **"＋ Add video"** and upload a course video (mp4, mp3, mkv, ...) or transcript JSON file(s).
2. Click **"Prepare uploaded videos"** — videos are transcribed and embedded in the background.
3. Type a question, e.g. "Where is backpropagation explained?", and press **Ask course**.

### Upload a YouTube course instead

If you prefer to build an offline course index from YouTube videos:

```bash
python video_download_script.py   # paste a playlist/video URL -> videos/ folder
python convet_mp4-mp3.py          # convert to audio -> audios/ folder
python create_chunks.py           # transcribe to jsons/ with Whisper
python merge_chunks.py            # group segments -> new_jsons/
python build_embeddings.py        # embed chunks -> embeddings.joblib
```

## Configuration

Customize behavior with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Chat model used for answers |
| `TRANSCRIPTION_MODEL` | `base` | Whisper model for video auto-transcription (`base` is fast, `large-v2` is most accurate) |
| `TRANSCRIPTION_LANGUAGE` | `auto` | Language hint for Whisper (e.g. `hi`), or auto-detect |
| `TRANSCRIPTION_TASK` | `transcribe` | `transcribe` (same language) or `translate` (English) |

## Acknowledgements

- [Whisper](https://github.com/openai/whisper) — speech-to-text transcription
- [Ollama](https://ollama.com/) — optional local embeddings (used by some helper scripts)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — video downloading
- Built with [Streamlit](https://streamlit.io/)
