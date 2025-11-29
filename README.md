# SmartTA – Multimodal RAG Teaching Assistant

SmartTA is a Streamlit application that powers a multimodal Retrieval-Augmented Generation (RAG) assistant plus 2 instructor dashboards. The system ingests lecture PDFs, slide images, and YouTube transcripts, builds hybrid indexes (text + CLIP(text and image) + BM25), and exposes:

**Student Assistant:** students ask questions via text or slide snapshots and receive grounded, citation-rich answers.  
**Slides Instructor Dashboard:** analytics on repeated questions, referenced slides, and retrieval explainability.  
**YouTube Instructor Dashboard:** transcript ingestion, query analytics, and video snippet browsing.  
**Course Feedback Dashboard:** aggregated satisfaction metrics and detailed response tables.

Every answer is explainable: retrieval diagnostics surface the exact slides and scores that fed the LLM, so instructors can trust (and audit) responses.

---

## Table of Contents

- [Demo Video](#demo-video)
- [Key Features](#key-features)
- [Credentials](#credentials)
- [High-Level Architecture](#high-level-architecture)
- [Repository Structure](#repository-structure)
- [Data & Index Files](#data--index-files)
- [Environment Requirements](#environment-requirements)
- [Local Setup & Run](#local-setup--run)
- [Docker Workflow](#docker-workflow)
- [Retrieval & Fusion Pipeline](#retrieval--fusion-pipeline)
- [Dashboards & Use Cases](#dashboards--use-cases)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Demo Video

🎥 **SmartTA Demonstration Videos**

- [Full Feature Demo + Docker Verification](https://mailaub-my.sharepoint.com/:f:/g/personal/jah30_mail_aub_edu/IgD_PSykHDqVQKog668blNdZAb-ocfJycuGhO2i9rkMJcx8?e=Jq57Jl)

This link includes:

1. A **complete walkthrough** showing all SmartTA features (Student Assistant, Instructor Dashboards, YouTube System and Feedback).
2. A **short verification clip** confirming that the **Docker environment is fully functional** and the app runs successfully inside the container.

For the best viewing experience, click the ⚙️ Settings icon (bottom-right corner) of the video player and set the quality to 1080p.

---

## Key Features

- **Multimodal RAG:** Combines OpenAI text embeddings, BM25 lexical scoring, and CLIP image embeddings.
- **Slide-aware Explainability:** Every answer shows referenced PDFs, page numbers, and retrieval scores.
- **Instructor Analytics:** Repeated question charts, referenced-slide histograms, word clouds, and query logs.
- **YouTube Integration:** Transcript ingestion, search, and analytics share the same question log as the main assistant.
- **Feedback Insights:** Streamlit dashboards surface satisfaction, engagement, and the latest responses.
- **Deployability:** Runs locally via `streamlit run` or in Docker using `python:3.11-slim`.

---

- **Deployability:** Runs locally via `streamlit run` or in Docker using `python:3.11-slim`.

---

## Credentials

Use these credentials to access the instructor dashboards:

- **Username:** `eece`
- **Password:** `690`

---

## High-Level Architecture

- **Ingestion:** PDFs and slides are chunked into text chunks and overlap text, text are encoded using text-embedding-3-large and clip text encoder and images encoded with CLIP VIT. YouTube transcripts are segmented with timestamps.
- **Hybrid Retrieval:** `hybrid_search_v3` merges text embeddings (FAISS), BM25, CLIP text→image, image→image, and page-level matching.
- **Answer Generation:** Top contexts feed GPT‑4o Mini with instructions to cite slides `[pdf:page]`.
- **Dashboards:** Streamlit tabs display analytics for slides, YouTube, and course feedback, all reading from shared data logs.

---

## Repository Structure

```bash
SmartTA/
│
├─ data/
│ ├─ chat_logs.ndjson # Student Assistant conversation history (NDJSON)
│ ├─ course_feedback.csv # Course feedback submissions
│ └─ youtube_queries_log.json # YouTube search queries
│
├─ SmartTA_Extensions/ # Youtube System
│ ├─ backend/
│ │ ├─ analytics.py # Query logging to youtube_queries_log.json, Plotly chart generation (lecture frequency, query timeline, engagement metrics, learning paths), 60s cached data loading
│ │ ├─ indexing.py # FAISS index management with SentenceTransformer embeddings (all-MiniLM-L6-v2), IVF-PQ optimization for >10k segments, incremental indexing, nprobe tuning, auto-reindexing on startup, progress tracking with ETA
│ │ ├─ resources.py # Singleton pattern for models (sentence-transformer all-MiniLM-L6-v2, cross-encoder ms-marco-MiniLM-L-6-v2) and FAISS index to prevent redundant loading, reduces memory footprint, optional optimization layer
│ │ ├─ search.py # Levenshtein typo correction (edit distance, corpus vocabulary matching), hybrid search combining FAISS semantic + BM25 lexical + phrase matching + token overlap, query expansion with synonyms, deduplication, question-type detection (factual/conceptual/procedural), adaptive weighting, optional cross-encoder reranking
│ │ ├─ transcription.py # YouTube audio download via yt-dlp with MP3 extraction, Whisper transcription (tiny model, CPU-forced, timestamp extraction), exponential backoff retry with MAX_RETRIES, incremental processing for new lectures, metadata rebuilding, segment storage in segments_metadata.json
│ │ └─ video.py # FFmpeg wrapper for extracting video/audio duration (regex parsing of ffmpeg output), used during transcription validation and lecture metadata building
│ │
│ ├─ data/
│ │ ├─ lectures/, transcripts/ # YouTube lecture metadata JSON files (individual per lecture), Whisper text transcripts
│ │ ├─ uploaded_files/.gitkeep # Placeholder directory for course materials (PDFs, docs)
│ │ ├─ course_materials.json, metadata.json, segments_metadata.json # Course materials index, FAISS metadata, timestamped transcript segments with embeddings
│ │ └─ course.index # FAISS vector index for semantic search over YouTube transcripts
│ │
│ ├─ frontend/
│ │ ├─ tabs/
│ │ │ ├─ student.py # Search interface with query input, optional lecture-only filter, typo correction (Levenshtein + corpus vocab), hybrid semantic+BM25 scoring in quality mode by default, phrase & token overlap bonuses, duplicate filtering, result caching, timestamp jump (?start autoplay), AI follow-up form using top segments with GPT model; optional precision rerank used when cross-encoder is available
│ │ │ └─ professor.py # Control panel (refresh, clear analytics, clean index, clear all), confused topics aggregation from query logs, YouTube lecture ingestion (category creation, URL validation, professor notes), lecture CRUD (add/edit/delete), course materials management (section grouping, ordering, uploads with notes & deletion), analytics (lecture frequency, timeline distributions, hourly patterns via Plotly or fallback)
│ │ ├─ components.py # Grouped results by lecture with category color badges, optional professor note display, score & keyword overlap badges, simple regex-based query term highlighting, timestamp jump button storing session state, course materials expander with secure download buttons
│ │ ├─ styles.py # Dark theme CSS with glassmorphism effects, gradient backgrounds, confetti/toast animations for user actions, skeleton loaders for async operations, mobile responsive breakpoints, custom gradient buttons with hover effects, scrollbar styling, card shadows and borders
│ │ └─ types.py # TypedDict definitions for type safety across modules (Segment with text/start/end/category, SearchResult with score/lecture/timestamp, MaterialFile with name/path/size/uploaded_at, MaterialSection with title/files, LectureInfo with id/title/url/category/duration, YouTubeData with lectures dict)
│ │
│ └─ utils/
│ ├─ config.py # Centralized configuration constants (DATA_DIR, MODEL_DIR paths, DEVICE=CPU/CUDA detection, INDEX_PATH, META_PATH, LOG_PATH with migration logic from extensions to project root, retry parameters MAX_RETRIES=3/RETRY_DELAY=2, search hyperparameters TOP_K=5/NPROBE=10, Whisper model size "tiny", embedding dimension EMBED_DIM=384)
│ ├─ helpers.py # YouTube lecture database utilities (load_yt_data, save_yt_data for individual JSON files in lectures/), YouTube ID extraction with 4 regex patterns (youtu.be, youtube.com/watch, youtube.com/embed, /v/ formats), process_with_progress wrapper for ETA display, safe JSON load/save with exception handling, get_all_lectures for metadata aggregation, file operation helpers for upload management
│ └─ check_imports.py # Diagnostic script testing critical dependencies (streamlit, faiss-cpu, sentence-transformers, openai-whisper, torch, yt-dlp, rank-bm25) with colored OK/FAIL output, import error messages for troubleshooting, can be run standalone for environment validation
│
├─ SmartTA_RAG/ #Course slide System Data
│ ├─ data/
│ │ ├─ raw/ # Original PDFs/slides
│ │ └─ index/ # See section “Data & Index Files”
│ ├─ rag_engine1.py # Legacy RAG engine notebook/runner
│ └─ All_chapters_Rag.ipynb # Jupyter notebook
│ └─ .env (create locally) # Contains OPENAI_API_KEY
│
├─ smartta_unified/ # Unified application combining slides RAG and YouTube systems
│ ├─ rag/ # Core RAG engine for slide/PDF search
│ │ ├─ embeddings.py # OpenAI text-embedding-3-large embeddings, CLIP ViT-L/14 image/text encoding, lazy model loading, L2 normalization for cosine similarity
│ │ ├─ search.py # Hybrid search combining OpenAI dense embeddings (FAISS) + BM25 lexical scoring (rank_bm25), CLIP image-to-image search, CLIP text-to-image cross-modal search, minmax score normalization, configurable fusion weights (0.6 dense + 0.4 BM25)
│ │ ├─ engine.py # High-level answer generation with GPT-4o-mini, adaptive signal fusion (text embeddings, BM25, CLIP text→image, CLIP image→image, page-level bonuses), visual focus detection for image queries, retrieval diagnostics with per-signal scoring, citation formatting [pdf:page]
│ │ ├─ indexes.py # FAISS index loading (text.faiss, images_*.faiss), chunk metadata (chunks.jsonl, chunk_meta.jsonl, image_meta.jsonl), BM25 index initialization, page-to-chunk mapping (page_to_chunk_ids.pkl), CLIP model tag generation for file naming
│ │ ├─ settings.py # Configuration constants (OpenAI API key, model IDs, index paths, device detection CPU/CUDA, epsilon for numerical stability)
│ │ └─ state.py # Global state management (FAISS indexes, embeddings, chunks/metadata lists, CLIP model singletons, BM25 index)
│ ├─ chat.py # Student Assistant UI (question input, image upload, conversation history, retrieval diagnostics accordion, answer streaming, feedback collection)
│ ├─ dashboard.py # Slides Instructor Dashboard (query analytics, repeated questions analysis, referenced slides histogram, confidence metrics, query logs export, Plotly visualizations)
│ ├─ youtube.py # YouTube system tab integration (bridges to SmartTA_Extensions, renders YouTube lecture search and professor dashboard)
│ ├─ feedback.py # Course feedback dashboard (satisfaction metrics aggregation, engagement analysis, response table with filters, CSV data reading from course_feedback.csv)
│ ├─ helpers.py # Shared utilities (session state management, PDF filtering, safe JSON operations)
│ ├─ style.py # CSS styling (dark theme, gradient backgrounds, card styling, mobile responsiveness)
│ └─ main.py # Streamlit entrypoint (mode switcher for Student/Instructor/YouTube/Feedback tabs, index initialization, session state setup, page config)
│
├─ requirements-docker.txt # Runtime deps for container build
├─ requirements.txt # Full dev dependency set
├─ dockerfile # Container recipe
└─ README.md # This document
```

---

## Data & Index Files

### SmartTA_RAG/data/index/

- `chunks.jsonl`: text chunks with IDs, source PDF names, page numbers.
- `chunk_meta.jsonl`: additional metadata (OCR, bounding boxes, confidence).
- `image_meta.jsonl`: CLIP image chunk metadata.
- `text.faiss` & `X_text.npy`: FAISS index + raw embeddings for text.
- `text_clip.faiss` & `X_text_clip.npy`: CLIP text embeddings/index.
- `images_openai_clip_vit_large_patch14_336.faiss` & `X_img_openai_clip_vit_large_patch14_336.npy`: CLIP image embeddings/index.
- `page_to_chunk_ids.pkl`: map from slide page to chunk IDs.

### SmartTA_Extensions/data/

- `metadata.json`, `segments_metadata.json`, `course_materials.json`, `course.index`: YouTube transcript metadata.
- `lectures/`, `transcripts/`: storage for ingested assets.
- `uploaded_files/.gitkeep`: ensures the upload directory exists.

---

## Environment Requirements

- Python 3.11
- OpenAI API key (`OPENAI_API_KEY`)
- System packages: ffmpeg, libgl1, build-essential
- Docker Desktop (for container builds)
- Internet access to call OpenAI and download packages

---

## Local Setup & Run

```bash
git clone https://github.com/HarbJohn/SmartTA.git
cd SmartTA
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create SmartTA_RAG/.env:

```bash
cd SmartTA_RAG
touch .env
echo 'OPENAI_API_KEY="s...A"' >> .env   # or open .env in your editor and add the key
```

Ensure indexes live in `SmartTA_RAG/data/index/` and the pdfs folder optionally remains intact `SmartTA_RAG/data/raw/`. Then:

Run this from the repository root (`cd SmartTA`):

```bash
streamlit run smartta_unified/main.py
```

Open [http://localhost:8501](http://localhost:8501) and use the mode switcher to navigate between Student Assistant, Slides dashboard, YouTube dashboards and Course Feedback.

---

## Docker Workflow

In the directory that contains Dockerfile in our case SmartTA, apply the following:

**1. Build**

```bash
cd SmartTA

docker build -t smartta .
```

**2. Run**

```bash

docker run --rm -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -e OPENAI_API_KEY=s...A \
  smartta

```

Replace `s...A` with your OpenAI API key.  
Open [http://localhost:8501](http://localhost:8501) while the container runs.

🧩 **Note for Professor:**

- After running the Docker image, the YouTube system index takes about 1–2 minutes to load.
- For the RAG module, after entering the first query, allow 5–6 minutes for the CLIP ViT model, text transformer, and embedding models to download. After that, each query takes about 5–10 seconds (text/image dependent).
- On a 4G connection, this initial setup used ~3 GB of bandwidth and took about 5–6 minutes.
- Once loaded, all components run smoothly and remain cached inside the container (as shown in the short Docker demo).
- The Docker image size is about 6.68 GB.

---

## Retrieval & Fusion Pipeline

**Chunking:** Each PDF page is split into text chunks and overlap text. Slide images are simultaneously encoded with CLIP ViT-L/14 (image encoder) so diagrams and handwritten notes become searchable.

### Signals

- **Text Embeddings:** OpenAI text-embedding-3-large vectors feed a FAISS index; cosine similarity gives dense-retrieval scores.
- **BM25 Lexical Scores:** `rank_bm25` finds exact-term matches to capture keywords or acronyms.
- **CLIP Text→Image:** The text query is projected into CLIP space to surface visual diagrams.
- **CLIP Image→Image:** Uploaded snapshots find visually similar slides.
- **CLIP Text→Text:** CLIP’s text encoder bridges between OCR’d text and captions.
- **Page-Level Bonuses:** If multiple signals agree on the same page, a bonus boosts that context.

### Fusion

We compute a weighted sum across signals for each chunk:

```
final_score[i] = w_text * text_score[i] + w_clip_t * text_clip_score[i] +
                 w_page_t * page_bonus_text[i] + w_page_i * page_bonus_img[i] +
                 w_img_c * img_to_chunk_score[i]
```

We then select the top N chunks (usually 5).

### LLM Answer

The selected contexts and the question are sent to GPT‑4o Mini with instructions to cite slides `[pdf:page]`.  
If an image was uploaded, we include its caption or OCR text.

### Diagnostics

Streamlit’s “Retrieval Diagnostics” accordion shows each signal (OpenAI text embedding, CLIP text→image, CLIP image→image, contexts) and retrieved pages with scores.

---

## Dashboards & Use Cases

**Student Assistant (`smartta_unified/chat.py`)**

- Accepts questions and slide uploads.
- Shows history, retrieval diagnostics, and feedback.

**Slides Instructor Dashboard (`smartta_unified/dashboard.py`)**

- Displays Q&A counts, confidence, repeated questions, and referenced slides.

**YouTube Instructor Dashboard (`smartta_unified/youtube.py` + SmartTA_Extensions)**

- Ingests transcripts, tracks queries, shows analytics.

**Course Feedback (`smartta_unified/feedback.py`)**

- Displays aggregated satisfaction and engagement metrics.

---

## Troubleshooting

| Problem              | Cause                                 | Fix                                  |
| -------------------- | ------------------------------------- | ------------------------------------ |
| Docker command fails | Folder renamed but `-v` path outdated | Use `$(pwd)` or update absolute path |
| venv scripts break   | Old path                              | Recreate virtual environment         |
| API key missing      | Not exported                          | Add `-e OPENAI_API_KEY=s...A`        |

---

## Future Improvements

- **Live Lecture Mode:** Enable real-time transcription and retrieval during ongoing lectures using Whisper streaming and low-latency RAG indexing.
- **Recommendation System:** Suggest related slides or video segments based on student queries.
- **Cloud Deployment:** Deploy SmartTA on AWS or Azure for scalability and secure multi-user access.
- **Cross-Platform Compatibility:** Make it usable on mobile phones, tablets, and other devices.
- **Voice Query Support:** Let students ask via audio, using Whisper for real-time transcription.
