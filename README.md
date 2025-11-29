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
- Key Features
- High-Level Architecture
- Repository Structure
- Data & Index Files
- Environment Requirements
- Local Setup & Run
- Docker Workflow
- Retrieval & Fusion Pipeline
- Dashboards & Use Cases
- Troubleshooting
- Future Improvements

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
│ │ ├─ analytics.py # Aggregations, charts data prep
│ │ ├─ indexing.py # Transcript ingestion, chunking helpers
│ │ ├─ search.py, video.py # YouTube-specific search logic
│ │ └─ transcription.py # Speech-to-text utilities
│ │
│ ├─ data/
│ │ ├─ lectures/, transcripts/ # YouTube metadata & embeddings
│ │ ├─ uploaded_files/.gitkeep # Placeholder for uploaded slides
│ │ ├─ course_materials.json, metadata.json, segments_metadata.json
│ │ └─ course.index # Index representation for transcripts
│ │
│ └─ frontend/
│ ├─ tabs/student.py # Student tab UI (YouTube context)
│ ├─ tabs/professor.py # Professor tab UI (YouTube analytics)
│ ├─ components.py, styles.py, types.py
│ └─ __init__.py # Streamlit entry helpers
│
├─ SmartTA_RAG/ #Course slide System Data
│ ├─ data/
│ │ ├─ raw/ # Original PDFs/slides
│ │ └─ index/ # See section “Data & Index Files”
│ ├─ rag_engine1.py # Legacy RAG engine notebook/runner
│ └─ All_Chapters_Rag_ipynb # notebook
│ └─ .env # Contains OPENAI_API_KEY
│
├─ smartta_unified/ # Combining Slides and Youtube System together
│ ├─ rag/ #Course slide System Functionalities
│ │ ├─ embeddings.py, search.py, engine.py # Core RAG logic, hybrid scoring
│ │ └─ __init__.py
│ ├─ chat.py # Student Assistant layout & logic
│ ├─ dashboard.py # Slides Instructor dashboard
│ ├─ youtube.py # YouTube tab bridging to SmartTA_Extensions
│ ├─ feedback.py # Course feedback tab
│ ├─ helpers.py, style.py # Shared theming, session utilities
│ └─ main.py # Streamlit entrypoint
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

🧩 Note for Professor:
• After running the Docker image, the YouTube system index takes about 1–2 minutes to load.
• For the RAG module, after entering the first query, please allow 5–6 minutes for the CLIP ViT model, text transformer and embedding models to download and after that each query will take around 5-10 second depends if text or image.
• On a 4G connection, this initial setup used approximately 3 GB of bandwidth and took around 5-6 minutes.
• Once loaded, all components run smoothly and remain cached inside the container for subsequent queries as shown in short demo docker video.
• Our Docker image size is 6.68 GB.

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
