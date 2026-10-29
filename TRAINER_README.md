# Trainer guide: Multimodal RAG

## Quick start

Open PowerShell in `Multimodal RAG\multimodal_rag` and execute the scripts in this order:

```powershell
# 1. Confirm that the local data and code work without API calls.
.venv\Scripts\python.exe test_code.py

# 2. Remove an old classroom index, if one exists.
.venv\Scripts\python.exe reset_local_db.py

# 3. Create embeddings and build the multimodal Chroma index.
.venv\Scripts\python.exe build_index.py

# 4. Retrieve evidence and generate a cited answer.
.venv\Scripts\python.exe ask.py

# 5. Check whether the expected modalities are retrieved.
.venv\Scripts\python.exe evaluate.py
```

When `ask.py` prompts for a question, press Enter to use the default cross-modal question, or paste one of these:

```text
Which support resolution channel is largest in 2025?
```

```text
When should a vision model inspect an image?
```

```text
Why is caption-index plus late vision recommended?
```

```text
Which support channel is largest in 2025, and when does the training video recommend using a vision model?
```

The final question is the recommended main demonstration because it requires both chart and video-transcript evidence. Make sure `.env` exists, but never open or project it during class.

## Session outcome

Students should leave able to explain that multimodal RAG does not simply put raw PDFs, images, and videos into a text vector database. It first creates a searchable representation for each artifact, preserves a pointer to the original source, retrieves across those representations, and uses expensive visual reasoning only on selected evidence.

## Before class

From `Multimodal RAG\multimodal_rag`, run:

```powershell
.venv\Scripts\python.exe test_code.py
.venv\Scripts\python.exe reset_local_db.py
.venv\Scripts\python.exe build_index.py
.venv\Scripts\python.exe ask.py
.venv\Scripts\python.exe evaluate.py
```

Press Enter when `ask.py` asks for a question. Confirm that the evidence contains the support chart and a timestamped video chunk. Never project `.env`.

Before students arrive, run `reset_local_db.py` again if you want to demonstrate first-time ingestion live.

## Recommended 90-minute teaching flow

### 0-10 minutes: connect to the presentation

Use the early slides to ask what text-only RAG misses. Open the chart and architecture diagram from `data\charts`. Explain that important evidence may live in axes, labels, shapes, screenshots, or spoken words.

### 10-25 minutes: inspect the three representations

Open:

- `data\sample_report.pdf` for extracted page text;
- `data\image_captions.json` for factual image representations;
- `data\transcripts\sample_youtube_transcript.json` for timestamped speech.

Connect this to the slides on caption-index plus late vision and transcript-first video RAG. Stress that captions must retain title, axes, units, trends, values, and caveats.

### 25-40 minutes: walk through ingestion

Open `build_index.py`, then follow its calls into `utils.py`:

1. `build_multimodal_records()` creates a common record shape.
2. `create_embeddings()` embeds every text representation.
3. `rebuild_collection()` stores document text, embedding, and metadata together.

Run:

```powershell
.venv\Scripts\python.exe build_index.py
```

Ask: “What is actually stored?” Expected answer: searchable text representations, embeddings, and traceability metadata—not a playable video.

### 40-60 minutes: retrieval and late vision

Open `ask.py`. Show that one question embedding searches the unified collection. `select_balanced_evidence()` prevents a single modality from consuming the entire prompt. `generate_answer()` attaches at most the first selected image; this is late vision because retrieval happens before image reasoning.

Run `ask.py` and press Enter for:

```text
Which support channel is largest in 2025, and when does the training video recommend using a vision model?
```

The intended answer combines the chart’s Text RAG value with the transcript’s recommendation to retrieve cheaply and apply vision only to top evidence.

### 60-75 minutes: citations and failure analysis

Inspect the printed sources. Point out page numbers, image filenames, and video timestamp ranges. Ask students what could fail:

- weak or incorrect image captions;
- chunks that are too large or too small;
- missing metadata;
- one modality flooding the top results;
- unsupported claims in generation.

### 75-85 minutes: evaluation

Run:

```powershell
.venv\Scripts\python.exe evaluate.py
```

Explain that this small check measures modality retrieval, not answer quality. A production evaluation should also score faithfulness, citation correctness, visual interpretation, latency, and cost.

### 85-90 minutes: recap

Return to the architecture slide and let students narrate the whole path: artifact, text representation, metadata, embedding, unified index, retrieval, balanced evidence, late vision, cited answer.

## Presentation-to-code mapping

| Presentation idea | Code or data to show |
|---|---|
| Searchable representations | `load_pdf_records()`, `load_image_records()`, `load_transcript_records()` |
| Caption-index plus late vision | `data\image_captions.json`, then `generate_answer()` |
| Transcript-first video RAG | `sample_youtube_transcript.json`, then `load_transcript_records()` |
| Preserve source pointers | each record's `metadata` dictionary |
| Unified index | `rebuild_collection()` |
| Avoid modality flooding | `select_balanced_evidence()` |
| Grounded answer with citations | `format_context()` and `generate_answer()` |
| Modality-aware evaluation | `evaluate.py` and `evaluation_questions.json` |

## Suggested questions

Chart-only:

```text
Which support resolution channel is largest in 2025?
```

Video-only:

```text
When should a vision model inspect an image?
```

PDF concept:

```text
Why is caption-index plus late vision recommended?
```

Cross-modal:

```text
Which channel leads in 2025, and when should vision be used?
```

## Likely student questions

**Why not store raw images in Chroma?** This lesson uses text embeddings, so it stores a caption plus a path back to the image. Multimodal embedding systems are valid, but they add a second concept before students understand the basic architecture.

**Is a caption enough?** It is enough for first-pass retrieval only when it preserves the facts users may ask about. Late vision lets the model inspect the original selected image before answering.

**Does transcript RAG understand video frames?** No. It understands spoken content. Frame sampling is a later extension when visual events matter.

**Why one collection?** A unified collection is the simplest baseline. Metadata and balanced selection retain modality awareness. Larger systems may use separate indexes or rerankers.

**Why are captions prepared?** Live caption generation adds cost, latency, and nondeterminism during class. Prepared captions make the indexed representation visible and reproducible.

## If time is short

Run only `build_index.py` and `ask.py`. Do not skip the metadata discussion or the retrieved-evidence printout; those are the central lessons.

## Troubleshooting

- `OPENAI_API_KEY is missing`: ensure `.env` exists in this folder.
- “Run build_index.py”: the local collection has not been built or was reset.
- stale results: run `reset_local_db.py`, then `build_index.py`.
- import error: use `.venv\Scripts\python.exe`, not a global interpreter.
- weak retrieval: inspect the actual caption/transcript wording before tuning parameters.

