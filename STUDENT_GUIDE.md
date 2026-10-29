# Student guide: Multimodal RAG from end to end

## 1. What this project teaches

A normal text RAG pipeline retrieves text. Real knowledge is also found in charts, diagrams, screenshots, and videos. This project makes those sources searchable without hiding the process behind a framework.

The central design is:

```text
PDF pages --------> text chunks -------------------+
Images -----------> factual captions + image path -+--> text embeddings
Video transcript -> timestamped chunks ------------+         |
                                                             v
                                                    one Chroma collection
                                                             |
question -> question embedding -> similarity search -> balanced evidence
                                                             |
                                        top retrieved image (when present)
                                                             |
                                                             v
                                                cited grounded answer
```

An image is represented by a caption for retrieval, but its path is retained. If that image is selected, the answer model can inspect the original pixels. This is **caption-index plus late vision**. A video is represented by its transcript first, with timestamps retained for citations.

## 2. Indexing time and question time

At **indexing time**, `build_index.py` reads the teaching assets, creates text records, embeds their `content`, and stores everything in Chroma. This work is repeated only when the corpus changes.

At **question time**, `ask.py` embeds one question, searches Chroma, chooses a small balanced evidence set, sends that evidence to the model, and prints a cited answer. If selected evidence includes an image, one image is attached for late visual inspection.

## 3. Common record design

All modalities use the same Python dictionary shape:

```python
{
    "id": "image-support_resolution_chart",
    "content": "A line chart showing ...",
    "metadata": {
        "source_type": "image",
        "source_name": "support_resolution_chart.png",
        "image_path": "C:\\...\\support_resolution_chart.png"
    }
}
```

- `id` uniquely identifies a Chroma record.
- `content` is the searchable text that receives an embedding.
- `metadata` preserves traceability. PDFs keep page numbers, images keep file paths, and videos keep timestamp ranges.

The original modalities differ, but a shared record shape lets one collection search all of them.

## 4. Run the project

From this project folder in PowerShell:

```powershell
.venv\Scripts\python.exe test_code.py
.venv\Scripts\python.exe reset_local_db.py
.venv\Scripts\python.exe build_index.py
.venv\Scripts\python.exe ask.py
.venv\Scripts\python.exe evaluate.py
```

`test_code.py` is offline. The last three scripts require the API key stored privately in `.env`.

## 5. Explanation of every Python file

### `build_index.py`

This is the indexing-time entry point. It finds the project folder, loads `.env`, asks `utils.py` to prepare all three modalities, creates OpenAI embeddings, and writes a persistent Chroma collection.

Constants:

- `PROJECT_FOLDER` is calculated from the script location, so the script works regardless of the terminal's current folder.
- `DATABASE_FOLDER` is the local directory where Chroma persists the index.

Function:

- `main()` loads environment variables, calls `build_multimodal_records()`, calls `rebuild_collection()`, and prints the total plus a count for each modality. The count is an easy verification that no source family was silently missed.

The final `if __name__ == "__main__"` guard runs `main()` only when this file is executed directly. It prevents accidental execution when another file imports it.

### `ask.py`

This is the question-time entry point. It does not rebuild the corpus. It reopens the persisted collection, retrieves evidence, and generates an answer.

Constants:

- `PROJECT_FOLDER` and `DATABASE_FOLDER` point to the project and Chroma storage.
- `DEMO_QUESTION` is used when the learner presses Enter. It deliberately needs a chart fact and a video-transcript recommendation.

Function:

- `main()` loads `.env`, accepts a question, creates the OpenAI client, opens Chroma, retrieves candidates, balances the evidence by modality, prints the evidence, and asks the model for a cited answer.

Reading this function top to bottom shows the complete query pipeline in a few lines.

### `evaluate.py`

This script tests whether retrieval brings back the expected modality. It does not ask the language model to generate answers, so it isolates retrieval behavior.

Constants:

- `QUESTIONS_PATH` points to the small evaluation dataset.
- The project and database constants work like those in the other scripts.

Functions:

- `load_evaluation_questions(path)` reads JSON and returns a list of evaluation cases. Each case has a question and one or more expected source types.
- `evaluate_question(case, collection, client)` retrieves eight candidates, collects the source types found, and returns `True` only if every expected type is present. It prints `PASS` or `FAIL` plus expected and found modalities.
- `main()` loads the API configuration, collection, and questions; evaluates every case; and prints the number passed.

This is a **modality retrieval** test. It does not prove that an answer is correct or faithful.

### `reset_local_db.py`

This small utility gives every classroom run a known starting point.

Function:

- `main()` checks only this project's `chroma_db` folder. If it exists, the function removes it and prints the exact path; otherwise, it reports that there is nothing to remove.

The path is fixed from the script location to avoid deleting an unrelated folder.

### `test_code.py`

This file uses Python's built-in `unittest` module. Its checks are deliberately offline, so setup and data logic can be verified without API cost.

Class:

- `MultimodalRagTests` groups the project tests.

Test methods:

- `test_split_text_keeps_overlap()` proves that neighboring PDF text chunks share the configured boundary words.
- `test_timestamp_format()` proves that numeric seconds become a readable `MM:SS` label.
- `test_all_three_modalities_load()` reads the real teaching data and checks that PDF, image, and video records are all produced with non-empty content.
- `test_balancing_caps_each_modality()` creates sample results and checks that selection keeps no more than the configured number from one modality.
- `test_source_labels_and_context()` checks that a PDF citation includes its filename and page, and that prompt context receives a source number.
- `test_image_data_url()` writes harmless temporary bytes and verifies that image encoding creates the data-URL prefix expected by the API.
- `test_chroma_collection_builds()` replaces API embeddings with two tiny fixed vectors, writes real records to a temporary Chroma collection, and confirms persistence works with the installed Chroma version.

### `utils.py`

This module contains shared pipeline functions. The entry-point scripts stay short by calling these functions.

Constant:

- `COLLECTION_NAME` gives every script the same Chroma collection name.

Functions, in pipeline order:

- `get_openai_client()` checks for `OPENAI_API_KEY` before creating the client. This produces a clear setup error instead of a confusing failure later.

- `split_text(text, chunk_size=120, overlap=20)` splits PDF text by words. It validates the settings, advances by `chunk_size - overlap`, and stops after the final chunk. Word-based chunking is easy to inspect; production systems may use token-aware or structure-aware splitting.

- `load_pdf_records(pdf_path)` uses `PdfReader` to extract each page. It chunks the page text and creates records with `source_type`, filename, and page number. Page-level metadata makes PDF citations possible.

- `load_image_records(captions_path, charts_folder)` reads prepared captions and structured facts. It joins those fields into searchable content and stores the absolute image path in metadata. It fails early when a caption points to a missing file.

- `seconds_to_timestamp(seconds)` converts a number such as `67` to `01:07` using integer division. It keeps video citations readable.

- `load_transcript_records(transcript_path, segments_per_chunk=2)` reads transcript segments in small groups. It combines their text and calculates the start and end of each group. Each record therefore preserves the spoken content and its timestamp range.

- `build_multimodal_records(project_folder)` is the ingestion coordinator. It calls the three loaders and returns one list containing PDF, image, and video records.

- `create_embeddings(texts, client)` sends a list of strings to the OpenAI embeddings endpoint. It reads the embedding model from `.env`, with `text-embedding-3-small` as the default, and returns vectors in the same order as the input strings.

- `open_chroma_client(database_folder)` creates a persistent local Chroma client. Persistence is why `ask.py` can reuse data created by `build_index.py`.

- `rebuild_collection(records, database_folder, client)` deletes only the named teaching collection if it already exists, recreates it with cosine distance, embeds all record content, and adds IDs, documents, metadata, and vectors together. Rebuilding avoids duplicate classroom data.

- `load_collection(database_folder)` reopens the existing collection. It replaces a low-level Chroma exception with a direct instruction to run `build_index.py` first.

- `retrieve(question, collection, client, top_k=8)` embeds the question, requests the nearest records from Chroma, and converts Chroma's column-oriented response into a learner-friendly list of dictionaries. Smaller cosine distance means greater similarity.

- `select_balanced_evidence(retrieved, limit=5, per_source_type=2)` walks through results in relevance order but caps how many items one modality can contribute. This is not a keyword router; it is a simple diversity step applied after unified retrieval.

- `source_label(metadata)` converts metadata into a readable label. PDF labels show a page, image labels show a filename, and video labels show a timestamp.

- `format_context(evidence)` numbers items as `[Source 1]`, `[Source 2]`, and so on. These exact numbers are placed in the model prompt so the final answer can cite them.

- `image_to_data_url(image_path)` reads image bytes, base64-encodes them, and adds the detected MIME type. This lets a local image be supplied as an `input_image` in an API request.

- `generate_answer(question, evidence, client)` builds a grounded prompt from numbered evidence. If selected evidence contains an image, it attaches only the first such image. Retrieval has already narrowed the corpus, so this is late vision rather than vision on every image. The function calls the Responses API and returns its text output.

- `print_evidence(evidence)` prints each label and vector distance before the answer. This is important in RAG: inspect retrieval before blaming generation.

## 6. Explanation of every supporting file

- `.env` contains the real API credential and model settings. It is private and should never be projected, shared, or committed.
- `.env.sample` documents the three required variable names with a fake key.
- `.gitignore` prevents the real key, virtual environment, generated database, and Python cache files from being committed.
- `requirements.txt` pins the four runtime libraries so trainer and students get consistent versions.
- `data/sample_report.pdf` is the PDF evidence used during ingestion.
- `data/charts/support_resolution_chart.png` contains exact values that make chart retrieval testable.
- `data/charts/multimodal_rag_architecture.png` visually summarizes the pipeline.
- `data/image_captions.json` contains factual, inspectable chart and diagram representations. Captions are prepared to keep the classroom run stable.
- `data/transcripts/sample_youtube_transcript.json` contains text plus start time and duration for every spoken segment. No live YouTube connection is required.
- `data/evaluation_questions.json` maps four test questions to expected source modalities.
- `chroma_db` is generated by `build_index.py`; it is not source data.
- `README.md` is the quick setup and run guide.
- `TRAINER_README.md` is the instructor's session plan.

## 7. Why the project is intentionally small

The practical excludes a web interface, notebooks, live video download, automatic speech recognition, live caption creation, framework chains, and keyword rules. Each is useful in a larger product, but each creates another failure point or hides the core learning objective.

The presentation mentions speech-to-text as a fallback when a transcript is missing. This project already has a reliable timestamped transcript, so it follows the recommended transcript-first path. Similarly, prepared captions make the retrieval representation visible, while late vision still demonstrates direct image understanding on retrieved evidence.

## 8. How to interpret results

A good run should show:

1. all three source families were indexed;
2. retrieved evidence includes source-aware labels;
3. the cross-modal question retrieves both chart and transcript evidence;
4. the answer uses `[Source n]` citations;
5. the evaluation reports whether expected modalities appeared.

If an answer is weak, inspect the printed evidence first. If the right fact was not retrieved, improve representation or retrieval. If the fact was retrieved but misstated, improve the answer prompt or model choice. This separation is one of the most important RAG debugging habits.

## 9. Safe exercises

1. Add one transcript segment and observe how timestamp chunks change.
2. Improve an image caption while keeping it factual, rebuild, and compare retrieval.
3. Change PDF chunk size and count the resulting records.
4. Add an evaluation question with the correct expected modality.
5. Temporarily lower `per_source_type` and observe evidence diversity.

Do not place secrets in Python files or print the `.env` file.

