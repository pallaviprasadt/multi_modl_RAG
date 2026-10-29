"""Shared functions for the beginner-friendly multimodal RAG demos.

Every source becomes searchable text with metadata pointing to its original
artifact. Images use prepared factual captions; videos use transcript chunks.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

import chromadb
from openai import OpenAI
from pypdf import PdfReader


COLLECTION_NAME = "multimodal_rag_demo"


def get_openai_client() -> OpenAI:
    """Create an OpenAI client after checking that an API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to the .env file.")
    return OpenAI()


def split_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into overlapping word chunks to preserve boundary context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1.")
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_pdf_records(pdf_path: Path) -> list[dict]:
    """Extract a PDF and return searchable records with page metadata."""
    records = []
    for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
        for chunk_number, chunk in enumerate(
            split_text(page.extract_text() or ""), start=1
        ):
            records.append(
                {
                    "id": f"pdf-{page_number}-{chunk_number}",
                    "content": chunk,
                    "metadata": {
                        "source_type": "pdf",
                        "source_name": pdf_path.name,
                        "page": page_number,
                    },
                }
            )
    return records


def load_image_records(captions_path: Path, charts_folder: Path) -> list[dict]:
    """Turn prepared image captions into records that point to image files."""
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    records = []
    for image_name, details in captions.items():
        image_path = charts_folder / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Caption refers to missing image: {image_path}")
        facts = " ".join(details.get("structured_facts", []))
        records.append(
            {
                "id": f"image-{image_path.stem}",
                "content": f"{details['caption']} {facts}".strip(),
                "metadata": {
                    "source_type": "image",
                    "source_name": image_name,
                    "image_kind": details.get("image_kind", "image"),
                    "image_path": str(image_path.resolve()),
                },
            }
        )
    return records


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds into a classroom-friendly MM:SS timestamp."""
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def load_transcript_records(
    transcript_path: Path, segments_per_chunk: int = 2
) -> list[dict]:
    """Group transcript segments and preserve each chunk's start and end time."""
    if segments_per_chunk <= 0:
        raise ValueError("segments_per_chunk must be greater than zero.")
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    records = []
    for start_index in range(0, len(segments), segments_per_chunk):
        group = segments[start_index : start_index + segments_per_chunk]
        start_seconds = float(group[0]["start"])
        last = group[-1]
        end_seconds = float(last["start"]) + float(last["duration"])
        timestamp = (
            f"{seconds_to_timestamp(start_seconds)}-"
            f"{seconds_to_timestamp(end_seconds)}"
        )
        records.append(
            {
                "id": f"video-{start_index // segments_per_chunk + 1}",
                "content": " ".join(segment["text"] for segment in group),
                "metadata": {
                    "source_type": "video",
                    "source_name": transcript_path.stem,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "timestamp": timestamp,
                },
            }
        )
    return records


def build_multimodal_records(project_folder: Path) -> list[dict]:
    """Load the sample PDF, image captions, and transcript into one record list."""
    data_folder = project_folder / "data"
    records = load_pdf_records(data_folder / "sample_report.pdf")
    records.extend(
        load_image_records(data_folder / "image_captions.json", data_folder / "charts")
    )
    records.extend(
        load_transcript_records(
            data_folder / "transcripts" / "sample_youtube_transcript.json"
        )
    )
    return records


def create_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Create one OpenAI text embedding for every supplied string."""
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def open_chroma_client(database_folder: Path):
    """Open the local persistent Chroma database."""
    return chromadb.PersistentClient(path=str(database_folder))


def rebuild_collection(
    records: list[dict], database_folder: Path, client: OpenAI
):
    """Replace the demo collection and add records, embeddings, and metadata."""
    chroma_client = open_chroma_client(database_folder)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception as error:
        if "does not exist" not in str(error).lower():
            raise

    collection = chroma_client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    embeddings = create_embeddings([record["content"] for record in records], client)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["content"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=embeddings,
    )
    return collection


def load_collection(database_folder: Path):
    """Open the already-built Chroma collection used by query scripts."""
    try:
        return open_chroma_client(database_folder).get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError("Run build_index.py before asking questions.") from error


def retrieve(
    question: str, collection, client: OpenAI, top_k: int = 8
) -> list[dict]:
    """Embed a question and return the closest records from all modalities."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    result = collection.query(
        query_embeddings=[create_embeddings([question], client)[0]],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"content": document, "metadata": metadata, "distance": float(distance)}
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def select_balanced_evidence(
    retrieved: list[dict], limit: int = 5, per_source_type: int = 2
) -> list[dict]:
    """Keep retrieval order while preventing one modality from filling the prompt."""
    selected = []
    counts = {}
    for item in retrieved:
        source_type = item["metadata"]["source_type"]
        if counts.get(source_type, 0) >= per_source_type:
            continue
        selected.append(item)
        counts[source_type] = counts.get(source_type, 0) + 1
        if len(selected) == limit:
            break
    return selected


def source_label(metadata: dict) -> str:
    """Build a readable citation label from record metadata."""
    if metadata["source_type"] == "pdf":
        return f"PDF: {metadata['source_name']}, page {metadata['page']}"
    if metadata["source_type"] == "image":
        return f"Image: {metadata['source_name']}"
    return f"Video transcript: {metadata['source_name']}, {metadata['timestamp']}"


def format_context(evidence: list[dict]) -> str:
    """Number retrieved items so the model can cite Source 1, Source 2, etc."""
    return "\n\n".join(
        f"[Source {number}] {source_label(item['metadata'])}\n{item['content']}"
        for number, item in enumerate(evidence, start=1)
    )


def image_to_data_url(image_path: Path) -> str:
    """Encode a local image as a data URL accepted by the Responses API."""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def generate_answer(question: str, evidence: list[dict], client: OpenAI) -> str:
    """Generate a grounded answer and use late vision on one retrieved image."""
    prompt = f"""Answer the question using only the evidence below.
Use citations such as [Source 1] after every factual claim.
If the evidence is insufficient, say so clearly.

Question: {question}

Evidence:
{format_context(evidence)}
"""
    content = [{"type": "input_text", "text": prompt}]
    image_record = next(
        (item for item in evidence if item["metadata"]["source_type"] == "image"),
        None,
    )
    if image_record:
        image_path = Path(image_record["metadata"]["image_path"])
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image_path),
                "detail": "high",
            }
        )

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        instructions=(
            "You are a careful teaching assistant. Ground the answer in the "
            "provided sources and never invent missing facts."
        ),
        input=[{"role": "user", "content": content}],
    )
    return response.output_text


def print_evidence(evidence: list[dict]) -> None:
    """Print retrieved sources and distances for classroom inspection."""
    print("\nRetrieved evidence:")
    for number, item in enumerate(evidence, start=1):
        label = source_label(item["metadata"])
        print(f"{number}. {label} | distance={item['distance']:.3f}")

