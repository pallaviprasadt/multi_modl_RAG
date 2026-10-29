"""Build the local Chroma index from PDF, image, and transcript evidence."""

from pathlib import Path

from dotenv import load_dotenv

from utils import build_multimodal_records, get_openai_client, rebuild_collection


PROJECT_FOLDER = Path(__file__).resolve().parent
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"


def main() -> None:
    """Load teaching data, create embeddings, and persist one collection."""
    load_dotenv(PROJECT_FOLDER / ".env")
    records = build_multimodal_records(PROJECT_FOLDER)
    collection = rebuild_collection(records, DATABASE_FOLDER, get_openai_client())
    print(f"Indexed {collection.count()} records in {DATABASE_FOLDER.name}.")
    for source_type in ("pdf", "image", "video"):
        count = sum(
            record["metadata"]["source_type"] == source_type for record in records
        )
        print(f"- {source_type}: {count}")


if __name__ == "__main__":
    main()

