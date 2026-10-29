"""Ask a question against the persisted multimodal RAG collection."""

from pathlib import Path

from dotenv import load_dotenv

from utils import (
    generate_answer,
    get_openai_client,
    load_collection,
    print_evidence,
    retrieve,
    select_balanced_evidence,
)


PROJECT_FOLDER = Path(__file__).resolve().parent
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"
DEMO_QUESTION = (
    "Which support channel is largest in 2025, and when does the training video "
    "recommend using a vision model?"
)


def main() -> None:
    """Retrieve evidence, show it, and generate a cited answer."""
    load_dotenv(PROJECT_FOLDER / ".env")
    question = input(f"Question [{DEMO_QUESTION}]: ").strip() or DEMO_QUESTION
    client = get_openai_client()
    collection = load_collection(DATABASE_FOLDER)
    evidence = select_balanced_evidence(retrieve(question, collection, client))
    print_evidence(evidence)
    print("\nAnswer:")
    print(generate_answer(question, evidence, client))


if __name__ == "__main__":
    main()

