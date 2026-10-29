"""Run a small modality-retrieval evaluation for the teaching collection."""

import json
from pathlib import Path

from dotenv import load_dotenv

from utils import get_openai_client, load_collection, retrieve


PROJECT_FOLDER = Path(__file__).resolve().parent
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"
QUESTIONS_PATH = PROJECT_FOLDER / "data" / "evaluation_questions.json"


def load_evaluation_questions(path: Path) -> list[dict]:
    """Read questions and their expected source modalities from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_question(case: dict, collection, client) -> bool:
    """Return True when all expected modalities appear in the retrieved results."""
    results = retrieve(case["question"], collection, client, top_k=8)
    found = {item["metadata"]["source_type"] for item in results}
    expected = set(case["expected_source_types"])
    missing = expected - found
    passed = not missing
    print(f"{'PASS' if passed else 'FAIL'}: {case['question']}")
    print(f"  expected={sorted(expected)} found={sorted(found)}")
    if missing:
        print(f"  missing={sorted(missing)}")
    return passed


def main() -> None:
    """Evaluate every sample question and print a simple pass rate."""
    load_dotenv(PROJECT_FOLDER / ".env")
    client = get_openai_client()
    collection = load_collection(DATABASE_FOLDER)
    cases = load_evaluation_questions(QUESTIONS_PATH)
    passed = sum(evaluate_question(case, collection, client) for case in cases)
    print(f"\nModality retrieval score: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()

