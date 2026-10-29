"""Remove only this lesson's generated Chroma database."""

import shutil
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parent
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"


def main() -> None:
    """Delete the generated database so build_index.py can start fresh."""
    if DATABASE_FOLDER.exists():
        shutil.rmtree(DATABASE_FOLDER)
        print(f"Removed {DATABASE_FOLDER}.")
    else:
        print("No local database exists. Nothing to remove.")


if __name__ == "__main__":
    main()

