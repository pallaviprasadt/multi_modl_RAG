"""Offline checks for loading, chunking, metadata, and citation helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import (
    build_multimodal_records,
    format_context,
    image_to_data_url,
    rebuild_collection,
    seconds_to_timestamp,
    select_balanced_evidence,
    source_label,
    split_text,
)


PROJECT_FOLDER = Path(__file__).resolve().parent


class MultimodalRagTests(unittest.TestCase):
    """Verify the local teaching pipeline without making API calls."""

    def test_split_text_keeps_overlap(self) -> None:
        """The next chunk should repeat the requested boundary words."""
        chunks = split_text("one two three four five six", chunk_size=4, overlap=2)
        self.assertEqual(chunks, ["one two three four", "three four five six"])

    def test_timestamp_format(self) -> None:
        """Seconds should become the timestamp format shown to learners."""
        self.assertEqual(seconds_to_timestamp(67), "01:07")

    def test_all_three_modalities_load(self) -> None:
        """The sample corpus should produce PDF, image, and video records."""
        records = build_multimodal_records(PROJECT_FOLDER)
        source_types = {record["metadata"]["source_type"] for record in records}
        self.assertEqual(source_types, {"pdf", "image", "video"})
        self.assertTrue(all(record["content"] for record in records))

    def test_balancing_caps_each_modality(self) -> None:
        """Balanced evidence should keep at most two records per modality."""
        retrieved = [
            {"metadata": {"source_type": "pdf"}, "content": str(number)}
            for number in range(4)
        ]
        retrieved.append({"metadata": {"source_type": "image"}, "content": "chart"})
        selected = select_balanced_evidence(retrieved, limit=5, per_source_type=2)
        self.assertEqual(len(selected), 3)

    def test_source_labels_and_context(self) -> None:
        """Citation helpers should expose page, filename, and source number."""
        item = {
            "content": "Example",
            "metadata": {
                "source_type": "pdf",
                "source_name": "report.pdf",
                "page": 2,
            },
        }
        self.assertEqual(source_label(item["metadata"]), "PDF: report.pdf, page 2")
        self.assertIn("[Source 1]", format_context([item]))

    def test_image_data_url(self) -> None:
        """A local image should be encoded without contacting an API."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.png"
            path.write_bytes(b"fake-png-bytes")
            self.assertTrue(image_to_data_url(path).startswith("data:image/png;base64,"))

    def test_chroma_collection_builds(self) -> None:
        """The installed Chroma API should persist records with supplied vectors."""
        records = [
            {
                "id": "one",
                "content": "first record",
                "metadata": {"source_type": "pdf", "source_name": "one.pdf"},
            },
            {
                "id": "two",
                "content": "second record",
                "metadata": {"source_type": "image", "source_name": "two.png"},
            },
        ]
        # Chroma may briefly keep a Windows file handle after the assertion.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            with patch("utils.create_embeddings", return_value=[[1.0, 0.0], [0.0, 1.0]]):
                collection = rebuild_collection(records, Path(folder), client=None)
            self.assertEqual(collection.count(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

