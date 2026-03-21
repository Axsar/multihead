"""Normalize and chunk records into span-addressable pieces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from multihead.knowledge_models import Record
from multihead.models import new_id


@dataclass
class Chunk:
    """A span-addressable piece of a record."""
    chunk_id: str = ""
    record_id: str = ""
    text: str = ""
    span_start: int = 0
    span_end: int = 0
    span_unit: str = "chars"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = new_id("chk_")


class Chunker:
    """Chunk text or JSONL records into overlapping segments."""

    def __init__(self, chunk_chars: int = 2000, overlap_chars: int = 200) -> None:
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_text(self, record_id: str, text: str) -> list[Chunk]:
        """Chunk plain text with overlap."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_chars, text_len)

            # Try to break at a paragraph or sentence boundary
            if end < text_len:
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_chars // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in (". ", ".\n", "! ", "? "):
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_chars // 2:
                            end = sent_break + len(sep)
                            break

            chunk_text = text[start:end]
            chunks.append(Chunk(
                record_id=record_id,
                text=chunk_text,
                span_start=start,
                span_end=end,
                span_unit="chars",
            ))

            # Advance with overlap
            if end >= text_len:
                break
            start = end - self.overlap_chars
            if start <= chunks[-1].span_start:
                start = end  # Avoid infinite loop on tiny overlap

        return chunks

    def chunk_jsonl(self, record_id: str, lines: list[str]) -> list[Chunk]:
        """Chunk JSONL records — each line or group of lines becomes a chunk."""
        if not lines:
            return []

        chunks = []
        current_text = ""
        current_start = 0
        char_offset = 0

        for line in lines:
            line_with_nl = line if line.endswith("\n") else line + "\n"

            if len(current_text) + len(line_with_nl) > self.chunk_chars and current_text:
                chunks.append(Chunk(
                    record_id=record_id,
                    text=current_text,
                    span_start=current_start,
                    span_end=current_start + len(current_text),
                    span_unit="chars",
                ))
                current_start = char_offset
                current_text = ""

            current_text += line_with_nl
            char_offset += len(line_with_nl)

        if current_text:
            chunks.append(Chunk(
                record_id=record_id,
                text=current_text,
                span_start=current_start,
                span_end=current_start + len(current_text),
                span_unit="chars",
            ))

        return chunks

    def chunk_record(self, record: Record, content: bytes) -> list[Chunk]:
        """Route to the right chunker based on record mime type."""
        mime = record.mime or ""

        if "jsonl" in mime:
            text = content.decode("utf-8", errors="replace")
            lines = text.strip().split("\n")
            return self.chunk_jsonl(record.record_id, lines)
        else:
            # Default: treat as text
            text = content.decode("utf-8", errors="replace")
            return self.chunk_text(record.record_id, text)

    def compute_coverage(self, chunks: list[Chunk], original_length: int) -> float:
        """Compute what fraction of the original text is covered by chunks."""
        if original_length == 0:
            return 1.0
        if not chunks:
            return 0.0

        # Build coverage bitmap (set of covered character positions)
        covered = set()
        for chunk in chunks:
            for i in range(chunk.span_start, min(chunk.span_end, original_length)):
                covered.add(i)

        return len(covered) / original_length
