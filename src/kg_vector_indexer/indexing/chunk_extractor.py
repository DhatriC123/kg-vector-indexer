from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExtractedMethodChunk:
    text: str
    start_line: int
    end_line: int


class JavaMethodExtractor:
    """Extract method bodies from Java source using brace balancing.

    This is intentionally simple for v1. It starts at the graph-provided method
    line and captures the first balanced block after the method signature.
    """

    def extract(self, file_path: Path, method_line: int, method_name: str) -> ExtractedMethodChunk | None:
        lines = file_path.read_text().splitlines()
        if method_line < 1 or method_line > len(lines):
            return None

        start_index = method_line - 1
        signature_line = self._find_signature_line(lines, start_index, method_name)
        if signature_line is None:
            return None

        signature_start = self._find_signature_start(lines, signature_line)
        block_start = self._find_block_start(lines, signature_line)
        if block_start is None:
            return None

        block_end = self._find_block_end(lines, block_start)
        if block_end is None:
            return None

        text = "\n".join(lines[signature_start : block_end + 1]).strip()
        return ExtractedMethodChunk(
            text=text,
            start_line=signature_start + 1,
            end_line=block_end + 1,
        )

    def _find_signature_line(self, lines: list[str], start_index: int, method_name: str) -> int | None:
        needle = f"{method_name}("
        for index in range(start_index, min(start_index + 25, len(lines))):
            if needle in lines[index]:
                return index
        return None

    def _find_signature_start(self, lines: list[str], start_index: int) -> int:
        index = start_index
        while index > 0:
            stripped = lines[index].strip()
            if stripped.startswith("@"):
                index -= 1
                continue
            if any(
                token in stripped
                for token in ("public ", "private ", "protected ", "void ", "static ", "default ")
            ) or "(" in stripped:
                break
            index -= 1
        return max(index, 0)

    def _find_block_start(self, lines: list[str], start_index: int) -> int | None:
        for index in range(start_index, len(lines)):
            if "{" in lines[index]:
                return index
        return None

    def _find_block_end(self, lines: list[str], block_start: int) -> int | None:
        depth = 0
        opened = False
        for index in range(block_start, len(lines)):
            for char in lines[index]:
                if char == "{":
                    depth += 1
                    opened = True
                elif char == "}":
                    depth -= 1
                    if opened and depth == 0:
                        return index
        return None
