"""Add `path` from chunks.json onto each entry in kg_dataset-linearized.json.

Match key: linearized entry's `chunk` field == chunks.json entry's `id` field.
Writes the linearized file back in place.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "chunks.json"
LINEARIZED_PATH = REPO_ROOT / "test_dataset" / "kg_dataset-linearized.json"


def build_path_index(chunks: list[dict]) -> dict[int, list[str]]:
    """Map chunk id -> path for O(1) lookup during merge."""
    return {c["id"]: c.get("path", []) for c in chunks}


def merge_paths(linearized: list[dict], path_by_id: dict[int, list[str]]) -> int:
    """Attach `path` from path_by_id onto each linearized entry, matched by chunk id.

    Returns the number of entries that were successfully matched.

    Every linearized chunk id is expected to exist in path_by_id; a miss
    means the two files have drifted and is treated as a hard error.
    """
    for entry in linearized:
        entry["path"] = path_by_id[entry["chunk"]]
    return len(linearized)


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    linearized = json.loads(LINEARIZED_PATH.read_text(encoding="utf-8"))

    path_by_id = build_path_index(chunks)
    matched = merge_paths(linearized, path_by_id)

    LINEARIZED_PATH.write_text(
        json.dumps(linearized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Matched {matched}/{len(linearized)} entries; wrote {LINEARIZED_PATH}")


if __name__ == "__main__":
    main()
