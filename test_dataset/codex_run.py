from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).resolve().parent / "kg_test_dataset.json"
CHUNKS_PATH = PROJECT_ROOT / "chunks.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "predictions.json"
AGENT_NAME = "kg-sde-extractor"
AGENT_PATH = PROJECT_ROOT / ".codex" / "agents" / f"{AGENT_NAME}.toml"


@dataclass
class Schema:
    """Schema dataclass to hold extracted node types and relationship types."""

    nodes: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    name: str
    developer_instructions: str


SCHEMA_NODE_TYPES = [
    "Subjekt",
    "Osoba",
    "Organizacia",
    "Adresa",
    "Lokacia",
    "Stat",
    "Banka",
    "Paragraf",
    "Odsek",
    "Pismeno",
    "Bod",
    "Priloha",
    "PravnyPredpis",
    "Dokument",
    "Konanie",
    "Rozhodnutie",
    "Ziadost",
    "Oznamenie",
    "Dan",
    "DanovePriznanie",
    "ZdanovacieObdobie",
    "SadzbaDane",
    "NadmernyOdpocet",
    "Sankcia",
    "Povinnost",
    "Pravo",
    "Podmienka",
    "Lehota",
    "Obdobie",
    "Datum",
    "Dovod",
    "Status",
    "Tovar",
    "Sluzba",
    "Majetok",
    "Nehnutelnost",
    "Vozidlo",
    "Ucet",
    "BankovyUcet",
    "Platba",
    "Suma",
    "Mnozstvo",
    "Obrat",
    "Mena",
    "Kurz",
    "Zmluva",
    "Pohladavka",
    "Zavazok",
    "Zastupenie",
    "Registracia",
    "Zaznam",
]


SCHEMA_RELATIONSHIP_TYPES = [
    "VZTAHUJE_SA_NA",
    "NEVZTAHUJE_SA_NA",
    "UPRAVUJE",
    "DEFINUJE",
    "URCUJE",
    "ODKAZUJE_NA",
    "VYPLYVA_Z",
    "JE_TYPOM",
    "JE_SUCASTOU",
    "OBSAHUJE",
    "PATRI_DO",
    "NACHADZA_SA_V",
    "MA",
    "MA_ADRESU",
    "MA_IDENTIFIKATOR",
    "MA_STATUS",
    "MA_DATUM",
    "MA_OBDOBIE",
    "MA_LEHOTU",
    "MA_SUMU",
    "MA_HODNOTU",
    "MA_PODMIENKU",
    "MA_PRAVO",
    "MA_POVINNOST",
    "MA_NAROK_NA",
    "NEMA_NAROK_NA",
    "SPLNA_PODMIENKY",
    "NESPLNA_PODMIENKY",
    "KONA_V_MENE",
    "JE_ZASTUPENA",
    "ZODPOVEDA_ZA",
    "PODAVA",
    "PREDKLADA",
    "DORUCUJE",
    "OZNAMUJE",
    "PRIJIMA",
    "VYDAVA",
    "ROZHODUJE_O",
    "REGISTRUJE",
    "UCHOVAVA",
    "DODAVA",
    "POSKYTUJE",
    "PLATI",
    "PODLIEHA",
    "OSLOBODZUJE_OD",
    "VZNIKA",
    "ZANIKA",
    "NADOBUDA",
    "PRECHADZA_NA",
    "MENI",
    "NAHRADZA",
    "RUSI",
    "SUVISI_S",
    "JE_OSLOBODENE_OD_DANE",
    "JE_PREDMETOM_DANE",
    "NIE_JE_PREDMETOM_DANE",
]


def build_project_schema() -> Schema:
    return Schema(
        nodes=list(SCHEMA_NODE_TYPES),
        relationships=list(SCHEMA_RELATIONSHIP_TYPES),
    )


def load_agent_config(path: Path = AGENT_PATH) -> AgentConfig:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return AgentConfig(
        name=str(data.get("name") or AGENT_NAME),
        developer_instructions=str(data.get("developer_instructions") or ""),
    )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a list, got {type(data).__name__}")
    return data


def load_chunks_as_items(path: Path) -> list[dict[str, Any]]:
    """Load chunks.json and normalize entries to the item shape used downstream.

    Each returned item has the same keys the rest of the pipeline expects:
      - chunk:   integer id (from chunks.json `id`)
      - text:    legal text body
      - path:    hierarchy path segments (e.g. ["§ 2", "1", "a)"])
      - headline:section title from chunks.json
    """
    if not path.exists():
        raise FileNotFoundError(f"chunks file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Chunks file must be a list, got {type(data).__name__}")
    items: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        items.append(
            {
                "chunk": entry.get("id"),
                "text": entry.get("text", ""),
                "path": entry.get("path", []),
                "headline": entry.get("headline"),
            }
        )
    return items


def load_chunks_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Chunks file must be a list, got {type(data).__name__}")
    index: dict[int, dict[str, Any]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("id")
        if isinstance(chunk_id, int):
            index[chunk_id] = entry
    return index


def resolve_chunk_path(
    item: dict[str, Any],
    chunks_index: dict[int, dict[str, Any]] | None = None,
) -> list[str]:
    direct = item.get("path")
    if isinstance(direct, list) and direct:
        return [str(segment) for segment in direct]
    if chunks_index is not None:
        chunk_id = item.get("chunk")
        if isinstance(chunk_id, int) and chunk_id in chunks_index:
            path_segments = chunks_index[chunk_id].get("path")
            if isinstance(path_segments, list):
                return [str(segment) for segment in path_segments]
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        fallback = metadata.get("path")
        if isinstance(fallback, list):
            return [str(segment) for segment in fallback]
    return []


def build_legal_context(path_segments: list[str]) -> tuple[str, str]:
    """Return (path_str, path_label) following the canonical legal ID format."""
    if not path_segments:
        return "", ""
    path_str = "Paragraf " + path_segments[0]
    path_label = "Paragraf"
    if len(path_segments) > 1:
        path_str += " Odsek " + path_segments[1]
        path_label = "Odsek"
    if len(path_segments) > 2:
        path_str += " Pismeno " + path_segments[2]
        path_label = "Pismeno"
    return path_str, path_label


def load_existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    raise ValueError(f"Predictions file must contain a list: {path}")


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def extraction_rules(agent_config: AgentConfig) -> str:
    marker = "Use this response schema exactly:"
    instructions = agent_config.developer_instructions
    if marker in instructions:
        instructions = instructions.split(marker, 1)[0].rstrip()
    return instructions


def build_prompt(
    item: dict[str, Any],
    schema: Schema,
    agent_config: AgentConfig,
    section_id: str,
    path_segments: list[str],
) -> str:
    path_str, _ = build_legal_context(path_segments)
    text = item.get("text", "")

    paragraf_line = (
        f"You are currently in Paragraf: {path_segments[0]}" if path_segments else ""
    )
    section_line = (
        f"Section context (current section): {path_str}" if path_str else ""
    )
    context_block = "\n".join(line for line in (paragraf_line, section_line) if line)
    if not context_block:
        context_block = "(no explicit paragraf context provided)"

    user_prompt = f"""Extract entities and relationships from the Slovak legal text strictly according to the provided schema.

# CONTEXT
{context_block}

# RULES
- use ONLY exact schema types
- no renaming or invented types
- prefer most specific valid type
- remove all diacritics
- use only Slovak language
- Node IDs must be in Title Case
- Relationship labels must be in SCREAMING_SNAKE_CASE
- decompose entities and relations into atomic legal units whenever valid
- if decomposition reduces legal meaning -> keep compound legal concept
- skip unsupported information


# LEGAL CONTEXT
If text contains:
- `odsek` without explicit paragraf:
  inherit active paragraf context

# SCHEMA
Entities: {", ".join(schema.nodes)}

Relationships: {", ".join(schema.relationships)}

# TEXT
{text}

# CHECK
- valid schema types only
- most specific types used
- full legal hierarchy in IDs
- odsek inheritance resolved
- odsek ranges atomically decomposed
- detailed legal entities included
"""

    return f"""You are executing the local Codex custom agent definition from {AGENT_PATH}.

Agent name: {agent_config.name}

# SYSTEM INSTRUCTIONS
{extraction_rules(agent_config)}

Process exactly one schema-driven extraction item. Do not run shell commands, do not edit files, and do not explain your work.

# USER REQUEST
{user_prompt}

Return only one JSON object with this exact shape:
{{
  "nodes": [
    {{
      "id": "Node Id",
      "type": "AllowedNodeLabel",
      "properties": {{}}
    }}
  ],
  "relationships": [
    {{
      "source": {{
        "id": "Source Node Id",
        "type": "AllowedSourceNodeLabel",
        "properties": {{}}
      }},
      "target": {{
        "id": "Target Node Id",
        "type": "AllowedTargetNodeLabel",
        "properties": {{}}
      }},
      "type": "ALLOWED_RELATIONSHIP_TYPE",
      "properties": {{
        "section": "{section_id}"
      }}
    }}
  ]
}}

Relationship evidence is still required for extraction quality, but do not include it in the final JSON.
"""


def run_codex_for_item(
    item: dict[str, Any],
    schema: Schema,
    agent_config: AgentConfig,
    section_id: str,
    path_segments: list[str],
    codex_bin: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    prompt = build_prompt(item, schema, agent_config, section_id, path_segments)
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", suffix=".json", delete=True
    ) as output_file:
        command = [
            codex_bin,
            "exec",
            "--cd",
            str(PROJECT_ROOT),
            "--sandbox",
            "workspace-write",
            "--output-last-message",
            output_file.name,
        ]
        if model:
            command.extend(["--model", model])
        if reasoning_effort:
            command.extend(
                ["--config", f'model_reasoning_effort="{reasoning_effort}"']
            )
        command.append(prompt)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(
                f"codex exited with {completed.returncode}: {stderr or stdout}"
            )

        output_file.seek(0)
        final_message = output_file.read().strip()
        return parse_codex_json(final_message or completed.stdout)


def parse_codex_json(output: str) -> dict[str, Any]:
    stripped = output.strip()
    if not stripped:
        raise ValueError("codex returned empty output")

    try:
        return require_json_object(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    for candidate in iter_json_candidates(stripped):
        try:
            return require_json_object(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    raise ValueError(f"could not parse JSON object from codex output: {stripped[:500]}")


def iter_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    fence = "```"
    start = 0
    while True:
        fence_start = text.find(fence, start)
        if fence_start == -1:
            break
        content_start = text.find("\n", fence_start + len(fence))
        if content_start == -1:
            break
        fence_end = text.find(fence, content_start + 1)
        if fence_end == -1:
            break
        candidates.append(text[content_start + 1 : fence_end].strip())
        start = fence_end + len(fence)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1])

    return candidates


def require_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"expected JSON object, got {type(value).__name__}")


def normalize_agent_payload(
    payload: dict[str, Any],
    schema: Schema,
    section_id: str,
) -> dict[str, Any]:
    allowed_nodes = set(schema.nodes)
    allowed_relationships = set(schema.relationships)

    raw_nodes = payload.get("nodes", [])
    raw_relationships = payload.get("relationships", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_relationships, list):
        raw_relationships = []

    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        label = str(node.get("label") or node.get("type") or "").strip()
        if not node_id or label not in allowed_nodes:
            continue
        nodes.append({"id": node_id, "type": label, "properties": {}})
        node_ids.add(node_id)

    relationships: list[dict[str, Any]] = []
    for relationship in raw_relationships:
        if not isinstance(relationship, dict):
            continue
        source = relationship.get("source")
        target = relationship.get("target")
        source_obj = source if isinstance(source, dict) else {}
        target_obj = target if isinstance(target, dict) else {}
        source_id = str(
            source_obj.get("id") or relationship.get("source_node_id") or ""
        ).strip()
        source_type = str(
            source_obj.get("type") or relationship.get("source_node_type") or ""
        ).strip()
        relation = str(relationship.get("type") or relationship.get("relation") or "").strip()
        target_id = str(
            target_obj.get("id") or relationship.get("target_node_id") or ""
        ).strip()
        target_type = str(
            target_obj.get("type") or relationship.get("target_node_type") or ""
        ).strip()
        evidence = str(relationship.get("evidence", "")).strip()
        if (
            not source_id
            or not target_id
            or source_type not in allowed_nodes
            or target_type not in allowed_nodes
            or relation not in allowed_relationships
            or source_id not in node_ids
            or target_id not in node_ids
        ):
            continue
        relationships.append(
            {
                "source": {
                    "id": source_id,
                    "type": source_type,
                    "properties": {},
                },
                "target": {
                    "id": target_id,
                    "type": target_type,
                    "properties": {},
                },
                "type": relation,
                "properties": {
                    "section": section_id,
                },
            }
        )

    return {"nodes": nodes, "relationships": relationships}


def build_prediction(
    item: dict[str, Any],
    payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload = payload or {"nodes": [], "relationships": []}
    return {
        "chunk": item.get("chunk"),
        "text": item.get("text", ""),
        "nodes": payload.get("nodes", []),
        "relationships": payload.get("relationships", []),
        "error": error,
    }


def main() -> int:
    # dataset_path: Path = DATASET_PATH  # used only by the disabled kg_test_dataset.json path
    output_path: Path = DEFAULT_OUTPUT_PATH
    codex_bin: str = os.getenv(
        "CODEX_BIN",
        "/Users/samuelbagin/.vscode/extensions/openai.chatgpt-26.506.31421-darwin-arm64/bin/macos-aarch64/codex",
    )
    model: str | None = "gpt-5.5"
    reasoning_effort: str | None = None
    chunk_ids: list[int] | None = [12, 45, 78, 103, 126, 149, 172, 195, 218, 241,
264, 287, 310, 333, 356, 379, 402, 425, 448, 471,
494, 517, 540, 563, 586, 609, 632, 655, 678, 701,
724, 747, 770, 793, 816, 839, 862, 885, 908, 931,
954, 977, 1000, 1023, 1046, 1069, 1092, 1115, 1138, 1161,
1184, 1207, 1230, 1253, 1276, 1299, 1322, 1345, 1368, 1391,
1414, 1437, 1460, 1483, 1506, 1529, 1552, 1575, 1598, 1600,
0, 31, 62, 93, 124, 155, 186, 217, 248, 279,
320, 361, 1140, 443, 484, 525, 566, 607, 648, 689,
730, 771, 812, 853, 894, 935, 976, 1017, 1058, 1099
]
    limit: int | None = None
    offset: int = 0
    start_chunk_index: int | None = None
    overwrite: bool = True
    timeout_seconds: int = 600

    schema = build_project_schema()
    agent_config = load_agent_config()

    # === ACTIVE: drive the loop from chunks.json (canonical source of text + path) ===
    dataset = load_chunks_as_items(CHUNKS_PATH)
    chunks_index = None  # not needed; chunks.json items already carry `path`

    # === DISABLED: kg_test_dataset.json path (kept for evaluation runs later) ===
    # dataset = load_dataset(dataset_path)
    # chunks_index = load_chunks_index(CHUNKS_PATH)

    selected_items = dataset[offset:]
    if chunk_ids is not None:
        chunk_id_set = set(chunk_ids)
        selected_items = [item for item in selected_items if item.get("chunk") in chunk_id_set]
    if limit is not None:
        selected_items = selected_items[:limit]

    predictions = [] if overwrite else load_existing_predictions(output_path)
    completed_chunks = {prediction.get("chunk") for prediction in predictions}
    section_start = offset if start_chunk_index is None else start_chunk_index

    for index, item in enumerate(selected_items, start=1):
        chunk = item.get("chunk")
        if chunk in completed_chunks:
            print(f"[{index}/{len(selected_items)}] chunk={chunk} already done")
            continue

        section_id = f"chunk_{section_start + index - 1}_kg_eval"
        path_segments = resolve_chunk_path(item, chunks_index)
        path_preview = " > ".join(path_segments) if path_segments else "(none)"
        print(
            f"[{index}/{len(selected_items)}] chunk={chunk} path={path_preview} running codex"
        )
        try:
            payload = run_codex_for_item(
                item=item,
                schema=schema,
                agent_config=agent_config,
                section_id=section_id,
                path_segments=path_segments,
                codex_bin=codex_bin,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            )
            normalized_payload = normalize_agent_payload(payload, schema, section_id)
            prediction = build_prediction(item, normalized_payload)
        except Exception as exc:
            prediction = build_prediction(item, error=str(exc))

        predictions.append(prediction)
        completed_chunks.add(chunk)
        write_predictions(output_path, predictions)
        print(
            f"[{index}/{len(selected_items)}] chunk={chunk} "
            f"nodes={len(prediction['nodes'])} "
            f"relationships={len(prediction['relationships'])} "
            f"error={prediction['error'] is not None}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
