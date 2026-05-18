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
DATASET_PATH = Path(__file__).resolve().parent / "kg_test_dataset_linearized.json"
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
    "JE_PODLA",
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


FEW_SHOT_EXAMPLES = """# EXAMPLES

## Example 1 — minimal: Paragraf hierarchy + UPRAVUJE for top-level regulated subject

TEXT: "Tento zákon upravuje daň z pridanej hodnoty (ďalej len „daň"). "
Context: Paragraf § 1

Expected JSON:
{
  "nodes": [
    {"id": "Tento Zakon", "type": "PravnyPredpis", "properties": {}},
    {"id": "Paragraf § 1", "type": "Paragraf", "properties": {}},
    {"id": "Dan Z Pridanej Hodnoty", "type": "Dan", "properties": {}}
  ],
  "relationships": [
    {"source": {"id": "Tento Zakon", "type": "PravnyPredpis", "properties": {}},
     "target": {"id": "Paragraf § 1", "type": "Paragraf", "properties": {}},
     "type": "OBSAHUJE", "properties": {"section": "<section_id>"}},
    {"source": {"id": "Tento Zakon", "type": "PravnyPredpis", "properties": {}},
     "target": {"id": "Dan Z Pridanej Hodnoty", "type": "Dan", "properties": {}},
     "type": "UPRAVUJE", "properties": {"section": "<section_id>"}}
  ]
}

DO NOT emit `Dan Z Pridanej Hodnoty -[JE_PODLA]-> Paragraf § 1` — that is a hallucination.
The regulator-to-concept direction is `Paragraf/Odsek -[UPRAVUJE]-> Concept`, never the reverse.

## Example 2 — Konanie (action) regulated by Odsek, with Podmienka and MA_DATUM/MA_LEHOTU

TEXT: "(1) Podľa doterajších predpisov sa až do uplynutia posudzujú všetky lehoty, ktoré začali plynúť pred účinnosťou tohto zákona."
Context: Paragraf § 85 Odsek 1

Expected JSON:
{
  "nodes": [
    {"id": "Paragraf § 85", "type": "Paragraf", "properties": {}},
    {"id": "Paragraf § 85 Odsek 1", "type": "Odsek", "properties": {}},
    {"id": "Doterajsie Predpisy", "type": "PravnyPredpis", "properties": {}},
    {"id": "Tento Zakon", "type": "PravnyPredpis", "properties": {}},
    {"id": "Posudzovanie Vsetkych Lehot Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona", "type": "Konanie", "properties": {}},
    {"id": "Vsetky Lehoty Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona", "type": "Lehota", "properties": {}},
    {"id": "Do Uplynutia Tychto Lehot", "type": "Lehota", "properties": {}},
    {"id": "Podmienka Zacali Plynut Pred Ucinnostou Tohto Zakona", "type": "Podmienka", "properties": {}},
    {"id": "Ucinnost Tohto Zakona", "type": "Datum", "properties": {}}
  ],
  "relationships": [
    {"source":{"id":"Paragraf § 85","type":"Paragraf","properties":{}},"target":{"id":"Paragraf § 85 Odsek 1","type":"Odsek","properties":{}},"type":"OBSAHUJE","properties":{"section":"<section_id>"}},
    {"source":{"id":"Paragraf § 85 Odsek 1","type":"Odsek","properties":{}},"target":{"id":"Posudzovanie Vsetkych Lehot Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Konanie","properties":{}},"type":"UPRAVUJE","properties":{"section":"<section_id>"}},
    {"source":{"id":"Paragraf § 85 Odsek 1","type":"Odsek","properties":{}},"target":{"id":"Doterajsie Predpisy","type":"PravnyPredpis","properties":{}},"type":"ODKAZUJE_NA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Posudzovanie Vsetkych Lehot Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Konanie","properties":{}},"target":{"id":"Vsetky Lehoty Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Lehota","properties":{}},"type":"VZTAHUJE_SA_NA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Posudzovanie Vsetkych Lehot Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Konanie","properties":{}},"target":{"id":"Doterajsie Predpisy","type":"PravnyPredpis","properties":{}},"type":"JE_PODLA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Posudzovanie Vsetkych Lehot Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Konanie","properties":{}},"target":{"id":"Do Uplynutia Tychto Lehot","type":"Lehota","properties":{}},"type":"MA_LEHOTU","properties":{"section":"<section_id>"}},
    {"source":{"id":"Vsetky Lehoty Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Lehota","properties":{}},"target":{"id":"Podmienka Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Podmienka","properties":{}},"type":"MA_PODMIENKU","properties":{"section":"<section_id>"}},
    {"source":{"id":"Podmienka Zacali Plynut Pred Ucinnostou Tohto Zakona","type":"Podmienka","properties":{}},"target":{"id":"Ucinnost Tohto Zakona","type":"Datum","properties":{}},"type":"MA_DATUM","properties":{"section":"<section_id>"}},
    {"source":{"id":"Ucinnost Tohto Zakona","type":"Datum","properties":{}},"target":{"id":"Tento Zakon","type":"PravnyPredpis","properties":{}},"type":"VZTAHUJE_SA_NA","properties":{"section":"<section_id>"}}
  ]
}

Note the compound node ids: the entire legal qualifier ("Vsetky Lehoty Ktore Zacali Plynut Pred Ucinnostou Tohto Zakona") is ONE node — do not split into "Vsetky Lehoty" + edges to "Ucinnost Tohto Zakona".
Note `Podmienka` is its own typed node with the full condition wording, not a property of the Lehota.

## Example 3 — DEFINUJE, NEVZTAHUJE_SA_NA, ODKAZUJE_NA cross-section, JE_PODLA for named-after concepts

TEXT: "(2) Za faktúru sa považuje aj každý doklad alebo oznámenie, ktoré mení pôvodnú faktúru a osobitne a jednoznačne sa na ňu vzťahuje. Faktúrou podľa prvej vety nie je opravný doklad podľa § 25a."
Context: Paragraf § 71 Odsek 2

Expected JSON:
{
  "nodes": [
    {"id": "Paragraf § 71", "type": "Paragraf", "properties": {}},
    {"id": "Paragraf § 71 Odsek 2", "type": "Odsek", "properties": {}},
    {"id": "Paragraf § 25a", "type": "Paragraf", "properties": {}},
    {"id": "Faktura Podla Paragrafu § 71 Odsek 2 Prvej Vety", "type": "Dokument", "properties": {}},
    {"id": "Povodna Faktura", "type": "Dokument", "properties": {}},
    {"id": "Doklad Meniaci Povodnu Fakturu", "type": "Dokument", "properties": {}},
    {"id": "Oznamenie Meniace Povodnu Fakturu", "type": "Oznamenie", "properties": {}},
    {"id": "Opravny Doklad Podla Paragrafu § 25a", "type": "Dokument", "properties": {}},
    {"id": "Podmienka Osobitne A Jednoznacne Sa Vztahuje Na Povodnu Fakturu", "type": "Podmienka", "properties": {}}
  ],
  "relationships": [
    {"source":{"id":"Paragraf § 71","type":"Paragraf","properties":{}},"target":{"id":"Paragraf § 71 Odsek 2","type":"Odsek","properties":{}},"type":"OBSAHUJE","properties":{"section":"<section_id>"}},
    {"source":{"id":"Paragraf § 71 Odsek 2","type":"Odsek","properties":{}},"target":{"id":"Faktura Podla Paragrafu § 71 Odsek 2 Prvej Vety","type":"Dokument","properties":{}},"type":"DEFINUJE","properties":{"section":"<section_id>"}},
    {"source":{"id":"Paragraf § 71 Odsek 2","type":"Odsek","properties":{}},"target":{"id":"Paragraf § 25a","type":"Paragraf","properties":{}},"type":"ODKAZUJE_NA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Doklad Meniaci Povodnu Fakturu","type":"Dokument","properties":{}},"target":{"id":"Faktura Podla Paragrafu § 71 Odsek 2 Prvej Vety","type":"Dokument","properties":{}},"type":"JE_TYPOM","properties":{"section":"<section_id>"}},
    {"source":{"id":"Oznamenie Meniace Povodnu Fakturu","type":"Oznamenie","properties":{}},"target":{"id":"Faktura Podla Paragrafu § 71 Odsek 2 Prvej Vety","type":"Dokument","properties":{}},"type":"JE_TYPOM","properties":{"section":"<section_id>"}},
    {"source":{"id":"Doklad Meniaci Povodnu Fakturu","type":"Dokument","properties":{}},"target":{"id":"Povodna Faktura","type":"Dokument","properties":{}},"type":"MENI","properties":{"section":"<section_id>"}},
    {"source":{"id":"Oznamenie Meniace Povodnu Fakturu","type":"Oznamenie","properties":{}},"target":{"id":"Povodna Faktura","type":"Dokument","properties":{}},"type":"MENI","properties":{"section":"<section_id>"}},
    {"source":{"id":"Doklad Meniaci Povodnu Fakturu","type":"Dokument","properties":{}},"target":{"id":"Povodna Faktura","type":"Dokument","properties":{}},"type":"VZTAHUJE_SA_NA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Oznamenie Meniace Povodnu Fakturu","type":"Oznamenie","properties":{}},"target":{"id":"Povodna Faktura","type":"Dokument","properties":{}},"type":"VZTAHUJE_SA_NA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Doklad Meniaci Povodnu Fakturu","type":"Dokument","properties":{}},"target":{"id":"Podmienka Osobitne A Jednoznacne Sa Vztahuje Na Povodnu Fakturu","type":"Podmienka","properties":{}},"type":"MA_PODMIENKU","properties":{"section":"<section_id>"}},
    {"source":{"id":"Oznamenie Meniace Povodnu Fakturu","type":"Oznamenie","properties":{}},"target":{"id":"Podmienka Osobitne A Jednoznacne Sa Vztahuje Na Povodnu Fakturu","type":"Podmienka","properties":{}},"type":"MA_PODMIENKU","properties":{"section":"<section_id>"}},
    {"source":{"id":"Podmienka Osobitne A Jednoznacne Sa Vztahuje Na Povodnu Fakturu","type":"Podmienka","properties":{}},"target":{"id":"Povodna Faktura","type":"Dokument","properties":{}},"type":"VZTAHUJE_SA_NA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Opravny Doklad Podla Paragrafu § 25a","type":"Dokument","properties":{}},"target":{"id":"Paragraf § 25a","type":"Paragraf","properties":{}},"type":"JE_PODLA","properties":{"section":"<section_id>"}},
    {"source":{"id":"Faktura Podla Paragrafu § 71 Odsek 2 Prvej Vety","type":"Dokument","properties":{}},"target":{"id":"Opravny Doklad Podla Paragrafu § 25a","type":"Dokument","properties":{}},"type":"NEVZTAHUJE_SA_NA","properties":{"section":"<section_id>"}}
  ]
}

Notice:
- `JE_PODLA` is emitted only because the concept's *name* embeds the reference ("Opravny Doklad **Podla Paragrafu § 25a**"). The cross-section reference itself is `Paragraf § 71 Odsek 2 -[ODKAZUJE_NA]-> Paragraf § 25a`.
- The negation "nie je" becomes `NEVZTAHUJE_SA_NA` between the two concepts — NOT a missing edge.
- One `Podmienka` node carries the full condition wording; multiple subjects share it.

# DO / DO NOT cheat-sheet derived from the examples

DO:
- Always emit `Paragraf § X Odsek Y -[UPRAVUJE]-> [main legal action/object the odsek regulates]`.
- Always emit `Paragraf -[OBSAHUJE]-> Odsek`, `Odsek -[OBSAHUJE]-> Pismeno`, etc. — every level in the hierarchy.
- Create one `Podmienka` node per condition, with the full legal wording as its id (Title Case, diacritics removed). Then link `Subject -[MA_PODMIENKU]-> Podmienka` and `Podmienka -[VZTAHUJE_SA_NA]-> [each entity the condition references]`.
- Create one `Povinnost`/`Pravo` node per obligation/right, with the full wording. Then `Subject -[MA_POVINNOST|MA_PRAVO|MA_NAROK_NA]-> Povinnost/Pravo` and `Povinnost/Pravo -[VZTAHUJE_SA_NA]-> [object]`.
- Preserve compound legal qualifiers as a single node id (e.g. `Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu`). Do not split into `Tovar` + topology edges.
- For "§ X až § Y" enumerate each: emit one `ODKAZUJE_NA` per concrete paragraph (§ X, § X+1, …, § Y). Do not invent a range node.
- Use `MA_NAROK_NA` (not `MA_PRAVO`) when the text says "má nárok na".
- Use `DODAVA` (not `POSKYTUJE`) for both tovar and služba supply.
- Use `MA_IDENTIFIKATOR` (not `MA_STATUS`) for ID numbers (IČO, DIČ, IČ DPH).

DO NOT:
- Do NOT emit `Concept -[JE_PODLA]-> Paragraf` unless the concept's id literally ends with "Podla Paragrafu § …".
- Do NOT emit `Concept -[VYPLYVA_Z]-> Paragraf` for cross-section references — use `Paragraf -[ODKAZUJE_NA]-> Paragraf`.
- Do NOT add `MA`, `MA_ADRESU`, `MA_HODNOTU` edges unless the text explicitly names a value/address for that subject.
- Do NOT collapse conditions onto the subject (`Subject -[MA_STATUS]-> X`). Use a `Podmienka` intermediate.
- Do NOT use `POSKYTUJE` for service supply (use `DODAVA`).
- Do NOT use `VZNIKA` for place-of-origin ("z tuzemska") — that is `VZTAHUJE_SA_NA` or part of a compound node id.
"""


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

{FEW_SHOT_EXAMPLES}

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
    dataset_path: Path = DATASET_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    codex_bin: str = os.getenv(
        "CODEX_BIN",
        "/Users/samuelbagin/.vscode/extensions/openai.chatgpt-26.506.31421-darwin-arm64/bin/macos-aarch64/codex",
    )
    model: str | None = "gpt-5.4-mini"
    reasoning_effort: str | None = "high"
    chunk_ids: list[int] | None = None
    limit: int | None = None
    offset: int = 0
    start_chunk_index: int | None = None
    overwrite: bool = True
    timeout_seconds: int = 600

    schema = build_project_schema()
    agent_config = load_agent_config()

    # === ACTIVE: drive the loop from kg_dataset.json (text + path live in each item) ===
    dataset = load_dataset(dataset_path)
    chunks_index = None  # not needed; kg_dataset.json items already carry `path`

    # === DISABLED: chunks.json path (kept for re-running extraction over raw chunks) ===
    # dataset = load_chunks_as_items(CHUNKS_PATH)
    # chunks_index = None

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
