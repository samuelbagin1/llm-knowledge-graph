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
    "Adresa",
    "Agentura",
    "Banka",
    "BankovyUcet",
    "Bod",
    "CasovyUdaj",
    "Cinnost",
    "Cislo",
    "ClenSkupiny",
    "ClenskyStat",
    "Dan",
    "DanovePriznanie",
    "Datum",
    "Doklad",
    "Dovod",
    "ElektronickyProstriedok",
    "Euro",
    "FinancneRiaditelstvo",
    "FyzickaOsoba",
    "Hodnota",
    "IdentifikacneCislo",
    "InvesticnyMajetok",
    "Konanie",
    "Kurz",
    "Lehota",
    "Limit",
    "Lokacia",
    "Majetok",
    "Mena",
    "Ministerstvo",
    "Mnozstvo",
    "NadmernyOdpocet",
    "Nariadenie",
    "NarodnaBanka",
    "Nehnutelnost",
    "Obrat",
    "Obdobie",
    "Odsek",
    "Oprava",
    "Organizacia",
    "OslobodenieOdDane",
    "Osoba",
    "Oznamenie",
    "Paragraf",
    "Pismeno",
    "Pohladavka",
    "Pokuta",
    "Podmienka",
    "Podnik",
    "Poukaz",
    "Povinnost",
    "Pravo",
    "PravnickaOsoba",
    "PravnyNastupca",
    "PravnyPredpis",
    "Prevazdkaren",
    "Priloha",
    "Registracia",
    "Rozhodnutie",
    "SadzbaDane",
    "Sankcia",
    "Sidlo",
    "Skupina",
    "Sluzba",
    "Smernica",
    "SpotrebnaDan",
    "SpravcaDane",
    "Stat",
    "StatnyOrgan",
    "Status",
    "Stavba",
    "Subjekt",
    "Sud",
    "Suma",
    "Tovar",
    "TretiStat",
    "Tuzemsko",
    "Ucet",
    "Urad",
    "Urok",
    "Uzemie",
    "Vozidlo",
    "Vyhlaska",
    "Vypocet",
    "Vyzva",
    "ZabezpekaNaDan",
    "Zakon",
    "Zasielka",
    "Zastupca",
    "Zavazok",
    "Zaznam",
    "ZdanitelnaOsoba",
    "ZdanovacieObdobie",
    "Ziadost",
    "Zlava",
    "Zmluva",
    "Zodpovednost",
]


SCHEMA_RELATIONSHIP_TYPES = [
    "APLIKUJE_SA_NA",
    "DEFINUJE",
    "DODA",
    "DODAVA",
    "DOPLNA",
    "DORUCUJE",
    "JE_CASTOU",
    "JE_CLENOM",
    "JE_DRUHOM",
    "JE_OSLOBODENE_OD",
    "JE_PODLA",
    "JE_POVINNY_PLATIT",
    "JE_PREDMETOM",
    "JE_SUCASTOU",
    "JE_TYPOM",
    "JE_ZASTUPENA",
    "KONA_V_MENE",
    "MA_ADRESU",
    "MA_BYDLISKO",
    "MA_CENU",
    "MA_DATUM",
    "MA_DOBU",
    "MA_DOKLAD",
    "MA_DOVOD",
    "MA_HODNOTU",
    "MA_IDENTIFIKACNE_CISLO",
    "MA_LEHOTU",
    "MA_MIESTO",
    "MA_MIESTO_DODANIA",
    "MA_MIESTO_PODNIKANIA",
    "MA_MNOZSTVO",
    "MA_NAROK_NA",
    "MA_NAZOV",
    "MA_OBDOBIE",
    "MA_OBSAH",
    "MA_ODKLADNY_UCINOK",
    "MA_ODSEK",
    "MA_PISMENO",
    "MA_PODMIENKU",
    "MA_POVINNOST",
    "MA_PRAVO",
    "MA_PREVADZKAREN",
    "MA_SADZBU",
    "MA_SIDLO",
    "MA_STATUS",
    "MA_SUMU",
    "MA_UCEL",
    "MA_UCINOK",
    "MA_VLASTNOST",
    "MA_VYNIMKU",
    "MA_ZAKLAD_DANE",
    "MA_ZASTUPCU",
    "NADOBUDA",
    "NACHADZA_SA_V",
    "NAHRADZA",
    "NASTAVA_PRI",
    "NEMA_NAROK_NA",
    "NEPLATI_PRE",
    "NESPLNA_PODMIENKY",
    "NEVZTAHUJE_SA_NA",
    "OBSAHUJE",
    "ODKAZUJE_NA",
    "OPRAVUJE",
    "OSLOBODZUJE_OD",
    "OZNAMUJE",
    "PATRI_DO",
    "PLATI_DO",
    "PLATI_OD",
    "PLATI_PRE",
    "PODAVA",
    "PODLIEHA",
    "PODMIENUJE",
    "POSKYTUJE",
    "POVAZUJE_SA_ZA",
    "PRECHADZA_NA",
    "PREDKLADA",
    "PRESAHUJE",
    "PREUKAZUJE",
    "PRIJIMA",
    "PRIDELUJE",
    "REGISTRUJE",
    "ROZHODUJE_O",
    "ROZUMIE_SA",
    "RUSI",
    "SPADA_POD",
    "SPLNA_PODMIENKY",
    "STAVA_SA",
    "SUVISI_S",
    "TYKA_SA",
    "UCHOVAVA",
    "UPRAVUJE",
    "URCUJE",
    "USKUTOCNUJE",
    "UVADZA",
    "VIES_ZAZNAMY_O",
    "VYDAVA",
    "VYCHADZA_Z",
    "VYKONAVA",
    "VYMEDZUJE",
    "VYPLNYVA_Z",
    "VZTAHUJE_SA_NA",
    "VZNIKA_PRI",
    "ZAHRNUJE",
    "ZANIKA",
    "ZAPLATI",
    "ZODPOVEDA_ZA",
    "ZRUSUJE",
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
) -> str:
    payload = {
        "chunk": item.get("chunk"),
        "page": item.get("page"),
        "text": item.get("text", ""),
        "metadata": item.get("metadata", {}),
        "schema": {
            "nodes": schema.nodes,
            "relationships": schema.relationships,
        },
    }

    return f"""You are executing the local Codex custom agent definition from {AGENT_PATH}.

Agent name: {agent_config.name}

Extraction instructions:
{extraction_rules(agent_config)}

Process exactly one schema-driven extraction item. Do not run shell commands, do not edit files, and do not explain your work.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

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
    codex_bin: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    prompt = build_prompt(item, schema, agent_config, section_id)
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
    model: str | None = "gpt-5.5"
    reasoning_effort: str | None = "high"
    chunk_ids: list[int] | None = None
    limit: int | None = None
    offset: int = 0
    start_chunk_index: int | None = None
    overwrite: bool = True
    timeout_seconds: int = 600

    schema = build_project_schema()
    agent_config = load_agent_config()
    dataset = load_dataset(dataset_path)
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
        print(f"[{index}/{len(selected_items)}] chunk={chunk} running codex")
        try:
            payload = run_codex_for_item(
                item=item,
                schema=schema,
                agent_config=agent_config,
                section_id=section_id,
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
