"""
Evaluate text-to-knowledge-graph extraction against kg_dataset.json.

The script supports two workflows:
  1. Run an extraction LLM over each dataset chunk, then evaluate the output.
  2. Evaluate a precomputed predictions JSON against the gold dataset.

It computes deterministic KG metrics first, then optionally asks an LLM judge
only for ambiguous matching cases:
  - whether two entity/node IDs are semantically the same,
  - whether a relationship type is semantically correct when string matching
    fails,
  - audit of unresolved false positives / false negatives.

It intentionally does not ask the judge whether a triple is supported by the
source text.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KGNode:
    id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGEdge:
    source_id: str
    source_type: str
    relation: str
    target_id: str
    target_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGGraph:
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)


@dataclass
class Match:
    gold_index: int
    pred_index: int
    score: float
    method: str
    type_ok: bool = True
    judge_reason: str = ""


@dataclass
class EvalConfig:
    use_llm_judge: bool = True
    judge_model: str = "gpt-4o-mini"
    extraction_model: str = "gpt-4o-mini"
    relaxed_threshold: float = 0.92
    judge_candidate_threshold: float = 0.55
    legal_reference_judge_threshold: float = 0.35
    relationship_judge_candidate_threshold: float = 0.45
    max_judge_entity_pairs: int = 80
    max_judge_relation_pairs: int = 80
    audit_limit: int = 20
    
@dataclass
class Schema:
    """Schema dataclass to hold extracted node types and relationship types."""
    nodes: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Normalization and matching helpers
# ---------------------------------------------------------------------------


_WS = re.compile(r"\s+")
_LEADING_BULLET = re.compile(r"^\s*[-*•]\s*")
_MARKDOWN = re.compile(r"[*_`#]+")
_BAD_ID_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^\s*[-*•]\s+"),
    re.compile(r"^\s*unknown\s*$", re.IGNORECASE),
    re.compile(r"^\s*none\s*$", re.IGNORECASE),
    re.compile(r"^\s|null\s*$", re.IGNORECASE),
]


def strip_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = strip_diacritics(text)
    text = _LEADING_BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text).strip().casefold()
    return text


def normalize_node_id(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\s*§\s*", " § ", text)
    text = text.replace("paragraf paragraf", "paragraf")
    text = normalize_legal_reference_id(text)
    text = _WS.sub(" ", text).strip()
    return text


def normalize_legal_reference_id(text: str) -> str:
    """Canonicalize common Slovak legal-reference variants for matching."""
    text = re.sub(
        r"\bparagraf\s+(?!§)([0-9]+[a-z]?)\b",
        r"paragraf § \1",
        text,
    )
    text = re.sub(
        r"^odsek\s+([0-9]+[a-z]?)\s+paragraf\s+§\s+[0-9]+[a-z]?$",
        r"odsek \1",
        text,
    )

    law_match = re.search(
        r"\b(?:zakon\s+)?(?:c\.?\s*)?([0-9]+)(?:\s*/\s*|\s+)([0-9]{4})\s+z\.?\s*z\.?\b",
        text,
    )
    if law_match:
        return f"zakon {law_match.group(1)}/{law_match.group(2)} z z"

    return text


def normalize_type(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace(" ", "")
    return text


def normalize_relation(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.upper()


NORMALIZED_NODE_TYPES = {normalize_type(t) for t in SCHEMA_NODE_TYPES}
NORMALIZED_RELATIONSHIP_TYPES = {normalize_relation(t) for t in SCHEMA_RELATIONSHIP_TYPES}


def string_similarity(left: str, right: str) -> float:
    left = normalize_node_id(left)
    right = normalize_node_id(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return fuzz.token_set_ratio(left, right) / 100.0


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def dedupe_nodes(nodes: Iterable[KGNode]) -> list[KGNode]:
    seen: set[tuple[str, str]] = set()
    out: list[KGNode] = []
    for node in nodes:
        key = (normalize_node_id(node.id), normalize_type(node.type))
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def dedupe_edges(edges: Iterable[KGEdge]) -> list[KGEdge]:
    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[KGEdge] = []
    for edge in edges:
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def edge_key(edge: KGEdge) -> tuple[str, str, str, str, str]:
    return (
        normalize_node_id(edge.source_id),
        normalize_type(edge.source_type),
        normalize_relation(edge.relation),
        normalize_node_id(edge.target_id),
        normalize_type(edge.target_type),
    )


def node_identity_key(node: KGNode) -> str:
    return normalize_node_id(node.id)


def node_full_key(node: KGNode) -> tuple[str, str]:
    return normalize_node_id(node.id), normalize_type(node.type)


def is_bad_id(value: str) -> bool:
    return any(pattern.search(value or "") for pattern in _BAD_ID_PATTERNS)


def strip_extraction_system_items(graph: KGGraph) -> KGGraph:
    """Remove extraction bookkeeping items that should not affect KG evaluation."""
    system_node_ids = {
        normalize_node_id(node.id)
        for node in graph.nodes
        if normalize_type(node.type) in {"chunk", "document"}
    }
    return KGGraph(
        nodes=[
            node
            for node in graph.nodes
            if normalize_type(node.type) not in {"chunk", "document"}
        ],
        edges=[
            edge
            for edge in graph.edges
            if normalize_relation(edge.relation) not in {"IN_CHUNK", "IN_DOCUMENT"}
            and normalize_type(edge.source_type) not in {"chunk", "document"}
            and normalize_type(edge.target_type) not in {"chunk", "document"}
            and normalize_node_id(edge.source_id) not in system_node_ids
            and normalize_node_id(edge.target_id) not in system_node_ids
        ],
    )


# ---------------------------------------------------------------------------
# Loading and serialization
# ---------------------------------------------------------------------------


def build_project_schema() -> Schema:

    return Schema(
        nodes=list(SCHEMA_NODE_TYPES),
        relationships=list(SCHEMA_RELATIONSHIP_TYPES),
    )


def load_gold_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def parse_node(raw: dict[str, Any]) -> KGNode:
    raw_properties = raw.get("properties")
    properties: dict[str, Any] = raw_properties if isinstance(raw_properties, dict) else {}
    return KGNode(
        id=str(raw.get("id", "")),
        type=str(raw.get("type") or raw.get("label") or ""),
        properties=properties,
    )


def parse_edge(raw: dict[str, Any]) -> KGEdge:
    raw_source = raw.get("source")
    raw_target = raw.get("target")
    source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
    target: dict[str, Any] = raw_target if isinstance(raw_target, dict) else {}
    raw_properties = raw.get("properties")
    properties: dict[str, Any] = raw_properties if isinstance(raw_properties, dict) else {}
    return KGEdge(
        source_id=str(source.get("id") or raw.get("source_id") or raw.get("source_node_id") or ""),
        source_type=str(source.get("type") or raw.get("source_type") or raw.get("source_node_type") or ""),
        relation=str(raw.get("type") or raw.get("relation") or ""),
        target_id=str(target.get("id") or raw.get("target_id") or raw.get("target_node_id") or ""),
        target_type=str(target.get("type") or raw.get("target_type") or raw.get("target_node_type") or ""),
        properties=properties,
    )


def graph_from_payload(payload: dict[str, Any], include_system_nodes: bool = False) -> KGGraph:
    nodes = [parse_node(n) for n in payload.get("nodes", []) if isinstance(n, dict)]
    edges = [parse_edge(e) for e in payload.get("relationships", []) if isinstance(e, dict)]
    graph = KGGraph(nodes=dedupe_nodes(nodes), edges=dedupe_edges(edges))

    if not include_system_nodes:
        graph = strip_extraction_system_items(graph)

    return KGGraph(nodes=dedupe_nodes(graph.nodes), edges=dedupe_edges(graph.edges))


def graph_from_graph_document(graph_doc: Any, include_system_nodes: bool = False) -> KGGraph:
    payload = {
        "nodes": [
            {
                "id": getattr(node, "id", ""),
                "type": getattr(node, "type", ""),
                "properties": getattr(node, "properties", {}) or {},
            }
            for node in getattr(graph_doc, "nodes", []) or []
        ],
        "relationships": [
            {
                "source_id": getattr(getattr(rel, "source", None), "id", ""),
                "source_type": getattr(getattr(rel, "source", None), "type", ""),
                "relation": getattr(rel, "type", ""),
                "target_id": getattr(getattr(rel, "target", None), "id", ""),
                "target_type": getattr(getattr(rel, "target", None), "type", ""),
                "properties": getattr(rel, "properties", {}) or {},
            }
            for rel in getattr(graph_doc, "relationships", []) or []
        ],
    }
    return graph_from_payload(payload, include_system_nodes=include_system_nodes)


def strip_graph_document_system_items(graph_doc: Any) -> Any:
    """Hard-remove extraction bookkeeping from a GraphDocument-like object."""
    raw_nodes = list(getattr(graph_doc, "nodes", []) or [])
    raw_relationships = list(getattr(graph_doc, "relationships", []) or [])

    system_node_ids = {
        normalize_node_id(getattr(node, "id", ""))
        for node in raw_nodes
        if normalize_type(getattr(node, "type", "")) in {"chunk", "document"}
    }
    nodes = [
        node
        for node in raw_nodes
        if normalize_type(getattr(node, "type", "")) not in {"chunk", "document"}
    ]

    relationships = []
    for rel in raw_relationships:
        source = getattr(rel, "source", None)
        target = getattr(rel, "target", None)
        if normalize_relation(getattr(rel, "type", "")) in {"IN_CHUNK", "IN_DOCUMENT"}:
            continue
        if normalize_type(getattr(source, "type", "")) in {"chunk", "document"}:
            continue
        if normalize_type(getattr(target, "type", "")) in {"chunk", "document"}:
            continue
        if normalize_node_id(getattr(source, "id", "")) in system_node_ids:
            continue
        if normalize_node_id(getattr(target, "id", "")) in system_node_ids:
            continue
        relationships.append(rel)

    return SimpleNamespace(
        nodes=nodes,
        relationships=relationships,
        source=getattr(graph_doc, "source", None),
    )


def graph_to_payload(graph: KGGraph, include_system_nodes: bool = False) -> dict[str, Any]:
    if not include_system_nodes:
        graph = strip_extraction_system_items(graph)

    return {
        "nodes": [
            {"id": n.id, "type": n.type, "properties": n.properties}
            for n in graph.nodes
        ],
        "relationships": [
            {
                "source": {
                    "id": e.source_id,
                    "type": e.source_type,
                    "properties": {},
                },
                "target": {
                    "id": e.target_id,
                    "type": e.target_type,
                    "properties": {},
                },
                "type": e.relation,
                "properties": e.properties,
            }
            for e in graph.edges
        ],
    }


def load_predictions(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Predictions JSON must be a list")

    by_chunk: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        chunk = item.get("chunk")
        if chunk is None:
            chunk = item.get("metadata", {}).get("chunk") if isinstance(item.get("metadata"), dict) else None
        if chunk is None:
            chunk = index
        try:
            by_chunk[int(chunk)] = item
        except (TypeError, ValueError):
            by_chunk[index] = item
    return by_chunk


def resolve_predictions_path(
    predictions_path: Optional[Path],
    output_dir: Path,
) -> Optional[Path]:
    if predictions_path is None:
        candidate = output_dir / "predictions.json"
        return candidate if candidate.exists() else None

    if predictions_path.is_dir():
        return predictions_path / "predictions.json"

    return predictions_path


# ---------------------------------------------------------------------------
# LLM extraction and judge
# ---------------------------------------------------------------------------


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM extraction or LLM judge.")


def _json_from_llm_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


_RAG_INSTANCE: Any = None


def _make_rag() -> Any:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is not None:
        return _RAG_INSTANCE

    _require_openai_key()
    try:
        from pdf_graphrag import PDFGraphRAG
    except Exception as exc:
        raise RuntimeError("LLM extraction requires code/pdf_graphrag.py dependencies") from exc

    _RAG_INSTANCE = PDFGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return _RAG_INSTANCE


async def extract_graph_with_llm(
    text: str,
    page: Any,
    chunk: int,
    schema: Any,
    config: EvalConfig,
) -> KGGraph:
    del config
    from langchain_core.documents import Document

    document = Document(page_content=text, metadata={"page": page, "chunk": chunk})
    graph_documents = await _make_rag().async_schema_driven_extraction(
        documents=[document],
        schema=schema,
        document_id="kg_eval",
        max_concurrent=1,
    )

    if not graph_documents:
        return KGGraph()
    graph_document = strip_graph_document_system_items(graph_documents[0])
    return graph_from_graph_document(graph_document)


class LLMJudge:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.enabled = config.use_llm_judge
        self._client: Any = None
        self.entity_calls = 0
        self.relation_calls = 0

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        _require_openai_key()
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError("LLM judge requires langchain-openai") from exc
        self._client = ChatOpenAI(model=self.config.judge_model, temperature=0, max_retries=3, timeout=60)
        return self._client

    def _ask_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.client.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        return _json_from_llm_text(str(content))

    def same_entity(self, gold: KGNode, pred: KGNode) -> tuple[bool, bool, float, str]:
        if not self.enabled or self.entity_calls >= self.config.max_judge_entity_pairs:
            return False, False, 0.0, "judge disabled or entity call limit reached"
        self.entity_calls += 1
        system = (
            "You are an evaluator for Slovak legal knowledge graph extraction. "
            "Decide whether the gold node and predicted node refer to the same "
            "entity or legal/conceptual item. Also decide whether the predicted "
            "type is acceptable for the gold type. Do not evaluate whether any "
            "triple is supported by source text. Return strict JSON with keys: "
            "same_entity boolean, type_acceptable boolean, confidence number, reason string."
        )
        data = self._ask_json(
            system,
            {
                "gold": {"id": gold.id, "type": gold.type},
                "predicted": {"id": pred.id, "type": pred.type},
            },
        )
        return (
            bool(data.get("same_entity")),
            bool(data.get("type_acceptable")),
            float(data.get("confidence", 0.0) or 0.0),
            str(data.get("reason", "")),
        )

    def same_relation(self, gold: KGEdge, pred: KGEdge) -> tuple[bool, float, str]:
        if not self.enabled or self.relation_calls >= self.config.max_judge_relation_pairs:
            return False, 0.0, "judge disabled or relation call limit reached"
        self.relation_calls += 1
        system = (
            "You are an evaluator for Slovak legal knowledge graph relation extraction. "
            "The endpoints are already considered matched. Decide whether the predicted "
            "relationship has the same meaning and direction as the gold relationship. "
            "Do not evaluate whether the triple is supported by source text. Return "
            "strict JSON with keys: same_relation boolean, confidence number, reason string."
        )
        data = self._ask_json(
            system,
            {
                "gold": {
                    "source": {"id": gold.source_id, "type": gold.source_type},
                    "relation": gold.relation,
                    "target": {"id": gold.target_id, "type": gold.target_type},
                },
                "predicted": {
                    "source": {"id": pred.source_id, "type": pred.source_type},
                    "relation": pred.relation,
                    "target": {"id": pred.target_id, "type": pred.target_type},
                },
            },
        )
        return (
            bool(data.get("same_relation")),
            float(data.get("confidence", 0.0) or 0.0),
            str(data.get("reason", "")),
        )

    def same_unmatched_relation(self, gold: KGEdge, pred: KGEdge) -> tuple[bool, float, str]:
        if not self.enabled or self.relation_calls >= self.config.max_judge_relation_pairs:
            return False, 0.0, "judge disabled or relation call limit reached"
        self.relation_calls += 1
        system = (
            "You are an evaluator for Slovak legal knowledge graph relation extraction. "
            "The endpoints were not matched deterministically, so compare the whole "
            "gold triple and predicted triple semantically. Decide whether they express "
            "the same graph fact with the same direction, allowing harmless wording and "
            "legal-reference formatting differences. Do not evaluate whether the triple "
            "is supported by source text. Return strict JSON with keys: same_relation "
            "boolean, confidence number, reason string."
        )
        data = self._ask_json(
            system,
            {
                "gold": {
                    "source": {"id": gold.source_id, "type": gold.source_type},
                    "relation": gold.relation,
                    "target": {"id": gold.target_id, "type": gold.target_type},
                },
                "predicted": {
                    "source": {"id": pred.source_id, "type": pred.source_type},
                    "relation": pred.relation,
                    "target": {"id": pred.target_id, "type": pred.target_type},
                },
            },
        )
        return (
            bool(data.get("same_relation")),
            float(data.get("confidence", 0.0) or 0.0),
            str(data.get("reason", "")),
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def greedy_assign(candidates: list[Match]) -> list[Match]:
    assigned_gold: set[int] = set()
    assigned_pred: set[int] = set()
    result: list[Match] = []
    for candidate in sorted(candidates, key=lambda m: m.score, reverse=True):
        if candidate.gold_index in assigned_gold or candidate.pred_index in assigned_pred:
            continue
        assigned_gold.add(candidate.gold_index)
        assigned_pred.add(candidate.pred_index)
        result.append(candidate)
    return result


def match_nodes(gold: list[KGNode], pred: list[KGNode], judge: LLMJudge, config: EvalConfig) -> list[Match]:
    candidates: list[Match] = []

    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            same_id = normalize_node_id(g.id) == normalize_node_id(p.id)
            same_type = normalize_type(g.type) == normalize_type(p.type)
            if same_id and same_type:
                candidates.append(Match(gi, pi, 1.0, "strict", type_ok=True))
                continue

            sim = string_similarity(g.id, p.id)
            if same_type and sim >= config.relaxed_threshold:
                candidates.append(Match(gi, pi, 0.90 + sim / 100, "relaxed_string", type_ok=True))
                continue

            judge_threshold = (
                config.legal_reference_judge_threshold
                if is_legal_reference(g.id, g.type) or is_legal_reference(p.id, p.type)
                else config.judge_candidate_threshold
            )
            if sim >= judge_threshold:
                try:
                    same_entity, type_ok, confidence, reason = judge.same_entity(g, p)
                except Exception as exc:
                    same_entity, type_ok, confidence, reason = False, False, 0.0, f"judge_error: {exc}"
                if same_entity:
                    candidates.append(
                        Match(
                            gi,
                            pi,
                            0.75 + min(confidence, 1.0) / 10,
                            "llm_judge_entity",
                            type_ok=type_ok,
                            judge_reason=reason,
                        )
                    )

    return greedy_assign(candidates)


def endpoint_match(edge_gold: KGEdge, edge_pred: KGEdge, node_map: dict[str, str]) -> bool:
    gold_source = normalize_node_id(edge_gold.source_id)
    gold_target = normalize_node_id(edge_gold.target_id)
    pred_source = normalize_node_id(edge_pred.source_id)
    pred_target = normalize_node_id(edge_pred.target_id)
    return node_map.get(gold_source) == pred_source and node_map.get(gold_target) == pred_target


def is_legal_reference(value: str, node_type: str = "") -> bool:
    normalized = normalize_node_id(value)
    normalized_type = normalize_type(node_type)
    return (
        normalized_type in {"paragraf", "odsek", "pismeno", "pravnypredpis"}
        or "§" in normalized
        or "paragraf" in normalized
        or "odsek" in normalized
        or "ods." in normalized
        or "pism" in normalized
    )


def edge_has_legal_reference(edge: KGEdge) -> bool:
    return (
        is_legal_reference(edge.source_id, edge.source_type)
        or is_legal_reference(edge.target_id, edge.target_type)
    )


def edge_candidate_similarity(gold: KGEdge, pred: KGEdge) -> float:
    source_score = string_similarity(gold.source_id, pred.source_id)
    target_score = string_similarity(gold.target_id, pred.target_id)
    relation_score = string_similarity(normalize_relation(gold.relation), normalize_relation(pred.relation))
    source_type_score = 1.0 if normalize_type(gold.source_type) == normalize_type(pred.source_type) else 0.0
    target_type_score = 1.0 if normalize_type(gold.target_type) == normalize_type(pred.target_type) else 0.0
    return (
        0.30 * source_score
        + 0.30 * target_score
        + 0.25 * relation_score
        + 0.075 * source_type_score
        + 0.075 * target_type_score
    )


def match_edges(
    gold_edges: list[KGEdge],
    pred_edges: list[KGEdge],
    node_matches: list[Match],
    gold_nodes: list[KGNode],
    pred_nodes: list[KGNode],
    judge: LLMJudge,
    config: EvalConfig,
) -> list[Match]:
    node_map = {
        normalize_node_id(gold_nodes[m.gold_index].id): normalize_node_id(pred_nodes[m.pred_index].id)
        for m in node_matches
    }
    candidates: list[Match] = []

    for gi, g in enumerate(gold_edges):
        for pi, p in enumerate(pred_edges):
            if endpoint_match(g, p, node_map):
                if normalize_relation(g.relation) == normalize_relation(p.relation):
                    candidates.append(Match(gi, pi, 1.0, "strict_relation", type_ok=True))
                    continue
                try:
                    same_relation, confidence, reason = judge.same_relation(g, p)
                except Exception as exc:
                    same_relation, confidence, reason = False, 0.0, f"judge_error: {exc}"
                if same_relation:
                    candidates.append(
                        Match(
                            gi,
                            pi,
                            0.75 + min(confidence, 1.0) / 10,
                            "llm_judge_relation",
                            type_ok=True,
                            judge_reason=reason,
                        )
                    )
                continue

            sim = edge_candidate_similarity(g, p)
            judge_threshold = (
                config.legal_reference_judge_threshold
                if edge_has_legal_reference(g) or edge_has_legal_reference(p)
                else config.relationship_judge_candidate_threshold
            )
            if sim < judge_threshold:
                continue

            try:
                same_relation, confidence, reason = judge.same_unmatched_relation(g, p)
            except Exception as exc:
                same_relation, confidence, reason = False, 0.0, f"judge_error: {exc}"
            if same_relation:
                candidates.append(
                    Match(
                        gi,
                        pi,
                        0.60 + min(confidence, 1.0) / 10 + min(sim, 1.0) / 20,
                        "llm_judge_unmatched_relation",
                        type_ok=True,
                        judge_reason=reason,
                    )
                )

    return greedy_assign(candidates)


def strict_node_matches(gold: list[KGNode], pred: list[KGNode]) -> int:
    gold_keys = {node_full_key(n) for n in gold}
    pred_keys = {node_full_key(n) for n in pred}
    return len(gold_keys & pred_keys)


def strict_edge_matches(gold: list[KGEdge], pred: list[KGEdge]) -> int:
    return len({edge_key(e) for e in gold} & {edge_key(e) for e in pred})


def schema_quality(pred: KGGraph, schema: Any) -> dict[str, float]:
    del schema
    node_count = len(pred.nodes)
    edge_count = len(pred.edges)
    node_ids = {normalize_node_id(n.id) for n in pred.nodes}

    invalid_nodes = [
        n for n in pred.nodes
        if normalize_type(n.type) not in NORMALIZED_NODE_TYPES
    ]
    invalid_edges = [
        e for e in pred.edges
        if normalize_relation(e.relation) not in NORMALIZED_RELATIONSHIP_TYPES
    ]
    dangling = [
        e for e in pred.edges
        if normalize_node_id(e.source_id) not in node_ids or normalize_node_id(e.target_id) not in node_ids
    ]
    unknown_nodes = [n for n in pred.nodes if normalize_type(n.type) == "unknown"]
    bad_ids = [n for n in pred.nodes if is_bad_id(n.id) or str(n.type).strip().startswith("-")]

    return {
        "ontology_conformance_nodes": 1.0 - safe_div(len(invalid_nodes), node_count),
        "ontology_conformance_relationships": 1.0 - safe_div(len(invalid_edges), edge_count),
        "invalid_node_type_rate": safe_div(len(invalid_nodes), node_count),
        "invalid_relationship_type_rate": safe_div(len(invalid_edges), edge_count),
        "dangling_edge_rate": safe_div(len(dangling), edge_count),
        "unknown_type_rate": safe_div(len(unknown_nodes), node_count),
        "bad_id_format_rate": safe_div(len(bad_ids), node_count),
    }


def evaluate_graph(
    gold: KGGraph,
    pred: KGGraph,
    schema: Any,
    judge: LLMJudge,
    config: EvalConfig,
) -> dict[str, Any]:
    node_matches = match_nodes(gold.nodes, pred.nodes, judge, config)
    edge_matches = match_edges(gold.edges, pred.edges, node_matches, gold.nodes, pred.nodes, judge, config)

    node_tp_identity = len(node_matches)
    node_tp_full = sum(1 for m in node_matches if m.type_ok)
    node_fp = max(0, len(pred.nodes) - node_tp_identity)
    node_fn = max(0, len(gold.nodes) - node_tp_identity)

    edge_tp = len(edge_matches)
    edge_fp = max(0, len(pred.edges) - edge_tp)
    edge_fn = max(0, len(gold.edges) - edge_tp)

    strict_node_tp = strict_node_matches(gold.nodes, pred.nodes)
    strict_edge_tp = strict_edge_matches(gold.edges, pred.edges)

    node_precision = safe_div(node_tp_identity, node_tp_identity + node_fp)
    node_recall = safe_div(node_tp_identity, node_tp_identity + node_fn)
    edge_precision = safe_div(edge_tp, edge_tp + edge_fp)
    edge_recall = safe_div(edge_tp, edge_tp + edge_fn)

    strict_node_precision = safe_div(strict_node_tp, len(pred.nodes))
    strict_node_recall = safe_div(strict_node_tp, len(gold.nodes))
    strict_edge_precision = safe_div(strict_edge_tp, len(pred.edges))
    strict_edge_recall = safe_div(strict_edge_tp, len(gold.edges))

    unmatched_gold_nodes = sorted(set(range(len(gold.nodes))) - {m.gold_index for m in node_matches})
    unmatched_pred_nodes = sorted(set(range(len(pred.nodes))) - {m.pred_index for m in node_matches})
    unmatched_gold_edges = sorted(set(range(len(gold.edges))) - {m.gold_index for m in edge_matches})
    unmatched_pred_edges = sorted(set(range(len(pred.edges))) - {m.pred_index for m in edge_matches})

    normalized_ged = safe_div(
        node_fp + node_fn + edge_fp + edge_fn + (node_tp_identity - node_tp_full),
        len(gold.nodes) + len(pred.nodes) + len(gold.edges) + len(pred.edges),
    )

    details = {
        "node_matches": [
            {
                "gold": gold.nodes[m.gold_index].id,
                "predicted": pred.nodes[m.pred_index].id,
                "method": m.method,
                "type_ok": m.type_ok,
                "reason": m.judge_reason,
            }
            for m in node_matches
        ],
        "edge_matches": [
            {
                "gold": f"{gold.edges[m.gold_index].source_id} -[{gold.edges[m.gold_index].relation}]-> {gold.edges[m.gold_index].target_id}",
                "predicted": f"{pred.edges[m.pred_index].source_id} -[{pred.edges[m.pred_index].relation}]-> {pred.edges[m.pred_index].target_id}",
                "method": m.method,
                "reason": m.judge_reason,
            }
            for m in edge_matches
        ],
        "unmatched_gold_nodes": [gold.nodes[i].id for i in unmatched_gold_nodes[: config.audit_limit]],
        "unmatched_pred_nodes": [pred.nodes[i].id for i in unmatched_pred_nodes[: config.audit_limit]],
        "unmatched_gold_edges": [
            f"{gold.edges[i].source_id} -[{gold.edges[i].relation}]-> {gold.edges[i].target_id}"
            for i in unmatched_gold_edges[: config.audit_limit]
        ],
        "unmatched_pred_edges": [
            f"{pred.edges[i].source_id} -[{pred.edges[i].relation}]-> {pred.edges[i].target_id}"
            for i in unmatched_pred_edges[: config.audit_limit]
        ],
    }

    metrics: dict[str, Any] = {
        "gold_nodes": len(gold.nodes),
        "pred_nodes": len(pred.nodes),
        "gold_relationships": len(gold.edges),
        "pred_relationships": len(pred.edges),
        "node_tp": node_tp_identity,
        "node_fp": node_fp,
        "node_fn": node_fn,
        "relationship_tp": edge_tp,
        "relationship_fp": edge_fp,
        "relationship_fn": edge_fn,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": f1(node_precision, node_recall),
        "node_type_accuracy_on_matches": safe_div(node_tp_full, node_tp_identity),
        "relationship_precision": edge_precision,
        "relationship_recall": edge_recall,
        "relationship_f1": f1(edge_precision, edge_recall),
        "strict_node_precision": strict_node_precision,
        "strict_node_recall": strict_node_recall,
        "strict_node_f1": f1(strict_node_precision, strict_node_recall),
        "strict_relationship_precision": strict_edge_precision,
        "strict_relationship_recall": strict_edge_recall,
        "strict_relationship_f1": f1(strict_edge_precision, strict_edge_recall),
        "node_hallucination_rate": safe_div(node_fp, len(pred.nodes)),
        "node_omission_rate": safe_div(node_fn, len(gold.nodes)),
        "relationship_hallucination_rate": safe_div(edge_fp, len(pred.edges)),
        "relationship_omission_rate": safe_div(edge_fn, len(gold.edges)),
        "normalized_ged": normalized_ged,
        **schema_quality(pred, schema),
        "details": details,
    }
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "gold_nodes": sum(r["gold_nodes"] for r in rows),
        "pred_nodes": sum(r["pred_nodes"] for r in rows),
        "node_tp": sum(r["node_tp"] for r in rows),
        "node_fp": sum(r["node_fp"] for r in rows),
        "node_fn": sum(r["node_fn"] for r in rows),
        "gold_relationships": sum(r["gold_relationships"] for r in rows),
        "pred_relationships": sum(r["pred_relationships"] for r in rows),
        "relationship_tp": sum(r["relationship_tp"] for r in rows),
        "relationship_fp": sum(r["relationship_fp"] for r in rows),
        "relationship_fn": sum(r["relationship_fn"] for r in rows),
    }

    node_p = safe_div(totals["node_tp"], totals["node_tp"] + totals["node_fp"])
    node_r = safe_div(totals["node_tp"], totals["node_tp"] + totals["node_fn"])
    rel_p = safe_div(totals["relationship_tp"], totals["relationship_tp"] + totals["relationship_fp"])
    rel_r = safe_div(totals["relationship_tp"], totals["relationship_tp"] + totals["relationship_fn"])

    averaged_keys = [
        "strict_node_f1",
        "strict_relationship_f1",
        "node_type_accuracy_on_matches",
        "ontology_conformance_nodes",
        "ontology_conformance_relationships",
        "invalid_node_type_rate",
        "invalid_relationship_type_rate",
        "dangling_edge_rate",
        "unknown_type_rate",
        "bad_id_format_rate",
        "normalized_ged",
    ]
    macro = {
        f"macro_{key}": safe_div(sum(float(r.get(key, 0.0)) for r in rows), len(rows))
        for key in averaged_keys
    }
    return {
        **totals,
        "chunks": len(rows),
        "micro_node_precision": node_p,
        "micro_node_recall": node_r,
        "micro_node_f1": f1(node_p, node_r),
        "micro_relationship_precision": rel_p,
        "micro_relationship_recall": rel_r,
        "micro_relationship_f1": f1(rel_p, rel_r),
        "node_hallucination_rate": safe_div(totals["node_fp"], totals["pred_nodes"]),
        "node_omission_rate": safe_div(totals["node_fn"], totals["gold_nodes"]),
        "relationship_hallucination_rate": safe_div(totals["relationship_fp"], totals["pred_relationships"]),
        "relationship_omission_rate": safe_div(totals["relationship_fn"], totals["gold_relationships"]),
        **macro,
    }


# ---------------------------------------------------------------------------
# LangSmith integration
# ---------------------------------------------------------------------------


def get_or_create_langsmith_dataset(client: Any, name: str, description: str) -> Any:
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        return existing[0]
    return client.create_dataset(dataset_name=name, description=description)


def sync_langsmith_dataset(client: Any, name: str, examples: list[dict[str, Any]]) -> str:
    dataset = get_or_create_langsmith_dataset(
        client,
        name,
        "KG extraction evaluation: text chunks with gold nodes and relationships.",
    )
    existing_count = len(list(client.list_examples(dataset_id=dataset.id)))
    for item in examples[existing_count:]:
        client.create_example(
            inputs={
                "chunk": item["chunk"],
                "page": item.get("page"),
                "text": item["text"],
            },
            outputs={
                "gold_graph": {
                    "nodes": item.get("nodes", []),
                    "relationships": item.get("relationships", []),
                }
            },
            dataset_id=dataset.id,
        )
    return dataset.name


def run_langsmith_evaluation(
    examples: list[dict[str, Any]],
    schema: Any,
    config: EvalConfig,
    predictions: Optional[dict[int, dict[str, Any]]],
    dataset_name: str,
) -> Any:
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except Exception as exc:
        raise RuntimeError("LangSmith mode requires langsmith package") from exc

    client = Client()
    ls_dataset = sync_langsmith_dataset(client, dataset_name, examples)
    judge = LLMJudge(config)
    gold_by_chunk = {int(item["chunk"]): item for item in examples}

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        chunk = int(inputs["chunk"])
        if predictions is not None:
            pred_payload = predictions.get(chunk)
            if pred_payload is None:
                return {"prediction": {"nodes": [], "relationships": []}, "error": "missing prediction"}
            return {"prediction": graph_to_payload(graph_from_payload(pred_payload))}
        graph = asyncio.run(
            extract_graph_with_llm(
                text=inputs["text"],
                page=inputs.get("page"),
                chunk=chunk,
                schema=schema,
                config=config,
            )
        )
        return {"prediction": graph_to_payload(graph)}

    def evaluator(run: Any, example: Any) -> dict[str, Any]:
        chunk = int(example.inputs["chunk"])
        gold_graph = graph_from_payload(gold_by_chunk[chunk])
        pred_graph = graph_from_payload(run.outputs.get("prediction", {}))
        result = evaluate_graph(gold_graph, pred_graph, schema, judge, config)
        comment = {
            "node_f1": result["node_f1"],
            "relationship_f1": result["relationship_f1"],
            "hallucination": result["relationship_hallucination_rate"],
            "omission": result["relationship_omission_rate"],
        }
        return {
            "key": "kg_relationship_f1",
            "score": result["relationship_f1"],
            "comment": json.dumps(comment, ensure_ascii=False),
        }

    return evaluate(
        target,
        data=ls_dataset,
        evaluators=[evaluator],
        experiment_prefix="kg_extraction_eval",
        client=client,
    )


# ---------------------------------------------------------------------------
# Local runner and reports
# ---------------------------------------------------------------------------


CSV_FIELDS = [
    "chunk",
    "page",
    "gold_nodes",
    "pred_nodes",
    "node_precision",
    "node_recall",
    "node_f1",
    "gold_relationships",
    "pred_relationships",
    "relationship_precision",
    "relationship_recall",
    "relationship_f1",
    "strict_relationship_f1",
    "relationship_hallucination_rate",
    "relationship_omission_rate",
    "ontology_conformance_nodes",
    "ontology_conformance_relationships",
    "bad_id_format_rate",
    "unknown_type_rate",
    "dangling_edge_rate",
    "normalized_ged",
    "latency_seconds",
]


def _render_error_audit(rows: list[dict[str, Any]]) -> str:
    lines = ["# KG Evaluation Error Audit", ""]
    for row in sorted(rows, key=lambda r: r["relationship_f1"])[:20]:
        details = row.get("details", {})
        lines.extend(
            [
                f"## Chunk {row['chunk']} | relationship_f1={row['relationship_f1']:.3f}",
                "",
                f"- Missing gold relationships: {details.get('unmatched_gold_edges', [])}",
                f"- Extra predicted relationships: {details.get('unmatched_pred_edges', [])}",
                f"- Missing gold nodes: {details.get('unmatched_gold_nodes', [])}",
                f"- Extra predicted nodes: {details.get('unmatched_pred_nodes', [])}",
                "",
            ]
        )
    return "\n".join(lines)


async def evaluate_examples(
    examples: list[dict[str, Any]],
    schema: Any,
    config: EvalConfig,
    predictions: Optional[dict[int, dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
    append: bool = False,
    write_predictions: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judge = LLMJudge(config)
    rows: list[dict[str, Any]] = []
    predictions_out: list[dict[str, Any]] = []

    csv_handle = None
    csv_writer = None
    by_chunk_json_path: Optional[Path] = None
    predictions_json_path: Optional[Path] = None
    audit_path: Optional[Path] = None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "by_chunk.csv"
        by_chunk_json_path = output_dir / "by_chunk.json"
        if write_predictions:
            predictions_json_path = output_dir / "predictions.json"
        audit_path = output_dir / "error_audit.md"

        if append:
            if by_chunk_json_path.exists():
                try:
                    data = json.loads(by_chunk_json_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        rows.extend(data)
                except json.JSONDecodeError:
                    pass
            if predictions_json_path is not None and predictions_json_path.exists():
                try:
                    data = json.loads(predictions_json_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        predictions_out.extend(data)
                except json.JSONDecodeError:
                    pass

        use_append_csv = append and csv_path.exists()
        csv_handle = csv_path.open("a" if use_append_csv else "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        if not use_append_csv:
            csv_writer.writeheader()
        csv_handle.flush()

        by_chunk_json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        if predictions_json_path is not None:
            predictions_json_path.write_text(json.dumps(predictions_out, indent=2, ensure_ascii=False), encoding="utf-8")
        audit_path.write_text(_render_error_audit(rows), encoding="utf-8")

    try:
        for index, item in enumerate(examples):
            chunk = int(item.get("chunk", index))
            gold_graph = graph_from_payload(item)
            started = time.perf_counter()

            if predictions is not None:
                pred_payload = predictions.get(chunk)
                pred_graph = graph_from_payload(pred_payload or {})
            else:
                pred_graph = await extract_graph_with_llm(
                    text=item["text"],
                    page=item.get("page"),
                    chunk=chunk,
                    schema=schema,
                    config=config,
                )

            elapsed = time.perf_counter() - started
            result = evaluate_graph(gold_graph, pred_graph, schema, judge, config)
            result["chunk"] = chunk
            result["page"] = item.get("page")
            result["latency_seconds"] = elapsed
            rows.append(result)
            predictions_out.append({"chunk": chunk, **graph_to_payload(pred_graph)})

            if csv_writer is not None and csv_handle is not None:
                csv_writer.writerow({key: result.get(key) for key in CSV_FIELDS})
                csv_handle.flush()

            if by_chunk_json_path is not None:
                by_chunk_json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            if predictions_json_path is not None:
                predictions_json_path.write_text(json.dumps(predictions_out, indent=2, ensure_ascii=False), encoding="utf-8")
            if audit_path is not None:
                audit_path.write_text(_render_error_audit(rows), encoding="utf-8")

            print(
                f"chunk={chunk} node_f1={result['node_f1']:.3f} "
                f"rel_f1={result['relationship_f1']:.3f} "
                f"rel_hall={result['relationship_hallucination_rate']:.3f} "
                f"rel_omis={result['relationship_omission_rate']:.3f}"
            )
    finally:
        if csv_handle is not None:
            csv_handle.close()

    return rows, predictions_out


def write_reports(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_dir: Path,
    write_csv: bool = True,
    append: bool = False,
) -> None:
    del predictions, append  # streamed per-chunk by evaluate_examples
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_csv:
        with (output_dir / "by_chunk.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in CSV_FIELDS})


def main() -> None:
    dataset_path: Path = Path(__file__).resolve().parent / "kg_test_dataset_linearized.json"
    predictions_path: Optional[Path] = Path(__file__).resolve().parent / "linearized_eval_results_few-shot_gpt-5.5"
    output_dir: Path = Path(__file__).resolve().parent / "linearized_eval_results_few-shot_gpt-5.5"
    limit: Optional[int] = None
    offset: int = 0
    start_chunk_index: Optional[int] = None
    no_write: bool = False
    no_llm_judge: bool = False
    judge_model: str = os.getenv("KG_EVAL_JUDGE_MODEL", "gpt-4o-mini")
    extraction_model: str = os.getenv("KG_EVAL_EXTRACTION_MODEL", "gpt-4o-mini")
    relaxed_threshold: float = 0.75
    judge_candidate_threshold: float = 0.55
    legal_reference_judge_threshold: float = 0.35
    relationship_judge_candidate_threshold: float = 0.35
    use_langsmith: bool = False
    langsmith_dataset: str = "kg_extraction_eval_v2"

    dataset = load_gold_dataset(dataset_path)
    examples = dataset[offset:]
    if start_chunk_index is not None:
        examples = examples[start_chunk_index:]
    if limit is not None:
        examples = examples[:limit]

    append_mode = start_chunk_index is not None

    schema = build_project_schema()
    resolved_predictions_path = resolve_predictions_path(predictions_path, output_dir)
    predictions = load_predictions(resolved_predictions_path) if resolved_predictions_path else None
    if predictions is not None:
        before_count = len(examples)
        examples = [
            item
            for index, item in enumerate(examples)
            if int(item.get("chunk", index)) in predictions
        ]
        skipped_count = before_count - len(examples)
        print(
            f"Loaded predictions for {len(predictions)} chunks from {resolved_predictions_path}. "
            f"Evaluating {len(examples)} matched chunks; skipping {skipped_count} gold chunks without predictions."
        )
        if not examples:
            raise ValueError("No matching chunk IDs between the gold dataset and predictions.")
    config = EvalConfig(
        use_llm_judge=not no_llm_judge,
        judge_model=judge_model,
        extraction_model=extraction_model,
        relaxed_threshold=relaxed_threshold,
        judge_candidate_threshold=judge_candidate_threshold,
        legal_reference_judge_threshold=legal_reference_judge_threshold,
        relationship_judge_candidate_threshold=relationship_judge_candidate_threshold,
    )

    if use_langsmith:
        result = run_langsmith_evaluation(
            examples=examples,
            schema=schema,
            config=config,
            predictions=predictions,
            dataset_name=langsmith_dataset,
        )
        print(result)
        return

    csv_output_dir = None if no_write else output_dir
    rows, pred_out = asyncio.run(
        evaluate_examples(
            examples,
            schema,
            config,
            predictions,
            output_dir=csv_output_dir,
            append=append_mode,
            write_predictions=predictions is None,
        )
    )
    summary = aggregate(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not no_write:
        write_reports(rows, pred_out, output_dir, write_csv=False, append=append_mode)
        print(f"Wrote reports to {output_dir}")


if __name__ == "__main__":
    main()
