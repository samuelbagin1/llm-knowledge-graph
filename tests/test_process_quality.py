"""
LangSmith-based stage quality evaluation for the three pipeline stages:
  1. ODD  — open_domain_detection
  2. REF  — schema_refinement
  3. SDE  — schema_driven_extraction

Prerequisites:
  pip install langsmith
  export LANGCHAIN_API_KEY=<your_key>
  export LANGCHAIN_TRACING_V2=true

Run from root:
  python tests/test_stage_quality.py

# ---------------------------------------------------------------------------
# TODO — MISSING EVALUATORS (not yet implemented)
# ---------------------------------------------------------------------------
#
# REF — eval_ref_deduplication
#   Check that known synonym groups collapse to a single canonical type.
#   e.g. ["Dokument", "Dokumentacia"] → only "Dokument" survives in node_types.
#   Controlled input already exists (DUPLICATE_ODD_SCHEMA), but no evaluator
#   reads example.outputs["expected_canonical_nodes"] to verify deduplication.
#
# ODD — PASCAL_CASE regex currently allows digits (^[A-Z][a-zA-Z0-9]+$).
#   Spec says ^[A-Z][a-zA-Z]+$ (letters only). Decide whether digits are
#   intentional (e.g. "ISO9001") and document the choice, or tighten the regex.
# ---------------------------------------------------------------------------
"""

import asyncio
import os
import re
import sys

sys.path.insert(0, "./code")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument
from langsmith import Client
from langsmith.evaluation import evaluate

from classes import Schema

load_dotenv()

# ---------------------------------------------------------------------------
# LangSmith client
# ---------------------------------------------------------------------------

ls_client = Client()

# ---------------------------------------------------------------------------
# Curated fixtures
# ---------------------------------------------------------------------------

# Short, representative Slovak legal text covering the main entity types
# the pipeline is expected to produce (PravnyPredpis, Paragraf, Platitel, etc.)
FIXTURE_TEXT = """
Zákon č. 222/2004 Z. z. o dani z pridanej hodnoty v znení neskorších predpisov.
§ 54 ods. 2 pism. a) – Platiteľ dane je povinný podať daňové priznanie do 25 dní
po skončení zdaňovacieho obdobia. Ministerstvo financií Slovenskej republiky vydalo
usmernenie č. 5/2024. Spoločnosť ABC s.r.o., so sídlom v Bratislave, je registrovaná
ako platiteľ DPH podľa § 4 zákona č. 222/2004 Z. z.
"""

# What ODD should minimally return for FIXTURE_TEXT
EXPECTED_ODD_NODE_TYPES = {"PravnyPredpis", "Paragraf", "Platitel", "Ministerstvo", "Spolocnost"}
EXPECTED_ODD_REL_TYPES = {"OBSAHUJE", "VYDAL", "PODLA"}

# Controlled raw schema with intentional duplicates for refinement testing
DUPLICATE_ODD_SCHEMA = Schema(
    nodes=[
        "Dokument", "Dokumentacia", "PravnyPredpis", "Zakon",
        "Osoba", "FyzickaOsoba", "Spolocnost", "Firma"
    ],
    relationships=[
        "OBSAHUJE", "MA_OBSAH", "VYDAL", "PUBLIKOVAL",
        "PRACUJE_PRE", "JE_ZAMESTNANCOM"
    ],
)

# Type pairs that are semantically distinct and must NOT be merged by refinement
DISTINCT_PAIRS = [
    ("Osoba", "Spolocnost"),
    ("PravnyPredpis", "Paragraf"),
]

# ---------------------------------------------------------------------------
# Format helpers — deterministic, no LLM, zero extra cost
# ---------------------------------------------------------------------------

DIACRITICS = re.compile(r"[áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]")
PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]+$")
UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]+$")


def _has_diacritics(text: str) -> bool:
    return bool(DIACRITICS.search(text))


def _is_pascal_case(text: str) -> bool:
    return bool(PASCAL_CASE.match(text))


def _is_upper_snake(text: str) -> bool:
    return bool(UPPER_SNAKE.match(text))


# ---------------------------------------------------------------------------
# ODD evaluators
# ---------------------------------------------------------------------------

def eval_odd_no_diacritics(run, example) -> dict:
    """All type names must be diacritic-free (prompt rule §8)."""
    schema: Schema = run.outputs["output"]
    violations = [t for t in schema.nodes + schema.relationships if _has_diacritics(t)]
    return {
        "key": "odd_no_diacritics",
        "score": 1.0 if not violations else 0.0,
        "comment": f"Violations: {violations}" if violations else "OK",
    }


def eval_odd_node_format(run, example) -> dict:
    """Node types must be PascalCase (prompt rule §2)."""
    schema: Schema = run.outputs["output"]
    bad = [t for t in schema.nodes if not _is_pascal_case(t)]
    score = max(0.0, 1.0 - len(bad) / max(len(schema.nodes), 1))
    return {
        "key": "odd_node_format",
        "score": score,
        "comment": f"Bad format: {bad}" if bad else "OK",
    }


def eval_odd_rel_format(run, example) -> dict:
    """Relationship types must be UPPER_SNAKE_CASE (prompt rule §3)."""
    schema: Schema = run.outputs["output"]
    bad = [t for t in schema.relationships if not _is_upper_snake(t)]
    score = max(0.0, 1.0 - len(bad) / max(len(schema.relationships), 1))
    return {
        "key": "odd_rel_format",
        "score": score,
        "comment": f"Bad format: {bad}" if bad else "OK",
    }


def eval_odd_not_empty(run, example) -> dict:
    """Both node and relationship lists must be non-empty."""
    schema: Schema = run.outputs["output"]
    ok = len(schema.nodes) > 0 and len(schema.relationships) > 0
    return {"key": "odd_not_empty", "score": 1.0 if ok else 0.0}


def eval_odd_recall(run, example) -> dict:
    """
    Recall: fraction of expected node AND relationship types found in the extracted schema.
    Scored separately per list, averaged into one score.
    Case-insensitive — the LLM may capitalise slightly differently.
    Reference: example.outputs["expected_node_types"] and ["expected_rel_types"]
    """
    schema: Schema = run.outputs["output"]
    expected_nodes: list = example.outputs.get("expected_node_types", [])
    expected_rels: list = example.outputs.get("expected_rel_types", [])

    extracted_nodes_lower = {t.lower() for t in schema.nodes}
    extracted_rels_lower = {t.lower() for t in schema.relationships}

    node_recall = (
        sum(1 for t in expected_nodes if t.lower() in extracted_nodes_lower) / len(expected_nodes)
        if expected_nodes else 1.0
    )
    rel_recall = (
        sum(1 for t in expected_rels if t.lower() in extracted_rels_lower) / len(expected_rels)
        if expected_rels else 1.0
    )
    score = (node_recall + rel_recall) / 2

    missing_nodes = [t for t in expected_nodes if t.lower() not in extracted_nodes_lower]
    missing_rels = [t for t in expected_rels if t.lower() not in extracted_rels_lower]
    comment_parts = []
    if missing_nodes:
        comment_parts.append(f"missing nodes: {missing_nodes}")
    if missing_rels:
        comment_parts.append(f"missing rels: {missing_rels}")
    return {
        "key": "odd_recall",
        "score": score,
        "comment": "; ".join(comment_parts) if comment_parts else "All expected types found",
    }


def eval_odd_precision(run, example) -> dict:
    """
    Precision: fraction of extracted types that are valid (present in the reference set).
    Only extracted types that also appear in the ground-truth reference are considered correct.
    Case-insensitive.
    Reference: example.outputs["expected_node_types"] and ["expected_rel_types"]

    Note: the reference is a *minimal* expected set — the LLM may correctly extract
    additional valid types not in the fixture. This evaluator therefore gives a lower-bound
    estimate of precision; review the 'extra' list manually to judge if they are correct.
    """
    schema: Schema = run.outputs["output"]
    expected_nodes: set = {t.lower() for t in example.outputs.get("expected_node_types", [])}
    expected_rels: set = {t.lower() for t in example.outputs.get("expected_rel_types", [])}

    if not expected_nodes and not expected_rels:
        return {"key": "odd_precision", "score": 1.0, "comment": "No reference"}

    extracted_nodes_lower = [t.lower() for t in schema.nodes]
    extracted_rels_lower = [t.lower() for t in schema.relationships]

    node_precision = (
        sum(1 for t in extracted_nodes_lower if t in expected_nodes) / len(extracted_nodes_lower)
        if extracted_nodes_lower else 1.0
    )
    rel_precision = (
        sum(1 for t in extracted_rels_lower if t in expected_rels) / len(extracted_rels_lower)
        if extracted_rels_lower else 1.0
    )
    score = (node_precision + rel_precision) / 2

    extra_nodes = [t for t in schema.nodes if t.lower() not in expected_nodes]
    extra_rels = [t for t in schema.relationships if t.lower() not in expected_rels]
    comment_parts = []
    if extra_nodes:
        comment_parts.append(f"extra nodes (check manually): {extra_nodes}")
    if extra_rels:
        comment_parts.append(f"extra rels (check manually): {extra_rels}")
    return {
        "key": "odd_precision",
        "score": score,
        "comment": "; ".join(comment_parts) if comment_parts else "All extracted types in reference",
    }


def eval_odd_no_instances(run, example) -> dict:
    """
    Heuristic: node types should be abstract schema types, not concrete instances.
    Prompt rule §1: return only abstract types, not specific instances.

    Two signals currently implemented:
    1. Contains digits  → likely a specific entity (e.g. 'Zakon222', 'Paragraf54')
    2. Contains a space → node types must be single PascalCase words; a space means the
                          LLM returned a phrase or named instance (e.g. 'Jan Novak',
                          '§ 54 ods. 2') instead of an abstract label.

    TODO: extend with domain-specific Slovak legal patterns, e.g.:
      - re.match(r'^§', t)         — paragraph references used as types
      - re.match(r'^č\\.', t)      — law citation numbers used as types
      - known Slovak place/person name prefixes (e.g. 'Bratislava', 'Ministerstvo financii')
      - cross-check extracted types against known instance IDs from your fixture document
    """
    schema: Schema = run.outputs["output"]
    flagged = [
        t for t in schema.nodes
        if any(c.isdigit() for c in t) or " " in t
    ]
    score = max(0.0, 1.0 - len(flagged) / max(len(schema.nodes), 1))
    return {
        "key": "odd_no_instances",
        "score": score,
        "comment": f"Possible instances: {flagged}" if flagged else "OK",
    }


# ---------------------------------------------------------------------------
# Refinement evaluators
# ---------------------------------------------------------------------------

def eval_ref_paragraf_present(run, example) -> dict:
    """'Paragraf' must always appear in refined node_types (hardcoded injection in prompt)."""
    data: dict = run.outputs["output"]
    present = "Paragraf" in data.get("node_types", [])
    return {"key": "ref_paragraf_present", "score": 1.0 if present else 0.0}


def eval_ref_no_new_types(run, example) -> dict:
    """
    Refinement must not invent types absent from both odd_schema and existing_schema.
    The only allowed injection is 'Paragraf' (explicitly added by the prompt).
    Reference: run.inputs["odd_schema"] and run.inputs["existing_schema"]
    """
    data: dict = run.outputs["output"]
    inputs: dict = run.inputs

    allowed_nodes = set(inputs["odd_schema"].nodes)
    if inputs.get("existing_schema"):
        allowed_nodes |= set(inputs["existing_schema"].nodes)
    allowed_nodes.add("Paragraf")

    allowed_rels = set(inputs["odd_schema"].relationships)
    if inputs.get("existing_schema"):
        allowed_rels |= set(inputs["existing_schema"].relationships)

    invented_nodes = [t for t in data.get("node_types", []) if t not in allowed_nodes]
    invented_rels = [t for t in data.get("relationship_types", []) if t not in allowed_rels]
    all_invented = invented_nodes + invented_rels
    return {
        "key": "ref_no_new_types",
        "score": 1.0 if not all_invented else 0.0,
        "comment": f"Invented types: {all_invented}" if all_invented else "OK",
    }


def eval_ref_merge_log_consistency(run, example) -> dict:
    """
    Every canonical key in merge_log must exist in the output type lists.
    Orphan keys mean the LLM documented a merge but forgot to include the canonical type.
    """
    data: dict = run.outputs["output"]
    node_types = set(data.get("node_types", []))
    rel_types = set(data.get("relationship_types", []))
    merge_log = data.get("merge_log", {})

    bad_node_keys = [k for k in merge_log.get("node_types", {}) if k not in node_types]
    bad_rel_keys = [k for k in merge_log.get("relationship_types", {}) if k not in rel_types]
    all_bad = bad_node_keys + bad_rel_keys
    return {
        "key": "ref_merge_log_consistency",
        "score": 1.0 if not all_bad else 0.0,
        "comment": f"Orphan merge_log keys: {all_bad}" if all_bad else "OK",
    }


def eval_ref_no_diacritics(run, example) -> dict:
    """All refined type names must be diacritic-free (same rule as ODD)."""
    data: dict = run.outputs["output"]
    all_types = data.get("node_types", []) + data.get("relationship_types", [])
    violations = [t for t in all_types if _has_diacritics(t)]
    return {
        "key": "ref_no_diacritics",
        "score": 1.0 if not violations else 0.0,
        "comment": f"Violations: {violations}" if violations else "OK",
    }


# TODO: eval_ref_deduplication — verify that synonym groups in DUPLICATE_ODD_SCHEMA
#   collapse correctly. Read example.outputs["expected_canonical_nodes"] and check
#   each canonical type is in data["node_types"] AND that its synonyms are absent.
#   Pattern: expected = ["Dokument", "PravnyPredpis", "Osoba", "Spolocnost"]
#   score = len(present_canonicals) / len(expected)


def eval_ref_distinct_not_merged(run, example) -> dict:
    """
    Semantically distinct type pairs must not be collapsed into one.
    Checks DISTINCT_PAIRS against the merge_log.
    A failure here means the model over-consolidated the schema.
    """
    data: dict = run.outputs["output"]
    merge_log_nodes = data.get("merge_log", {}).get("node_types", {})
    violations = []
    for a, b in DISTINCT_PAIRS:
        if a in merge_log_nodes and b in merge_log_nodes.get(a, []):
            violations.append(f"{b} incorrectly merged into {a}")
        if b in merge_log_nodes and a in merge_log_nodes.get(b, []):
            violations.append(f"{a} incorrectly merged into {b}")
    return {
        "key": "ref_distinct_not_merged",
        "score": 1.0 if not violations else 0.0,
        "comment": f"Over-merges: {violations}" if violations else "OK",
    }


# ---------------------------------------------------------------------------
# SDE evaluators
# ---------------------------------------------------------------------------

def eval_sde_schema_compliance_nodes(run, example) -> dict:
    """
    All extracted node types (except 'Chunk') must belong to the provided schema.
    Violations mean the LLM ignored schema constraints.
    Reference: run.inputs["schema"].nodes
    """
    gd: GraphDocument = run.outputs["output"]
    schema: Schema = run.inputs["schema"]
    allowed = {t.lower() for t in schema.nodes} | {"chunk"}
    violations = [n.type for n in gd.nodes if n.type.lower() not in allowed]
    score = max(0.0, 1.0 - len(violations) / max(len(gd.nodes), 1))
    return {
        "key": "sde_schema_compliance_nodes",
        "score": score,
        "comment": f"Non-schema node types: {set(violations)}" if violations else "OK",
    }


def eval_sde_schema_compliance_rels(run, example) -> dict:
    """
    All relationship types (except system rels IN_CHUNK, IN_DOCUMENT) must be from schema.
    Reference: run.inputs["schema"].relationships
    """
    gd: GraphDocument = run.outputs["output"]
    schema: Schema = run.inputs["schema"]
    system_rels = {"in_chunk", "in_document"}
    allowed = {t.lower() for t in schema.relationships} | system_rels
    violations = [r.type for r in gd.relationships if r.type.lower() not in allowed]
    score = max(0.0, 1.0 - len(violations) / max(len(gd.relationships), 1))
    return {
        "key": "sde_schema_compliance_rels",
        "score": score,
        "comment": f"Non-schema rel types: {set(violations)}" if violations else "OK",
    }


def eval_sde_referential_integrity(run, example) -> dict:
    """
    Every relationship's source.id and target.id must exist in gd.nodes.
    Broken references mean the LLM hallucinated a node reference in a relationship
    without producing the corresponding Node object.
    """
    gd: GraphDocument = run.outputs["output"]
    node_ids = {n.id for n in gd.nodes}
    broken = [
        (r.source.id, r.type, r.target.id)
        for r in gd.relationships
        if r.source.id not in node_ids or r.target.id not in node_ids
    ]
    score = max(0.0, 1.0 - len(broken) / max(len(gd.relationships), 1))
    return {
        "key": "sde_referential_integrity",
        "score": score,
        "comment": f"Broken refs (first 5): {broken[:5]}" if broken else "OK",
    }


def eval_sde_node_ids_no_diacritics(run, example) -> dict:
    """Node IDs must not contain diacritics (same rule as type names in ODD/REF)."""
    gd: GraphDocument = run.outputs["output"]
    violations = [n.id for n in gd.nodes if _has_diacritics(str(n.id))]
    score = max(0.0, 1.0 - len(violations) / max(len(gd.nodes), 1))
    return {
        "key": "sde_node_ids_no_diacritics",
        "score": score,
        "comment": f"Diacritic IDs (first 5): {violations[:5]}" if violations else "OK",
    }


def eval_sde_in_chunk_coverage(run, example) -> dict:
    """
    Every non-Chunk node must have at least one IN_CHUNK relationship.
    This is added by _convert_to_graph_document — a missing link means
    the conversion logic dropped a node-chunk association.
    """
    gd: GraphDocument = run.outputs["output"]
    non_chunk_ids = {n.id for n in gd.nodes if n.type != "Chunk"}
    nodes_with_in_chunk = {r.source.id for r in gd.relationships if r.type == "IN_CHUNK"}
    missing = non_chunk_ids - nodes_with_in_chunk
    score = max(0.0, 1.0 - len(missing) / max(len(non_chunk_ids), 1))
    return {
        "key": "sde_in_chunk_coverage",
        "score": score,
        "comment": f"Missing IN_CHUNK (first 5): {list(missing)[:5]}" if missing else "OK",
    }



def eval_sde_entity_recall(run, example) -> dict:
    """
    Fraction of expected entity names (from fixture) found among extracted node IDs.
    Uses case-insensitive substring match — entities may be slightly reformatted by the LLM.
    Reference: example.outputs["expected_entities"]
    """
    gd: GraphDocument = run.outputs["output"]
    expected_entities: list = example.outputs.get("expected_entities", [])
    if not expected_entities:
        return {"key": "sde_entity_recall", "score": 1.0, "comment": "No reference"}
    node_ids_blob = " ".join(str(n.id).lower() for n in gd.nodes)
    found = [e for e in expected_entities if e.lower() in node_ids_blob]
    missing = [e for e in expected_entities if e.lower() not in node_ids_blob]
    return {
        "key": "sde_entity_recall",
        "score": len(found) / len(expected_entities),
        "comment": f"Missing entities: {missing}" if missing else "All expected entities found",
    }


# ---------------------------------------------------------------------------
# LangSmith dataset helpers
# ---------------------------------------------------------------------------

def _get_or_create_dataset(name: str, description: str):
    existing = list(ls_client.list_datasets(dataset_name=name))
    if existing:
        return existing[0]
    return ls_client.create_dataset(dataset_name=name, description=description)


def create_odd_dataset():
    ds = _get_or_create_dataset(
        "odd_eval_v1",
        "ODD: curated Slovak legal text + expected schema types"
    )
    if not list(ls_client.list_examples(dataset_id=ds.id)):
        ls_client.create_example(
            inputs={"text": FIXTURE_TEXT},
            outputs={
                "expected_node_types": list(EXPECTED_ODD_NODE_TYPES),
                "expected_rel_types": list(EXPECTED_ODD_REL_TYPES),
            },
            dataset_id=ds.id,
        )
    return ds


def create_ref_dataset():
    ds = _get_or_create_dataset(
        "ref_eval_v1",
        "Refinement: controlled schema with known synonyms and distinct pairs"
    )
    if not list(ls_client.list_examples(dataset_id=ds.id)):
        ls_client.create_example(
            inputs={
                "odd_schema": DUPLICATE_ODD_SCHEMA,
                "existing_schema": None,
            },
            outputs={
                "expected_canonical_nodes": ["Dokument", "PravnyPredpis", "Osoba", "Spolocnost"],
            },
            dataset_id=ds.id,
        )
    return ds


def create_sde_dataset():
    ds = _get_or_create_dataset(
        "sde_eval_v1",
        "SDE: curated text with fixed schema + expected extracted entities"
    )
    if not list(ls_client.list_examples(dataset_id=ds.id)):
        ls_client.create_example(
            inputs={
                "text": FIXTURE_TEXT,
                "schema": Schema(
                    nodes=["PravnyPredpis", "Paragraf", "Platitel", "Ministerstvo", "Spolocnost"],
                    relationships=["OBSAHUJE", "VYDAL", "PODLA", "JE_REGISTROVANY"]
                ),
            },
            outputs={
                # TODO: VERIFY THESE AGAINST ACTUAL LLM OUTPUT.
                # These are the entity strings the LLM is expected to extract
                # from FIXTURE_TEXT and store as node.id values.
                # Run SDE once manually, print [n.id for n in gd.nodes], and
                # adjust the list to match real LLM formatting
                # (e.g. "Zakon 222/2004" vs "Zákon č. 222/2004 Z. z.").
                # The eval uses case-insensitive substring match so minor
                # reformatting is tolerated, but the core token must be present.
                "expected_entities": [
                    "Zakon c. 222/2004 Z. z.",
                    "§ 54 ods. 2 pism. a)",
                    "Ministerstvo financii",
                    "ABC s.r.o.",
                ],
            },
            dataset_id=ds.id,
        )
    return ds


# ---------------------------------------------------------------------------
# Target functions — called by LangSmith evaluate() per example
# ---------------------------------------------------------------------------

def _make_rag():
    from pdf_graphrag import PDFGraphRAG
    return PDFGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def run_odd(inputs: dict) -> dict:
    rag = _make_rag()
    doc = Document(page_content=inputs["text"])
    schema: Schema = asyncio.run(rag.open_domain_detection(0, doc))
    return {"output": schema}


def run_refinement(inputs: dict) -> dict:
    rag = _make_rag()
    result = rag.schema_refinement(
        odd_schema=inputs["odd_schema"],
        existing_schema=inputs.get("existing_schema"),
    )
    return {"output": result}


def run_sde(inputs: dict) -> dict:
    rag = _make_rag()
    doc = Document(page_content=inputs["text"])
    graph_doc: GraphDocument = asyncio.run(
        rag.schema_driven_extraction(0, doc, inputs["schema"])
    )
    return {"output": graph_doc}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Setting up LangSmith datasets...")
    odd_ds = create_odd_dataset()
    ref_ds = create_ref_dataset()
    sde_ds = create_sde_dataset()

    print("\n--- ODD evaluation ---")
    odd_results = evaluate(
        run_odd,
        data=odd_ds.name,
        evaluators=[
            eval_odd_no_diacritics,
            eval_odd_node_format,
            eval_odd_rel_format,
            eval_odd_not_empty,
            eval_odd_recall,
            eval_odd_precision,
            eval_odd_no_instances,
        ],
        experiment_prefix="odd",
        client=ls_client,
    )
    print(odd_results)

    print("\n--- Refinement evaluation ---")
    ref_results = evaluate(
        run_refinement,
        data=ref_ds.name,
        evaluators=[
            eval_ref_paragraf_present,
            eval_ref_no_new_types,
            eval_ref_merge_log_consistency,
            eval_ref_no_diacritics,
            eval_ref_distinct_not_merged,
        ],
        experiment_prefix="refinement",
        client=ls_client,
    )
    print(ref_results)

    print("\n--- SDE evaluation ---")
    sde_results = evaluate(
        run_sde,
        data=sde_ds.name,
        evaluators=[
            eval_sde_schema_compliance_nodes,
            eval_sde_schema_compliance_rels,
            eval_sde_referential_integrity,
            eval_sde_node_ids_no_diacritics,
            eval_sde_in_chunk_coverage,
            eval_sde_entity_recall,
        ],
        experiment_prefix="sde",
        client=ls_client,
    )
    print(sde_results)
