"""
Query-quality evaluation for PDFGraphRAG against three published benchmarks:

  - CoLoTa  (ARK-V1, arXiv 2509.18063)   — binary yes/no, 200 items x 30 runs
  - FactKG  (KG-GPT, EMNLP-F 2023)       — binary Supported/Refuted, full test set
  - MetaQA  (KG-GPT)                      — entity QA, Hits@1, 1/2/3-hop splits

Prerequisites:
  - .env with NEO4J_URI, NEO4J_USERNAME (or NEO4J_USER), NEO4J_PASSWORD,
    OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
  - Three Neo4j databases pre-populated: colota, factkg, metaqa
  - Dataset files under tests/datasets/data/:
      CoLoTa_qa.json
      factkg_test.json
      1-hop/vanilla/qa_test.txt
      2-hop/vanilla/qa_test.txt
      3-hop/vanilla/qa_test.txt

Run from repo root:
  python tests/test_query_quality.py [colota|factkg|metaqa|all]

Env vars:
  COLOTA_RUNS=30   number of stochastic runs (default 30, set to 1 for smoke test)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, "./code")
sys.path.insert(0, "./tests")

from dotenv import load_dotenv

import datasets.colota_loader as colota_loader
import datasets.factkg_loader as factkg_loader
import datasets.metaqa_loader as metaqa_loader
import datasets.metrics as metrics
from pdf_graphrag import PDFGraphRAG


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_rag(database: str) -> PDFGraphRAG:
    """Create a PDFGraphRAG instance pointing at the given Neo4j database."""
    return PDFGraphRAG(
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_user=os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        google_api_key=os.environ["GOOGLE_API_KEY"],
        claude_api_key=os.environ["ANTHROPIC_API_KEY"],
        database=database,
    )


def _dump_results(name: str, summary: dict, preds, gold) -> None:
    """Write results JSON to tests/datasets/results/<name>_<timestamp>.json."""
    out = Path("tests/datasets/results") / f"{name}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "preds": preds, "gold": gold},
                   default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to {out}")


# ---------------------------------------------------------------------------
# CoLoTa — binary yes/no QA over long-tail Wikidata entities (ARK-V1)
#
# Metrics (ARK-V1 Table 2):
#   answer_rate:           fraction of items with a non-None prediction
#   conditional_accuracy:  accuracy among answered items
#   overall_accuracy:      accuracy over all items (None = wrong)
#   reliability:           entropy-based consistency across 30 runs (1 = stable)
# ---------------------------------------------------------------------------

def test_colota() -> None:
    load_dotenv()
    data_path = Path("tests/datasets/data/CoLoTa_qa.json")
    items = colota_loader.load_colota_items(data_path)[:200]
    print(f"CoLoTa: {len(items)} items loaded")

    rag = _make_rag(database="colota")
    num_runs = int(os.getenv("COLOTA_RUNS", "30"))
    all_runs: List[List[Optional[bool]]] = []

    for run_idx in range(num_runs):
        preds: List[Optional[bool]] = []
        for item in items:
            raw = rag.query(item.query, verbose=False)
            preds.append(colota_loader.parse_answer(raw))
        all_runs.append(preds)
        m = metrics.colota_metrics(preds, [i.answer for i in items])
        print(f"Run {run_idx + 1}/{num_runs}  overall_acc={m['overall_accuracy']:.3f}  "
              f"cond_acc={m['conditional_accuracy']:.3f}  "
              f"answer_rate={m['answer_rate']:.3f}")

    gold = [i.answer for i in items]
    summary = metrics.colota_metrics(all_runs[-1], gold)
    summary["reliability"] = metrics.colota_reliability(all_runs)
    _dump_results("colota", summary, all_runs, gold)
    print("\n=== CoLoTa results ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# FactKG — binary Supported/Refuted fact verification over DBpedia (KG-GPT)
#
# Metrics (KG-GPT Table 3):
#   accuracy:    overall binary accuracy
#   by_pattern:  accuracy per reasoning pattern (num1, multi hop, existence,
#                negation, num2) — matches the paper's Table 3 breakdown
# ---------------------------------------------------------------------------

def test_factkg() -> None:
    load_dotenv()
    data_path = Path("tests/datasets/data/factkg_test.json")
    items = factkg_loader.load_factkg_test(data_path)
    print(f"FactKG: {len(items)} items loaded")

    rag = _make_rag(database="factkg")
    preds: List[Optional[bool]] = []
    gold: List[bool] = []
    strata: List[str] = []

    for i, item in enumerate(items):
        raw = rag.query(item.claim, verbose=False)
        pred = factkg_loader.parse_answer(raw)
        preds.append(pred)
        gold.append(item.label)
        strata.append(item.types[0] if item.types else "other")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}  "
                  f"acc_so_far={metrics.binary_accuracy(preds, gold):.3f}")

    summary = {
        "accuracy": metrics.binary_accuracy(preds, gold),
        "by_pattern": metrics.stratified_accuracy(preds, gold, strata),
    }
    _dump_results("factkg", summary, preds, gold)
    print("\n=== FactKG results ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# MetaQA — entity QA over a movies KG, Hits@1 per hop (KG-GPT)
#
# Metrics (KG-GPT Table 4):
#   hop_1_hits@1, hop_2_hits@1, hop_3_hits@1
#   (KG-GPT 12-shot baselines: 0.963 / 0.944 / 0.940)
# ---------------------------------------------------------------------------

def test_metaqa() -> None:
    load_dotenv()
    rag = _make_rag(database="metaqa")

    hop_files = {
        1: Path("tests/datasets/data/1-hop/vanilla/qa_test.txt"),
        2: Path("tests/datasets/data/2-hop/vanilla/qa_test.txt"),
        3: Path("tests/datasets/data/3-hop/vanilla/qa_test.txt"),
    }
    summary: dict = {}

    for hop, path in hop_files.items():
        items = metaqa_loader.load_qa(path)
        print(f"MetaQA {hop}-hop: {len(items)} items")
        hits = 0
        for i, item in enumerate(items):
            raw = rag.query(item.question, verbose=False)
            pred = metaqa_loader.parse_answer(raw, item.answers)
            hits += metrics.hits_at_1(pred, item.answers)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(items)}  hits@1_so_far={hits / (i + 1):.3f}")
        h1 = hits / len(items) if items else 0.0
        summary[f"hop_{hop}_hits@1"] = h1
        print(f"{hop}-hop hits@1: {h1:.3f}")

    _dump_results("metaqa", summary, None, None)
    print("\n=== MetaQA results ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "colota"):
        test_colota()
    if which in ("all", "factkg"):
        test_factkg()
    if which in ("all", "metaqa"):
        test_metaqa()
