"""
Regenerate summary.json from an existing by_chunk.csv.

The CSV stores per-chunk precision/recall/F1 but not the raw TP/FP/FN counts
that aggregate() needs for micro-averaging. Those integers can be recovered
exactly because precision = TP / pred_count and pred_count is an integer.

Two macro fields cannot be recovered from the CSV and are reported as
"unavailable_macros" in the summary rather than silently defaulting to 0.0:
  - strict_node_f1
  - node_type_accuracy_on_matches

Usage:
    python summarize_from_csv.py                       # uses default paths
    python summarize_from_csv.py path/to/by_chunk.csv  # override input
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from evaluation import aggregate


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "evaluation_results" / "by_chunk.csv"

UNAVAILABLE_MACROS = ("strict_node_f1", "node_type_accuracy_on_matches")


def reconstruct_counts(pred: int, gold: int, precision: float, recall: float) -> tuple[int, int, int]:
    """Recover (TP, FP, FN) from precision/recall and the integer pred/gold totals.

    precision = TP / pred  →  TP = round(precision * pred)
    recall    = TP / gold  →  TP = round(recall * gold)
    Both formulas should agree exactly. If they don't (rare float edge case),
    we trust the precision-derived value and emit a warning to stderr.
    """
    if pred == 0 and gold == 0:
        return 0, 0, 0

    tp_from_p = round(precision * pred) if pred > 0 else None
    tp_from_r = round(recall * gold) if gold > 0 else None

    if tp_from_p is not None and tp_from_r is not None and tp_from_p != tp_from_r:
        print(
            f"[warn] TP mismatch: precision*pred={tp_from_p} vs recall*gold={tp_from_r}; "
            f"using precision-derived value",
            file=sys.stderr,
        )

    tp = tp_from_p if tp_from_p is not None else tp_from_r
    tp = max(0, min(tp, pred, gold))
    return tp, pred - tp, gold - tp


def row_from_csv(raw: dict[str, str]) -> dict[str, Any]:
    """Convert one CSV row into the dict shape aggregate() expects."""
    gold_nodes = int(float(raw["gold_nodes"]))
    pred_nodes = int(float(raw["pred_nodes"]))
    gold_rels = int(float(raw["gold_relationships"]))
    pred_rels = int(float(raw["pred_relationships"]))

    node_tp, node_fp, node_fn = reconstruct_counts(
        pred_nodes,
        gold_nodes,
        float(raw["node_precision"]),
        float(raw["node_recall"]),
    )
    rel_tp, rel_fp, rel_fn = reconstruct_counts(
        pred_rels,
        gold_rels,
        float(raw["relationship_precision"]),
        float(raw["relationship_recall"]),
    )

    ont_nodes = float(raw["ontology_conformance_nodes"])
    ont_rels = float(raw["ontology_conformance_relationships"])

    return {
        "chunk": int(float(raw["chunk"])),
        "page": raw.get("page"),
        "gold_nodes": gold_nodes,
        "pred_nodes": pred_nodes,
        "node_tp": node_tp,
        "node_fp": node_fp,
        "node_fn": node_fn,
        "gold_relationships": gold_rels,
        "pred_relationships": pred_rels,
        "relationship_tp": rel_tp,
        "relationship_fp": rel_fp,
        "relationship_fn": rel_fn,
        "strict_relationship_f1": float(raw["strict_relationship_f1"]),
        "ontology_conformance_nodes": ont_nodes,
        "ontology_conformance_relationships": ont_rels,
        "invalid_node_type_rate": 1.0 - ont_nodes,
        "invalid_relationship_type_rate": 1.0 - ont_rels,
        "dangling_edge_rate": float(raw["dangling_edge_rate"]),
        "unknown_type_rate": float(raw["unknown_type_rate"]),
        "bad_id_format_rate": float(raw["bad_id_format_rate"]),
        "normalized_ged": float(raw["normalized_ged"]),
        "latency_seconds": float(raw["latency_seconds"]),
    }


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw or not raw.get("chunk"):
                continue
            rows.append(row_from_csv(raw))
    return rows


def main() -> None:
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows parsed from {csv_path}")

    summary = aggregate(rows)

    for key in UNAVAILABLE_MACROS:
        summary.pop(f"macro_{key}", None)
    summary["unavailable_macros"] = list(UNAVAILABLE_MACROS)

    latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    if latencies:
        summary["latency_seconds_total"] = sum(latencies)
        summary["latency_seconds_mean"] = sum(latencies) / len(latencies)

    out_path = csv_path.parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
