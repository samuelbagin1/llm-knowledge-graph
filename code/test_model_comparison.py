import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_text_splitters import RecursiveCharacterTextSplitter

from classes import Schema
from pdf_graphrag import PDFGraphRAG
from to_json import odd_to_json, refinement_to_json, sde_to_json

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5.4", "gpt-5.4-mini"]

PDF_PATH = "./code/assets/ZZ_2004_222_20260101.pdf"
PAGE_START = 40
PAGE_END = 60  # exclusive — processes pages 40–59 (20 pages)

ODD_CHUNK_SIZE = 1200
ODD_CHUNK_OVERLAP = 200
SDE_CHUNK_SIZE = 1024
SDE_CHUNK_OVERLAP = 128

COOLDOWN_SECONDS = 30
OUTPUT_DIR = "./extracted_data"

# Internal node types injected by the pipeline (not extracted by the model)
PIPELINE_INJECTED_TYPES = {"Chunk", "Document"}


# ─── Metric collection ────────────────────────────────────────────────────────

def collect_odd_metrics(schema_list: list, elapsed: float, total_chunks: int) -> dict:
    node_counts = [len(s.nodes) for s in schema_list if s is not None]
    rel_counts = [len(s.relationships) for s in schema_list if s is not None]
    all_nodes = [n for s in schema_list if s is not None for n in s.nodes]
    all_rels = [r for s in schema_list if s is not None for r in s.relationships]
    unique_nodes = sorted(set(all_nodes))
    unique_rels = sorted(set(all_rels))
    failed = total_chunks - len(node_counts)

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "time_seconds": round(elapsed, 2),
        "total_chunks": total_chunks,
        "successful_chunks": len(node_counts),
        "failed_chunks": failed,
        "unique_node_types_count": len(unique_nodes),
        "unique_rel_types_count": len(unique_rels),
        "unique_node_types": unique_nodes,
        "unique_rel_types": unique_rels,
        "total_node_type_mentions": len(all_nodes),
        "total_rel_type_mentions": len(all_rels),
        "avg_node_types_per_chunk": safe_mean(node_counts),
        "avg_rel_types_per_chunk": safe_mean(rel_counts),
        "min_node_types_per_chunk": min(node_counts) if node_counts else 0,
        "max_node_types_per_chunk": max(node_counts) if node_counts else 0,
        "min_rel_types_per_chunk": min(rel_counts) if rel_counts else 0,
        "max_rel_types_per_chunk": max(rel_counts) if rel_counts else 0,
    }


def collect_refinement_metrics(refined_data: dict, odd_metrics: dict, elapsed: float) -> dict:
    node_types = refined_data.get("node_types", [])
    rel_types = refined_data.get("relationship_types", [])
    merge_log = refined_data.get("merge_log", {})

    odd_unique_nodes = odd_metrics["unique_node_types_count"]
    odd_unique_rels = odd_metrics["unique_rel_types_count"]

    compression_nodes = round(odd_unique_nodes / len(node_types), 2) if node_types else 0.0
    compression_rels = round(odd_unique_rels / len(rel_types), 2) if rel_types else 0.0

    return {
        "time_seconds": round(elapsed, 2),
        "refined_node_types_count": len(node_types),
        "refined_rel_types_count": len(rel_types),
        "refined_node_types": sorted(node_types),
        "refined_rel_types": sorted(rel_types),
        "compression_ratio_nodes": compression_nodes,
        "compression_ratio_rels": compression_rels,
        "merge_groups_nodes": len(merge_log.get("node_types", {})),
        "merge_groups_rels": len(merge_log.get("relationship_types", {})),
    }


def collect_sde_metrics(graph_docs: list, refined_schema: Schema, elapsed: float, total_chunks: int) -> dict:
    # Exclude pipeline-injected nodes from analysis
    entity_nodes = [
        n for gd in graph_docs
        for n in gd.nodes
        if n.type not in PIPELINE_INJECTED_TYPES
    ]
    all_rels = [r for gd in graph_docs for r in gd.relationships]

    node_counts_per_chunk = [
        len([n for n in gd.nodes if n.type not in PIPELINE_INJECTED_TYPES])
        for gd in graph_docs
    ]
    rel_counts_per_chunk = [len(gd.relationships) for gd in graph_docs]
    failed = total_chunks - len(graph_docs)

    unique_node_ids = set(n.id for n in entity_nodes)
    unique_node_types_used = set(n.type for n in entity_nodes)
    unique_rel_types_used = set(r.type for r in all_rels if r.type != "IN_CHUNK" and r.type != "IN_DOCUMENT")

    schema_node_set = set(refined_schema.nodes)
    schema_rel_set = set(refined_schema.relationships)

    off_schema_nodes = sorted(unique_node_types_used - schema_node_set)
    off_schema_rels = sorted(unique_rel_types_used - schema_rel_set)

    total_types_used = len(unique_node_types_used) + len(unique_rel_types_used)
    off_schema_total = len(off_schema_nodes) + len(off_schema_rels)
    adherence_score = round(1 - (off_schema_total / total_types_used), 4) if total_types_used else 1.0

    schema_node_coverage = round(len(unique_node_types_used & schema_node_set) / len(schema_node_set), 4) if schema_node_set else 0.0
    schema_rel_coverage = round(len(unique_rel_types_used & schema_rel_set) / len(schema_rel_set), 4) if schema_rel_set else 0.0

    nodes_with_props = sum(1 for n in entity_nodes if n.properties)
    rels_with_props = sum(1 for r in all_rels if r.properties)

    node_type_dist = dict(Counter(n.type for n in entity_nodes).most_common())
    rel_type_dist = dict(Counter(r.type for r in all_rels).most_common())

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "time_seconds": round(elapsed, 2),
        "total_chunks": total_chunks,
        "successful_chunks": len(graph_docs),
        "failed_chunks": failed,
        "total_nodes": len(entity_nodes),
        "total_relationships": len(all_rels),
        "unique_node_ids": len(unique_node_ids),
        "unique_node_types_used": sorted(unique_node_types_used),
        "unique_rel_types_used": sorted(unique_rel_types_used),
        "schema_node_coverage": schema_node_coverage,
        "schema_rel_coverage": schema_rel_coverage,
        "avg_nodes_per_chunk": safe_mean(node_counts_per_chunk),
        "avg_rels_per_chunk": safe_mean(rel_counts_per_chunk),
        "min_nodes_per_chunk": min(node_counts_per_chunk) if node_counts_per_chunk else 0,
        "max_nodes_per_chunk": max(node_counts_per_chunk) if node_counts_per_chunk else 0,
        "min_rels_per_chunk": min(rel_counts_per_chunk) if rel_counts_per_chunk else 0,
        "max_rels_per_chunk": max(rel_counts_per_chunk) if rel_counts_per_chunk else 0,
        "nodes_with_properties_pct": round(nodes_with_props / len(entity_nodes), 4) if entity_nodes else 0.0,
        "rels_with_properties_pct": round(rels_with_props / len(all_rels), 4) if all_rels else 0.0,
        "node_type_distribution": node_type_dist,
        "rel_type_distribution": rel_type_dist,
        "off_schema_node_types": off_schema_nodes,
        "off_schema_rel_types": off_schema_rels,
        "schema_adherence_score": adherence_score,
    }


def compute_derived_metrics(odd_m: dict, ref_m: dict, sde_m: dict) -> dict:
    total_time = odd_m["time_seconds"] + ref_m["time_seconds"] + sde_m["time_seconds"]
    sde_chunks = sde_m["total_chunks"]
    unique_ids = sde_m["unique_node_ids"]

    density_nodes = round(sde_m["total_nodes"] / sde_chunks, 2) if sde_chunks else 0.0
    density_rels = round(sde_m["total_relationships"] / sde_chunks, 2) if sde_chunks else 0.0
    connectedness = round(sde_m["total_relationships"] / unique_ids, 2) if unique_ids else 0.0

    return {
        "total_pipeline_time_seconds": round(total_time, 2),
        "extraction_density_nodes_per_chunk": density_nodes,
        "extraction_density_rels_per_chunk": density_rels,
        "graph_connectedness": connectedness,
    }


# ─── Summary printing ─────────────────────────────────────────────────────────

def print_summary(results: dict):
    models = list(results.keys())
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    rows = [
        ("ODD time (s)",         lambda m: results[m]["odd"]["time_seconds"]),
        ("ODD unique node types", lambda m: results[m]["odd"]["unique_node_types_count"]),
        ("ODD unique rel types",  lambda m: results[m]["odd"]["unique_rel_types_count"]),
        ("Refined node types",    lambda m: results[m]["refinement"]["refined_node_types_count"]),
        ("Refined rel types",     lambda m: results[m]["refinement"]["refined_rel_types_count"]),
        ("Compression ratio (nodes)", lambda m: results[m]["refinement"]["compression_ratio_nodes"]),
        ("SDE time (s)",          lambda m: results[m]["sde"]["time_seconds"]),
        ("SDE total nodes",       lambda m: results[m]["sde"]["total_nodes"]),
        ("SDE total rels",        lambda m: results[m]["sde"]["total_relationships"]),
        ("SDE unique entity IDs", lambda m: results[m]["sde"]["unique_node_ids"]),
        ("Schema node coverage",  lambda m: results[m]["sde"]["schema_node_coverage"]),
        ("Schema rel coverage",   lambda m: results[m]["sde"]["schema_rel_coverage"]),
        ("Schema adherence score",lambda m: results[m]["sde"]["schema_adherence_score"]),
        ("Graph connectedness",   lambda m: results[m]["derived"]["graph_connectedness"]),
        ("Total pipeline time (s)",lambda m: results[m]["derived"]["total_pipeline_time_seconds"]),
    ]

    col_w = 20
    header = f"{'Metric':<35}" + "".join(f"{m:<{col_w}}" for m in models)
    print(header)
    print("-" * (35 + col_w * len(models)))
    for label, fn in rows:
        values = []
        for m in models:
            try:
                values.append(str(fn(m)))
            except (KeyError, TypeError):
                values.append("N/A")
        print(f"{label:<35}" + "".join(f"{v:<{col_w}}" for v in values))
    print("=" * 80 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    graphrag = PDFGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "fseijkfbsj48@"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Load and slice PDF once — same input for all models
    all_docs = graphrag.load_pdf(PDF_PATH)
    documents = all_docs[PAGE_START:PAGE_END]
    print(f"Loaded {len(documents)} pages (pages {PAGE_START}–{PAGE_END - 1})")

    # Chunk once — deterministic, reused across all models
    odd_splitter = RecursiveCharacterTextSplitter(chunk_size=ODD_CHUNK_SIZE, chunk_overlap=ODD_CHUNK_OVERLAP)
    sde_splitter = RecursiveCharacterTextSplitter(chunk_size=SDE_CHUNK_SIZE, chunk_overlap=SDE_CHUNK_OVERLAP)
    odd_chunks = odd_splitter.split_documents(documents)
    sde_chunks = sde_splitter.split_documents(documents)
    print(f"ODD chunks: {len(odd_chunks)}  |  SDE chunks: {len(sde_chunks)}\n")

    all_results = {}

    for i, model_name in enumerate(MODELS):
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model_name}  ({i + 1}/{len(MODELS)})")
        print(f"{'=' * 60}")

        try:
            # Swap the extraction model
            graphrag.openai_graph_transform = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
                max_retries=3,
                timeout=120,
            )

            # Stage 1: ODD
            print(f"\n[{model_name}] Stage 1: Open Domain Detection...")
            t0 = time.time()
            schema_list = asyncio.run(graphrag.async_open_domain_detection(odd_chunks))
            odd_elapsed = time.time() - t0
            odd_m = collect_odd_metrics(schema_list, odd_elapsed, len(odd_chunks))
            odd_to_json(schema_list, output_dir=OUTPUT_DIR, name=f"{model_name}")
            print(f"  Done in {odd_elapsed:.1f}s — {odd_m['unique_node_types_count']} node types, {odd_m['unique_rel_types_count']} rel types")

            # Stage 2: Schema Refinement (Gemini)
            print(f"\n[{model_name}] Stage 2: Schema Refinement (Gemini)...")
            merged_schema = Schema(
                nodes=list(set(n for s in schema_list if s is not None for n in s.nodes)),
                relationships=list(set(r for s in schema_list if s is not None for r in s.relationships)),
            )
            t0 = time.time()
            refined_data = graphrag.schema_refinement(odd_schema=merged_schema)
            ref_elapsed = time.time() - t0
            refined_dict = refined_data if isinstance(refined_data, dict) else refined_data.model_dump()
            ref_m = collect_refinement_metrics(refined_dict, odd_m, ref_elapsed)
            refinement_to_json(refined_dict, output_dir=OUTPUT_DIR, name=f"{model_name}")
            refined_schema = graphrag._convert_to_schema(refined_dict)
            print(f"  Done in {ref_elapsed:.1f}s — {ref_m['refined_node_types_count']} node types, {ref_m['refined_rel_types_count']} rel types (compression: {ref_m['compression_ratio_nodes']}x nodes)")

            # Stage 3: SDE
            print(f"\n[{model_name}] Stage 3: Schema Driven Extraction...")
            t0 = time.time()
            graph_docs = asyncio.run(graphrag.async_schema_driven_extraction(sde_chunks, schema=refined_schema))
            sde_elapsed = time.time() - t0
            sde_m = collect_sde_metrics(graph_docs, refined_schema, sde_elapsed, len(sde_chunks))
            sde_to_json(graph_docs, output_dir=OUTPUT_DIR, name=f"{model_name}")
            print(f"  Done in {sde_elapsed:.1f}s — {sde_m['total_nodes']} nodes, {sde_m['total_relationships']} rels, adherence={sde_m['schema_adherence_score']}")

            derived_m = compute_derived_metrics(odd_m, ref_m, sde_m)

            all_results[model_name] = {
                "odd": odd_m,
                "refinement": ref_m,
                "sde": sde_m,
                "derived": derived_m,
            }

        except Exception as e:
            print(f"\n[{model_name}] FAILED: {e}")
            all_results[model_name] = {"error": str(e)}

        # Cooldown between models (skip after last)
        if i < len(MODELS) - 1:
            print(f"\nCooldown {COOLDOWN_SECONDS}s before next model...")
            time.sleep(COOLDOWN_SECONDS)

    # Save comparison JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comparison = {
        "metadata": {
            "pdf": PDF_PATH,
            "pages": f"{PAGE_START}–{PAGE_END - 1}",
            "page_count": PAGE_END - PAGE_START,
            "odd_chunk_config": {"size": ODD_CHUNK_SIZE, "overlap": ODD_CHUNK_OVERLAP},
            "sde_chunk_config": {"size": SDE_CHUNK_SIZE, "overlap": SDE_CHUNK_OVERLAP},
            "odd_chunk_count": len(odd_chunks),
            "sde_chunk_count": len(sde_chunks),
            "refinement_model": "gemini-2.5-pro",
            "timestamp": ts,
            "models_tested": MODELS,
        },
        "results": all_results,
    }
    out_path = os.path.join(OUTPUT_DIR, f"model_comparison_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to: {out_path}")

    # Print summary table
    successful = {k: v for k, v in all_results.items() if "error" not in v}
    if successful:
        print_summary(successful)


if __name__ == "__main__":
    main()
