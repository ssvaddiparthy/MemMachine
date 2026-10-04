"""Benchmark runner for the procedural memory tier.

Integrates the procedural memory modules (graph store, procedure extractor,
community detector, compositional retriever) with the existing proced_mem_bench
evaluation harness.

Usage:
    # Stage 2: Ingest trajectories into the procedure graph
    python proced_mem_bench_procedural_ingest.py \
        --data-path ../data/proced_mem_bench/trajectories.json \
        --config-path ../retrieval_agent/configuration.yml

    # Stage 3: Run Louvain community detection
    python proced_mem_bench_procedural_ingest.py \
        --data-path ../data/proced_mem_bench/trajectories.json \
        --config-path ../retrieval_agent/configuration.yml \
        --detect-communities

    # Stage 4: Run compositional retrieval on benchmark queries
    python proced_mem_bench_procedural_ingest.py \
        --data-path ../data/proced_mem_bench/queries.json \
        --config-path ../retrieval_agent/configuration.yml \
        --mode search
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = [
    REPO_ROOT,
    REPO_ROOT / "packages" / "common" / "src",
    REPO_ROOT / "packages" / "server" / "src",
    REPO_ROOT / "packages" / "client" / "src",
]
for package_root in PACKAGE_ROOTS:
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.append(package_root_str)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Procedural memory benchmark integration",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to trajectories.json (ingest) or queries.json (search)",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "detect", "search", "delete", "stats"],
        default="ingest",
        help="Operation mode",
    )
    parser.add_argument(
        "--detect-communities",
        action="store_true",
        help="Also run community detection after ingest",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution parameter (default: 1.0)",
    )
    parser.add_argument(
        "--result-path",
        default="result/procedural_search_output.json",
        help="Path to save search results",
    )
    return parser


async def get_neo4j_driver(config_path: str):
    """Get Neo4j driver from MemMachine's config."""
    from evaluation.utils import agent_utils

    resource_manager = agent_utils.load_eval_config(config_path)
    # Access the Neo4j driver from the database manager
    db_manager = resource_manager._database_manager
    # Get the first available Neo4j driver
    for storage_id, driver in db_manager.neo4j_drivers.items():
        logger.info("Using Neo4j driver: %s", storage_id)
        return driver

    # If no driver exists yet, create it
    for storage_id in db_manager.conf.neo4j_confs:
        driver = await db_manager.async_get_neo4j_driver(storage_id)
        return driver

    raise RuntimeError("No Neo4j driver available")


async def run_ingest(args) -> None:
    """Stage 2: Ingest trajectories into the procedure graph."""
    from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore
    from memmachine_server.procedural_memory.procedure_extractor import ProcedureExtractor

    driver = await get_neo4j_driver(args.config_path)
    store = ProceduralGraphStore(driver)
    extractor = ProcedureExtractor(store)

    # Ensure schema
    await store.ensure_schema()

    # Load trajectory data
    with open(args.data_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        trajectories = data.get("trajectories", [])
    elif isinstance(data, list):
        trajectories = data
    else:
        raise TypeError(f"Unexpected data format: {type(data)}")

    print(f"Loaded {len(trajectories)} trajectories from {args.data_path}")

    start = time.time()
    results = await extractor.extract_batch(trajectories, default_success=True)
    elapsed = time.time() - start

    print(f"\nIngestion complete:")
    print(f"  Trajectories: {len(results)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Rate: {len(results)/elapsed:.1f} trajectories/sec")

    # Print graph stats
    counts = await store.get_trajectory_count()
    tools = await store.get_tool_names()
    print(f"\nGraph state:")
    print(f"  Tools: {len(tools)} ({', '.join(tools[:10])}{'...' if len(tools) > 10 else ''})")
    print(f"  Trajectories: {counts}")

    if args.detect_communities:
        await run_detect(args, driver, store)


async def run_detect(args, driver=None, store=None) -> None:
    """Stage 3: Run Louvain community detection."""
    from memmachine_server.procedural_memory.community_detector import CommunityDetector
    from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

    if driver is None:
        driver = await get_neo4j_driver(args.config_path)
    if store is None:
        store = ProceduralGraphStore(driver)

    detector = CommunityDetector(store, driver)

    print(f"\nRunning Louvain community detection (resolution={args.resolution})...")
    start = time.time()
    communities = await detector.detect_communities(resolution=args.resolution)
    elapsed = time.time() - start

    print(f"\nCommunity detection complete ({elapsed:.1f}s):")
    print(f"  Total communities: {len(communities)}")
    success_comms = [c for c in communities if c.community_type == "success"]
    failure_comms = [c for c in communities if c.community_type == "failure"]
    print(f"  Success communities: {len(success_comms)}")
    print(f"  Failure communities: {len(failure_comms)}")

    for c in communities:
        prefix = "✅" if c.community_type == "success" else "❌"
        print(f"  {prefix} C{c.community_id}: {c.description} "
              f"(members={c.member_count}, trajectories={c.trajectory_count}, "
              f"avg_weight={c.avg_weight:.3f})")


async def run_search(args) -> None:
    """Stage 4: Run compositional retrieval on benchmark queries."""
    from memmachine_server.procedural_memory.compositional_retriever import CompositionalRetriever
    from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

    from evaluation.procedural_memory.procedure_utils import compute_ir_metrics

    driver = await get_neo4j_driver(args.config_path)
    store = ProceduralGraphStore(driver)
    retriever = CompositionalRetriever(store)

    # Load query data
    with open(args.data_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        queries = data.get("queries", [])
    elif isinstance(data, list):
        queries = data
    else:
        raise TypeError(f"Unexpected data format: {type(data)}")

    print(f"Loaded {len(queries)} queries from {args.data_path}")

    # Get all available tools for capability extraction
    available_tools = await store.get_tool_names()
    print(f"Available tools in graph: {available_tools}")

    results = []
    for query in queries:
        query_id = query.get("query_id", "unknown")
        query_text = query.get("query_text", "")
        tier = query.get("tier", "unknown")

        # Extract required tools from query text (simple keyword matching)
        required_tools = _extract_tools_from_query(query_text, available_tools)

        print(f"  Query {query_id} [{tier}]: {query_text[:60]}...")
        print(f"    Required tools: {required_tools}")

        start = time.time()
        composed = await retriever.compose(required_tools)
        elapsed = time.time() - start

        if composed:
            print(f"    Composed: {composed.tool_sequence} "
                  f"(confidence={composed.confidence:.3f}, cost={composed.total_cost:.3f})")
            result = {
                "query_id": query_id,
                "query_text": query_text,
                "tier": tier,
                "composed_steps": composed.steps,
                "tool_sequence": composed.tool_sequence,
                "confidence": composed.confidence,
                "total_cost": composed.total_cost,
                "avoid_warnings": composed.avoid_warnings,
                "latency_ms": elapsed * 1000,
            }
        else:
            print(f"    No composition found")
            result = {
                "query_id": query_id,
                "query_text": query_text,
                "tier": tier,
                "composed_steps": [],
                "tool_sequence": [],
                "confidence": 0.0,
                "total_cost": 0.0,
                "avoid_warnings": [],
                "latency_ms": elapsed * 1000,
            }

        results.append(result)

    # Save results
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {result_path}")
    print(f"Composed: {sum(1 for r in results if r['tool_sequence'])}/{len(results)} queries")


async def run_stats(args) -> None:
    """Print procedure graph statistics."""
    from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

    driver = await get_neo4j_driver(args.config_path)
    store = ProceduralGraphStore(driver)

    tools = await store.get_tool_names()
    counts = await store.get_trajectory_count()
    edges = await store.get_tool_transition_graph()
    communities = await store.get_communities()

    print("=== Procedure Graph Statistics ===")
    print(f"Tools: {len(tools)} — {', '.join(tools)}")
    print(f"Trajectories: {counts}")
    print(f"Tool transitions: {len(edges)}")
    print(f"Communities: {len(communities)}")

    if edges:
        print("\nTop 10 transitions by weight:")
        for e in edges[:10]:
            print(f"  {e['from_tool']} → {e['to_tool']}: "
                  f"weight={e['weight']:.3f} "
                  f"(success={e['success_count']}, fail={e['fail_count']})")

    if communities:
        print(f"\nCommunities:")
        for c in communities:
            print(f"  C{c['community_id']} ({c['type']}): {c['description']}")


async def run_delete(args) -> None:
    """Delete all procedural memory data from Neo4j."""
    from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

    driver = await get_neo4j_driver(args.config_path)
    store = ProceduralGraphStore(driver)

    counts = await store.delete_all_procedural_data()
    print(f"Deleted procedural memory data: {counts}")


def _extract_tools_from_query(
    query_text: str, available_tools: list[str]
) -> list[str]:
    """Extract required tools from the query by matching against the
    graph's known tool names via keyword heuristics.

    Generic -- works against whatever tools are in the procedure graph,
    not a hardcoded ALFWorld verb list.
    """
    query_lower = query_text.lower()

    generic_keyword_map = {
        "go_to":     ["go to", "navigate to", "move to", "travel to", "get to"],
        "pick_up":   ["pick up", "pick_up", "take", "grab", "retrieve", "collect"],
        "put":       ["put", "place", "set", "deposit", "store"],
        "open":      ["open", "unlock"],
        "close":     ["close", "shut", "lock"],
        "use":       ["use", "activate", "turn on", "apply"],
        "examine":   ["examine", "inspect", "look at", "check", "observe"],
        "send":      ["send", "submit", "post", "email"],
        "search":    ["search", "find", "look up", "query"],
        "read":      ["read", "load", "fetch", "get"],
        "write":     ["write", "save", "create", "update"],
        "delete":    ["delete", "remove", "clear"],
        "execute":   ["execute", "run", "call", "invoke"],
        "clean":     ["clean", "wash", "rinse"],
        "heat":      ["heat", "warm", "microwave"],
        "cool":      ["cool", "chill", "refrigerate"],
    }

    required = []
    for tool_segment, keywords in generic_keyword_map.items():
        matching_tools = [
            t for t in available_tools
            if tool_segment in t or t in tool_segment
        ]
        if not matching_tools:
            continue
        for kw in keywords:
            if kw in query_lower:
                best = tool_segment if tool_segment in available_tools else matching_tools[0]
                if best not in required:
                    required.append(best)
                break

    if required and "go_to" in available_tools and "go_to" not in required:
        required.insert(0, "go_to")

    return required


async def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "ingest":
        await run_ingest(args)
    elif args.mode == "detect":
        await run_detect(args)
    elif args.mode == "search":
        await run_search(args)
    elif args.mode == "stats":
        await run_stats(args)
    elif args.mode == "delete":
        await run_delete(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
