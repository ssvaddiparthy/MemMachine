"""Delete all proced_mem_bench trajectory data from MemMachine.

Removes all episodes ingested under the proced_mem_bench session prefix,
giving a clean slate for re-runs with different configurations.

Usage:
    python proced_mem_bench_delete.py \
        --data-path ../data/proced_mem_bench/trajectories.json \
        --config-path configuration.yml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

BENCHMARK_SESSION_PREFIX = "proced_mem_bench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete proced_mem_bench data from MemMachine",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to AgentInstruct trajectory JSON (to enumerate session IDs)",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    return parser


async def main() -> None:
    from evaluation.utils import agent_utils

    args = build_parser().parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Extract trajectory list (same logic as ingest)
    if isinstance(data, dict):
        trajectories = data.get("trajectories", [])
    elif isinstance(data, list):
        trajectories = data
    else:
        raise TypeError(f"Unexpected data format: {type(data)}")

    resource_manager = agent_utils.load_eval_config(args.config_path)

    # All trajectories are ingested into a single shared session,
    # so one delete call clears everything.
    memory, _, _, _, _ = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id=BENCHMARK_SESSION_PREFIX,
    )
    await memory.delete_session_episodes()
    print(
        f"Deleted all episodes for session '{BENCHMARK_SESSION_PREFIX}' "
        f"({len(trajectories)} trajectories)."
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
