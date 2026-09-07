"""Ingest AgentInstruct trajectories into MemMachine episodic memory.

Baseline phase: trajectories are stored as standard episodic entries using
MemMachine's existing memory.add() API. No procedure graph, no special schema.

Usage:
    python proced_mem_bench_ingest.py \
        --data-path ../data/proced_mem_bench/trajectories.json \
        --config-path configuration.yml \
        --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


DEFAULT_CONCURRENCY = 5
BENCHMARK_SESSION_PREFIX = "proced_mem_bench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest AgentInstruct trajectories into MemMachine episodic memory",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to AgentInstruct trajectory JSON file",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to configuration.yml",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent ingestion tasks (default: {DEFAULT_CONCURRENCY})",
    )
    return parser


def trajectory_to_episodes(
    trajectory: dict,
    trajectory_id: str,
    session_id: str,
    base_time: datetime,
) -> list:
    """Convert a single AgentInstruct trajectory into MemMachine Episode objects.

    Each state-action pair in the trajectory becomes one episode.
    The task description becomes the first episode to provide context.

    AgentInstruct trajectory format (from qpiai/Proced_mem_bench):
        {
            "task_instance_id": "alfworld_0",
            "task_description": "find two laptop and put them in bed.",
            "state_action_pairs": [
                {"step_id": 1, "state": "You are in...", "action": "go to table 1"},
                ...
            ],
            "total_steps": 14,
            "source": "agentinstruct"
        }

    Args:
        trajectory: dict with keys above.
        trajectory_id: unique ID for this trajectory (task_instance_id).
        session_id: MemMachine session key grouping this trajectory.
        base_time: timestamp for the first episode; subsequent episodes
                   are offset by 1 second each.

    Returns:
        List of Episode objects ready for memory.add_memory_episodes().
    """
    from memmachine_server.common.episode_store import Episode

    episodes = []

    # First episode: the task description
    task_desc = trajectory.get("task_description", "")
    if task_desc:
        episodes.append(
            Episode(
                uid=str(uuid4()),
                content=f"[TASK] {task_desc}",
                session_key=session_id,
                created_at=base_time,
                producer_id="benchmark",
                producer_role="system",
                metadata={
                    "trajectory_id": trajectory_id,
                    "benchmark": BENCHMARK_SESSION_PREFIX,
                    "episode_type": "task_description",
                },
            )
        )

    # Subsequent episodes: state-action pairs
    state_action_pairs = trajectory.get("state_action_pairs", [])
    for step_idx, step in enumerate(state_action_pairs):
        step_id = step.get("step_id", step_idx + 1)
        state_text = step.get("state", "")
        action_text = step.get("action", "")

        content = f"[STATE] {state_text}\n[ACTION] {action_text}"

        episodes.append(
            Episode(
                uid=str(uuid4()),
                content=content,
                session_key=session_id,
                created_at=base_time + timedelta(seconds=step_idx + 1),
                producer_id="agent",
                producer_role="agent",
                metadata={
                    "trajectory_id": trajectory_id,
                    "benchmark": BENCHMARK_SESSION_PREFIX,
                    "episode_type": "state_action",
                    "step_id": step_id,
                },
            )
        )

    return episodes


async def ingest_trajectory(
    trajectory: dict,
    trajectory_idx: int,
    resource_manager: object,
    base_time: datetime,
) -> int:
    """Ingest a single trajectory into MemMachine episodic memory.

    Returns:
        Number of episodes ingested.
    """
    from evaluation.utils import agent_utils

    trajectory_id = trajectory.get("task_instance_id", f"traj_{trajectory_idx}")
    session_id = BENCHMARK_SESSION_PREFIX

    memory, _, _, _, _ = await agent_utils.init_memmachine_params(
        resource_manager=resource_manager,
        session_id=session_id,
    )

    episodes = trajectory_to_episodes(
        trajectory=trajectory,
        trajectory_id=trajectory_id,
        session_id=session_id,
        base_time=base_time,
    )

    if episodes:
        await memory.add_memory_episodes(episodes=episodes)
        print(
            f"  Ingested trajectory {trajectory_id}: "
            f"{len(episodes)} episodes ({len(trajectory.get('state_action_pairs', []))} steps)"
        )

    return len(episodes)


async def main() -> None:
    from memmachine_server.common.utils import async_with

    from evaluation.utils import agent_utils

    args = build_parser().parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Data format: dict with "trajectories" key (AgentInstruct format)
    # or a flat list of trajectories
    if isinstance(data, dict):
        trajectories = data.get("trajectories", [])
    elif isinstance(data, list):
        trajectories = data
    else:
        raise TypeError(f"Unexpected data format: {type(data)}")

    print(f"Loaded {len(trajectories)} trajectories from {args.data_path}")

    resource_manager = agent_utils.load_eval_config(args.config_path)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        async_with(
            semaphore,
            ingest_trajectory(traj, idx, resource_manager, base_time),
        )
        for idx, traj in enumerate(trajectories)
    ]

    results = await asyncio.gather(*tasks)
    total_episodes = sum(results)
    print(
        f"\nIngestion complete: {len(trajectories)} trajectories, "
        f"{total_episodes} total episodes"
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
