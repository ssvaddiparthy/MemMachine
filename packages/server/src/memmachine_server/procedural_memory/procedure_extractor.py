"""Procedure Extractor — ingests trajectories into the procedural graph.

This module converts raw trajectory data (from ALFWorld, benchmarks, or
production agent logs) into the Neo4j procedure graph. Both successful and
failed trajectories are ingested with appropriate edge weights.

Novel mechanism: Failure-aware graph ingestion. Failed trajectories get
negative/zero edge weights so the Steiner tree naturally avoids known-bad
paths during compositional retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from memmachine_server.procedural_memory.data_types import TrajectoryMeta
from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

logger = logging.getLogger(__name__)


class ProcedureExtractor:
    """Extracts procedural structure from agent trajectories.

    Takes raw trajectory data and ingests it into the Neo4j procedure graph
    via the ProceduralGraphStore.
    """

    def __init__(self, graph_store: ProceduralGraphStore) -> None:
        self._store = graph_store

    async def extract_from_benchmark_trajectory(
        self,
        trajectory: dict[str, Any],
        success: bool | None = None,
    ) -> TrajectoryMeta:
        """Extract and ingest a single benchmark trajectory.

        Handles the proced_mem_bench / ALFWorld trajectory format:
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
            trajectory: Raw trajectory dict from the benchmark data.
            success: Override success flag. If None, defaults to True
                     (benchmark data is all successful trajectories).

        Returns:
            TrajectoryMeta for the ingested trajectory.
        """
        trajectory_id = trajectory.get("task_instance_id", "unknown")
        task_description = trajectory.get("task_description", "")
        total_steps = trajectory.get("total_steps", 0)
        source = trajectory.get("source", "unknown")

        if success is None:
            # Default: benchmark trajectories are all successful
            success = True

        # Convert state-action pairs to the action format expected by graph_store
        state_action_pairs = trajectory.get("state_action_pairs", [])
        actions = []

        for pair in state_action_pairs:
            step_id = pair.get("step_id", 0)
            state_text = pair.get("state", "")
            action_text = pair.get("action", "")

            # Extract tool_name and parameters from the action text.
            # ALFWorld actions are like "go to table 1", "pick up laptop 1",
            # "clean soapbar 1 with sinkbasin 1", etc.
            tool_name, params = self._parse_alfworld_action(action_text)

            actions.append({
                "tool_name": tool_name,
                "parameters": params,
                "step_id": step_id,
                "state_description": state_text,
            })

        # Ingest into Neo4j
        num_actions = await self._store.ingest_trajectory(
            trajectory_id=trajectory_id,
            task_description=task_description,
            success=success,
            actions=actions,
            source=source,
        )

        meta = TrajectoryMeta(
            trajectory_id=trajectory_id,
            task_description=task_description,
            success=success,
            total_steps=total_steps or num_actions,
            source=source,
        )

        return meta

    async def extract_batch(
        self,
        trajectories: list[dict[str, Any]],
        default_success: bool = True,
    ) -> list[TrajectoryMeta]:
        """Extract and ingest a batch of trajectories.

        Args:
            trajectories: List of trajectory dicts.
            default_success: Default success flag for trajectories without one.

        Returns:
            List of TrajectoryMeta for all ingested trajectories.
        """
        results = []
        for i, traj in enumerate(trajectories):
            try:
                meta = await self.extract_from_benchmark_trajectory(
                    traj, success=default_success
                )
                results.append(meta)
                if (i + 1) % 50 == 0:
                    logger.info("Extracted %d/%d trajectories", i + 1, len(trajectories))
            except Exception:
                logger.exception("Failed to extract trajectory %d", i)
        logger.info(
            "Batch extraction complete: %d/%d trajectories ingested",
            len(results), len(trajectories),
        )
        return results

    @staticmethod
    def _parse_alfworld_action(action_text: str) -> tuple[str, dict[str, Any]]:
        """Parse an ALFWorld action string into tool_name + parameters.

        ALFWorld actions follow patterns like:
        - "go to table 1"           → tool="go_to", params={"target": "table 1"}
        - "pick up laptop 1"        → tool="pick_up", params={"object": "laptop 1"}
        - "put laptop 1 in/on bed"  → tool="put", params={"object": "laptop 1", "receptacle": "bed"}
        - "clean soapbar 1 with sinkbasin 1" → tool="clean", params={"object": "soapbar 1", "with": "sinkbasin 1"}
        - "open cabinet 1"          → tool="open", params={"target": "cabinet 1"}
        - "close cabinet 1"         → tool="close", params={"target": "cabinet 1"}
        - "use lamp 1"              → tool="use", params={"target": "lamp 1"}
        - "examine object"          → tool="examine", params={"target": "object"}
        - "heat/cool X with Y"      → tool="heat/cool", params={"object": "X", "with": "Y"}

        Returns:
            (tool_name, parameters_dict)
        """
        action = action_text.strip().lower()

        # "go to X"
        m = re.match(r"go to (.+)", action)
        if m:
            return "go_to", {"target": m.group(1)}

        # "pick up X" or "take X from Y"
        m = re.match(r"(?:pick up|take) (.+?)(?:\s+from\s+(.+))?$", action)
        if m:
            params: dict[str, Any] = {"object": m.group(1)}
            if m.group(2):
                params["from"] = m.group(2)
            return "pick_up", params

        # "put X in/on Y"
        m = re.match(r"put (.+?)\s+(?:in/on|in|on)\s+(.+)", action)
        if m:
            return "put", {"object": m.group(1), "receptacle": m.group(2)}

        # "clean/heat/cool X with Y"
        m = re.match(r"(clean|heat|cool) (.+?)\s+with\s+(.+)", action)
        if m:
            return m.group(1), {"object": m.group(2), "with": m.group(3)}

        # "open/close/use/toggle X"
        m = re.match(r"(open|close|use|toggle|examine|look) (.+)", action)
        if m:
            return m.group(1), {"target": m.group(2)}

        # "inventory" or other single-word actions
        m = re.match(r"(\w+)$", action)
        if m:
            return m.group(1), {}

        # Fallback: use the whole string as tool_name
        return action.replace(" ", "_")[:50], {}
