"""Procedure Extractor -- ingests trajectories into the procedural graph.

Two extraction paths exist:

1. **Session-level (production path)**: `extract_from_session_episodes()`
   Called at session end with the full ordered episode list reconstructed from
   LTM. An LLM determines whether the trajectory contains a procedure and
   extracts structured steps (tool, params, order, state). This path is
   integration-agnostic -- it works with any agent framework.

2. **Benchmark path**: `extract_from_benchmark_trajectory()`
   Takes ALFWorld-format JSON directly. Used by eval scripts
   (proced_mem_bench_ingest.py) only. Not called from production code.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from memmachine_server.procedural_memory.data_types import TrajectoryMeta
from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_PROCEDURE_EXTRACTION_PROMPT = """\
You are a procedure extraction system. Given a sequence of agent messages,
determine whether the conversation represents a completed procedural task
(a sequence of tool calls or actions that achieves a specific goal).

A procedure is present when:
- The agent executed a sequence of 2+ distinct tool calls or actions
- The actions have a logical ordering (each step depends on or follows from prior steps)
- Together they accomplish a concrete goal

Respond with a JSON object only, no prose:

If NO procedure is present:
{"has_procedure": false}

If a procedure IS present:
{
  "has_procedure": true,
  "goal": "<one-sentence description of what the procedure accomplishes>",
  "success": <true if the goal was achieved, false if it failed or was abandoned>,
  "steps": [
    {
      "step_id": 1,
      "tool_name": "<tool or action name, snake_case>",
      "parameters": {"<key>": "<value>"},
      "state_description": "<environment/context state before this step>"
    },
    ...
  ]
}

Rules:
- tool_name should be a normalized snake_case identifier (e.g. go_to, pick_up, send_email)
- parameters should only include values explicitly present in the message
- state_description should describe what was observable before the step was taken
- Include only steps that are part of the core procedure, skip conversational turns
- success is true only if the final state indicates the goal was achieved

Conversation:
{conversation}
"""


class ProcedureExtractor:
    """Extracts procedural structure from agent trajectories.

    Takes raw trajectory data and ingests it into the Neo4j procedure graph
    via the ProceduralGraphStore.
    """

    def __init__(self, graph_store: ProceduralGraphStore, llm: Any | None = None) -> None:
        self._store = graph_store
        self._llm = llm  # LLM client (Instructor-wrapped Bedrock or compatible)

    # ------------------------------------------------------------------
    # Production path: session-level LLM extraction
    # ------------------------------------------------------------------

    async def extract_from_session_episodes(
        self,
        session_id: str,
        episodes: list[Any],  # list[Episode], ordered by created_at
    ) -> TrajectoryMeta | None:
        """Extract a procedure from a completed session's episode sequence.

        Called at session end with episodes reconstructed from LTM, ordered
        by created_at. An LLM determines if the trajectory contains a
        procedure and extracts structured steps.

        Args:
            session_id: The session/run_id that produced these episodes.
            episodes: Full ordered episode list from LTM for this session.

        Returns:
            TrajectoryMeta if a procedure was found and ingested, else None.
        """
        if not episodes:
            return None

        if self._llm is None:
            logger.warning(
                "ProcedureExtractor: no LLM configured, skipping session %s", session_id
            )
            return None

        # Build conversation text for the LLM
        conversation_lines = []
        for ep in episodes:
            role = (ep.metadata or {}).get("role", "unknown") if ep.metadata else "unknown"
            content = ep.content or ""
            conversation_lines.append(f"[{role}]: {content}")
        conversation_text = "\n".join(conversation_lines)

        # LLM call: detect procedure and extract steps
        try:
            result = await self._call_llm_extract(conversation_text)
        except Exception:
            logger.exception(
                "ProcedureExtractor: LLM extraction failed for session %s", session_id
            )
            return None

        if not result.get("has_procedure", False):
            logger.debug(
                "ProcedureExtractor: no procedure found in session %s", session_id
            )
            return None

        goal = result.get("goal", "")
        success = result.get("success", True)
        steps = result.get("steps", [])

        if len(steps) < 2:
            logger.debug(
                "ProcedureExtractor: too few steps (%d) in session %s, skipping",
                len(steps), session_id,
            )
            return None

        # Build actions list for graph_store
        actions = []
        for step in steps:
            actions.append({
                "tool_name": step.get("tool_name", "unknown"),
                "parameters": step.get("parameters", {}),
                "step_id": step.get("step_id", 0),
                "state_description": step.get("state_description", ""),
            })

        # Ingest into Neo4j (success-only gate: skip failed trajectories in v1)
        if not success:
            logger.debug(
                "ProcedureExtractor: session %s marked failed, skipping ingest (v1 gate)",
                session_id,
            )
            return TrajectoryMeta(
                trajectory_id=session_id,
                task_description=goal,
                success=False,
                total_steps=len(steps),
                source="production",
            )

        num_actions = await self._store.ingest_trajectory(
            trajectory_id=session_id,
            task_description=goal,
            success=True,
            actions=actions,
            source="production",
        )

        logger.info(
            "ProcedureExtractor: ingested %d steps for session %s (goal: %s)",
            num_actions, session_id, goal,
        )

        return TrajectoryMeta(
            trajectory_id=session_id,
            task_description=goal,
            success=True,
            total_steps=num_actions,
            source="production",
        )

    async def _call_llm_extract(self, conversation_text: str) -> dict[str, Any]:
        """Call the LLM to detect and extract procedure from conversation text.

        Returns the parsed JSON dict from the LLM response.
        """
        prompt = _PROCEDURE_EXTRACTION_PROMPT.format(conversation=conversation_text)

        from memmachine_server.procedural_memory.llm_utils import (
            llm_complete,
            parse_json_object,
        )

        raw = await llm_complete(self._llm, prompt)
        return parse_json_object(raw or "{}")

    # ------------------------------------------------------------------
    # Benchmark path: ALFWorld eval scripts only
    # ------------------------------------------------------------------

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
            success = True

        state_action_pairs = trajectory.get("state_action_pairs", [])
        actions = []

        for pair in state_action_pairs:
            step_id = pair.get("step_id", 0)
            state_text = pair.get("state", "")
            action_text = pair.get("action", "")

            tool_name, params = self._parse_alfworld_action(action_text)

            actions.append({
                "tool_name": tool_name,
                "parameters": params,
                "step_id": step_id,
                "state_description": state_text,
            })

        num_actions = await self._store.ingest_trajectory(
            trajectory_id=trajectory_id,
            task_description=task_description,
            success=success,
            actions=actions,
            source=source,
        )

        return TrajectoryMeta(
            trajectory_id=trajectory_id,
            task_description=task_description,
            success=success,
            total_steps=total_steps or num_actions,
            source=source,
        )

    async def extract_batch(
        self,
        trajectories: list[dict[str, Any]],
        default_success: bool = True,
    ) -> list[TrajectoryMeta]:
        """Extract and ingest a batch of benchmark trajectories."""
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

        Used by the benchmark path only.
        """
        action = action_text.strip().lower()

        m = re.match(r"go to (.+)", action)
        if m:
            return "go_to", {"target": m.group(1)}

        m = re.match(r"(?:pick up|take) (.+?)(?:\s+from\s+(.+))?$", action)
        if m:
            params: dict[str, Any] = {"object": m.group(1)}
            if m.group(2):
                params["from"] = m.group(2)
            return "pick_up", params

        m = re.match(r"put (.+?)\s+(?:in/on|in|on)\s+(.+)", action)
        if m:
            return "put", {"object": m.group(1), "receptacle": m.group(2)}

        m = re.match(r"(clean|heat|cool) (.+?)\s+with\s+(.+)", action)
        if m:
            return m.group(1), {"object": m.group(2), "with": m.group(3)}

        m = re.match(r"(open|close|use|toggle|examine|look) (.+)", action)
        if m:
            return m.group(1), {"target": m.group(2)}

        m = re.match(r"(\w+)$", action)
        if m:
            return m.group(1), {}

        return action.replace(" ", "_")[:50], {}
