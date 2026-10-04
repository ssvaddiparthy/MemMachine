"""Data types for the procedural memory tier."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


@dataclass
class ToolNode:
    """A unique tool type in the agent's action space.

    Corresponds to a :ProcTool node in Neo4j. MERGE'd on name so there is
    exactly one per distinct tool_name across all trajectories.
    """

    name: str
    description: str = ""
    total_invocations: int = 0
    schema_hash: str = ""


@dataclass
class ActionNode:
    """One specific invocation of a tool within one trajectory.

    Corresponds to a :ProcAction node in Neo4j. Each state-action pair in a
    trajectory becomes one ActionNode.
    """

    action_id: str  # "{trajectory_id}_{step_id}"
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    step_id: int = 0
    trajectory_id: str = ""
    success: bool = True
    state_description: str = ""


@dataclass
class TrajectoryMeta:
    """Metadata for a complete trajectory.

    Corresponds to a :ProcTrajectory node in Neo4j.
    """

    trajectory_id: str
    task_description: str
    success: bool
    total_steps: int
    source: str = "unknown"
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
