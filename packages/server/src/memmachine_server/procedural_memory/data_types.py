"""Data types for the procedural memory tier."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


@dataclass
class ToolNode:
    """A unique tool type in the agent's action space.

    Corresponds to a :Tool node in Neo4j. MERGE'd on name so there is exactly
    one :Tool per distinct tool_name across all trajectories.
    """

    name: str
    description: str = ""
    total_invocations: int = 0
    schema_hash: str = ""


@dataclass
class ActionNode:
    """One specific invocation of a tool within one trajectory.

    Corresponds to an :Action node in Neo4j. Each state-action pair in a
    trajectory becomes one ActionNode.
    """

    action_id: str  # "{trajectory_id}_{step_id}"
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    step_id: int = 0
    trajectory_id: str = ""
    success: bool = True
    state_description: str = ""
    community_id: int | None = None


@dataclass
class TrajectoryMeta:
    """Metadata for a complete trajectory.

    Corresponds to a :Trajectory node in Neo4j.
    """

    trajectory_id: str
    task_description: str
    success: bool
    total_steps: int
    source: str = "unknown"
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Community:
    """A Louvain-discovered cluster of frequently co-occurring actions.

    Corresponds to a :Community node in Neo4j. Created by the community
    detection stage (Stage 3).
    """

    community_id: int
    community_type: str  # "success" or "failure"
    description: str = ""
    member_count: int = 0
    trajectory_count: int = 0
    avg_weight: float = 0.0
    member_actions: list[str] = field(default_factory=list)


@dataclass
class ToolTransition:
    """An aggregate tool-to-tool transition edge.

    Corresponds to a :TOOL_TRANSITION relationship in Neo4j between two :Tool
    nodes. Accumulates success/failure counts across all trajectories.
    """

    from_tool: str
    to_tool: str
    success_count: int = 0
    fail_count: int = 0
    epsilon: float = 0.01

    @property
    def weight(self) -> float:
        """Reliability weight in [0, 1]. Higher = more reliable."""
        return self.success_count / (
            self.success_count + self.fail_count + self.epsilon
        )

    @property
    def cost(self) -> float:
        """Dijkstra cost = 1 - weight. Lower weight = higher cost."""
        return 1.0 - self.weight


@dataclass
class ComposedProcedure:
    """A procedure composed from fragments of multiple trajectories.

    Returned by the Steiner tree compositional retriever. This procedure may
    never have existed as a single stored trajectory — it was synthesized at
    query time from the best subpaths in the procedure graph.
    """

    steps: list[str]
    source_trajectories: list[str] = field(default_factory=list)
    confidence: float = 0.0  # min edge weight along the composed path
    avoid_warnings: list[str] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    total_cost: float = 0.0

    def to_text(self) -> str:
        """Serialize to a human-readable procedure description."""
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step}")
        if self.avoid_warnings:
            lines.append("")
            for warning in self.avoid_warnings:
                lines.append(f"⚠️ Avoid: {warning}")
        return "\n".join(lines)
