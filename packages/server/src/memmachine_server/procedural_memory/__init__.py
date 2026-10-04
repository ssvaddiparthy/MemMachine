"""Procedural memory tier for MemMachine.

Gap-aware procedural memory: rather than hand an agent a whole retrieved
trajectory, we decompose the task into ordered sub-goals, align them against
the best stored trajectory, and return an explicit edit-script diagnosis of
which required sub-goals the trajectory COVERS and which are MISSING. The
diagnosis -- not the raw trajectory -- is the active ingredient (see
evaluation/procedural_memory/DETAILED_REPORT.md).

Two seams:
1. Ingestion -- ProcedureExtractor writes a trajectory's actions (and its
   ordered sub-goal signature) into the procedural graph.
2. Retrieval -- GroundedRetriever decomposes the query, anchors it to the best
   trajectory, and emits a COVERED / REBIND / GAP edit script.

The procedural graph is stored in the same Neo4j instance MemMachine uses for
episodic memory.
"""

from memmachine_server.procedural_memory.data_types import (
    ActionNode,
    ToolNode,
    TrajectoryMeta,
)
from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore
from memmachine_server.procedural_memory.grounded_retriever import (
    EditOp,
    GroundedProcedure,
    GroundedRetriever,
)
from memmachine_server.procedural_memory.procedure_extractor import ProcedureExtractor

__all__ = [
    "ActionNode",
    "EditOp",
    "GroundedProcedure",
    "GroundedRetriever",
    "ProceduralGraphStore",
    "ProcedureExtractor",
    "ToolNode",
    "TrajectoryMeta",
]
