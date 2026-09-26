"""Procedural memory tier for MemMachine.

This package implements failure-aware procedural memory with three novel mechanisms:
1. Failure-aware graph ingestion (success/failure weighted edges)
2. Louvain community detection for emergent sub-procedure discovery
3. Steiner tree compositional retrieval for novel procedure synthesis

The procedural graph is stored in the same Neo4j instance MemMachine uses for
episodic memory, and retrieval flows through the existing ToolSelectAgent pipeline.
"""

from memmachine_server.procedural_memory.data_types import (
    ActionNode,
    Community,
    ComposedProcedure,
    ToolNode,
    TrajectoryMeta,
)
from memmachine_server.procedural_memory.graph_store import ProceduralGraphStore
from memmachine_server.procedural_memory.procedure_extractor import ProcedureExtractor
from memmachine_server.procedural_memory.community_detector import CommunityDetector
from memmachine_server.procedural_memory.compositional_retriever import (
    CompositionalRetriever,
)

__all__ = [
    "ActionNode",
    "Community",
    "CommunityDetector",
    "ComposedProcedure",
    "CompositionalRetriever",
    "ProceduralGraphStore",
    "ProcedureExtractor",
    "ToolNode",
    "TrajectoryMeta",
]
