"""
Defines the core memory instance for a specific conversational context.

This module provides the `EpisodicMemory` class, which acts as the primary
orchestrator for an individual memory session. It integrates short-term
(session) and long-term (declarative) memory stores to provide a unified
interface for adding and retrieving conversational data.

Key responsibilities include:
- Managing the lifecycle of the memory instance through reference counting.
- Adding new conversational `Episode` objects to both session and declarative
  memory.
- Retrieving relevant context for a query by searching both memory types.
- Interacting with a language model for memory-related tasks.
- Each instance is managed by the `EpisodicMemoryManager`.
"""

import asyncio
import datetime
import logging
import time
from collections.abc import Coroutine, Iterable
from enum import StrEnum
from typing import cast, get_args, Any

from pydantic import BaseModel, Field, InstanceOf, model_validator

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.episode_store import (
    Episode,
    EpisodeResponse,
)
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    LongTermMemory,
)
from memmachine_server.episodic_memory.short_term_memory.short_term_memory import (
    ShortTermMemory,
)

logger = logging.getLogger(__name__)


class EpisodicMemoryParams(BaseModel):
    """
    Parameters for configuring the EpisodicMemory.

    Attributes:
        session_key (str): The unique identifier for the session.
        metrics_factory (MetricsFactory): The metrics factory.
        long_term_memory (LongTermMemory): The long-term memory.
        short_term_memory (ShortTermMemory): The short-term memory.
        enabled (bool): Whether the episodic memory is enabled.

    """

    session_key: str = Field(
        ...,
        min_length=1,
        description="The unique identifier for the session",
    )
    metrics_factory: InstanceOf[MetricsFactory] = Field(
        ...,
        description="The metrics factory",
    )
    long_term_memory: InstanceOf[LongTermMemory] | None = Field(
        default=None,
        description="The long-term memory",
    )
    short_term_memory: InstanceOf[ShortTermMemory] | None = Field(
        default=None,
        description="The short-term memory",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the episodic memory is enabled",
    )
    # Procedural memory tier (optional). When set, trajectories are also
    # ingested into the Neo4j procedure graph alongside STM and LTM.
    procedural_graph_store: Any | None = Field(
        default=None,
        description="Optional ProceduralGraphStore for the procedural memory tier",
    )

    @model_validator(mode="after")
    def validate_memory_params(self) -> "EpisodicMemoryParams":
        if not self.enabled:
            return self
        if self.short_term_memory is None and self.long_term_memory is None:
            raise ValueError(
                "At least one of short_term_memory or long_term_memory must be provided.",
            )
        return self


class EpisodicMemory:
    """
    Represents a single, isolated memory instance for a specific session.

    This class orchestrates the interaction between short-term (session)
    memory and long-term (declarative) memory. It manages the lifecycle of
    the memory, handles adding new information (episodes), and provides
    methods to retrieve contextual information for queries.

    Each instance is tied to a unique session key
    """

    def __init__(
        self,
        params: EpisodicMemoryParams,
    ) -> None:
        """
        Initialize a EpisodicMemory instance.

        Args:
            params (EpisodicMemoryParams): Parameters for the EpisodicMemory.

        """
        self._closed = False

        self._session_key = params.session_key

        self._short_term_memory: ShortTermMemory | None = params.short_term_memory
        self._long_term_memory: LongTermMemory | None = params.long_term_memory
        self._procedural_graph_store = params.procedural_graph_store

        self._enabled = params.enabled
        if not self._enabled:
            return
        if self._short_term_memory is None and self._long_term_memory is None:
            raise ValueError("No memory is configured")

        metrics_manager = params.metrics_factory
        # Initialize metrics
        self._ingestion_latency_summary = metrics_manager.get_summary(
            "Ingestion_latency",
            "Latency of Episode ingestion in milliseconds",
        )
        self._query_latency_summary = metrics_manager.get_summary(
            "query_latency",
            "Latency of query processing in milliseconds",
        )
        self._ingestion_counter = metrics_manager.get_counter(
            "Ingestion_count",
            "Count of Episode ingestion",
        )
        self._query_counter = metrics_manager.get_counter(
            "query_count",
            "Count of query processing",
        )

    @property
    def short_term_memory(self) -> ShortTermMemory | None:
        """
        Get the short-term memory of the episodic memory instance.

        Returns:
            The short-term memory of the episodic memory instance.

        """
        return self._short_term_memory

    @short_term_memory.setter
    def short_term_memory(self, value: ShortTermMemory | None) -> None:
        """
        Set the short-term memory of the episodic memory instance.

        This makes the short term memory can be injected.

        Args:
            value: The new short-term memory of the episodic memory instance.

        """
        self._short_term_memory = value

    @property
    def long_term_memory(self) -> LongTermMemory | None:
        """
        Get the long-term memory of the episodic memory instance.

        Returns:
            The long-term memory of the episodic memory instance.

        """
        return self._long_term_memory

    @long_term_memory.setter
    def long_term_memory(self, value: LongTermMemory | None) -> None:
        """
        Set the long-term memory of the episodic memory instance.

        This makes the long term memory can be injected.

        Args:
            value: The new long-term memory of the episodic memory instance.

        """
        self._long_term_memory = value

    @property
    def session_key(self) -> str:
        """
        Get the session key of the episodic memory instance.

        Returns:
            The session key of the episodic memory instance.

        """
        return self._session_key

    async def add_memory_episodes(self, episodes: list[Episode]) -> None:
        """
        Add a new memory episode to both session and declarative memory.

        Args:
            episodes: Episode instances to ingest.

        """
        if not self._enabled:
            return
        start_time = time.monotonic_ns()

        if self._closed:
            raise RuntimeError(f"Memory is closed {self._session_key}")
        # Create filterable property
        for episode in episodes:
            if episode.metadata is not None and episode.filterable_metadata is None:
                episode.filterable_metadata = {}
                for key, value in episode.metadata.items():
                    if isinstance(value, get_args(PropertyValue)):
                        episode.filterable_metadata[key] = value

        # Add the episode to both memory stores concurrently
        tasks: list[Coroutine] = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.add_episodes(episodes))
        if self._long_term_memory:
            tasks.append(self._long_term_memory.add_episodes(episodes))
        # Procedural memory tier: ingest episodes into the procedure graph
        # alongside STM and LTM. The procedural_graph_store handles its own
        # extraction logic (parsing tool names, building graph edges).
        if self._procedural_graph_store is not None:
            tasks.append(
                self._ingest_to_procedural_graph(episodes)
            )
        await asyncio.gather(
            *tasks,
        )
        end_time = time.monotonic_ns()
        delta = (end_time - start_time) / 1000000
        self._ingestion_latency_summary.observe(delta)
        self._ingestion_counter.increment()

    async def _ingest_to_procedural_graph(self, episodes: list[Episode]) -> None:
        """Ingest episodes into the procedural memory graph.

        Extracts tool-call structure from episode content and builds the
        procedure graph in Neo4j. Episodes with [STATE] and [ACTION] markers
        (from benchmark trajectories) are parsed into tool nodes and edges.

        This runs concurrently with STM and LTM ingestion via asyncio.gather
        in add_memory_episodes().
        """
        if self._procedural_graph_store is None:
            return

        try:
            # Group episodes by trajectory_id from metadata
            trajectory_episodes: dict[str, list[Episode]] = {}
            for ep in episodes:
                tid = (ep.metadata or {}).get("trajectory_id", "")
                if tid:
                    trajectory_episodes.setdefault(tid, []).append(ep)

            if not trajectory_episodes:
                return

            for tid, traj_episodes in trajectory_episodes.items():
                # Convert episodes to action dicts for graph_store
                actions = []
                for ep in sorted(traj_episodes, key=lambda e: (e.metadata or {}).get("step_id", 0)):
                    content = ep.content or ""
                    metadata = ep.metadata or {}

                    # Skip non-action episodes (task descriptions, etc.)
                    if "[ACTION]" not in content and "[STATE]" not in content:
                        continue

                    # Extract state and action from episode content
                    state_desc = ""
                    action_text = ""
                    if "[STATE]" in content:
                        parts = content.split("[ACTION]", 1)
                        state_desc = parts[0].replace("[STATE]", "").strip()
                        if len(parts) > 1:
                            action_text = parts[1].strip()
                    elif "[ACTION]" in content:
                        action_text = content.replace("[ACTION]", "").strip()

                    if not action_text:
                        continue

                    # Parse tool name from action text (reuse extractor logic)
                    from memmachine_server.procedural_memory.procedure_extractor import (
                        ProcedureExtractor,
                    )
                    tool_name, params = ProcedureExtractor._parse_alfworld_action(
                        action_text
                    )

                    actions.append({
                        "tool_name": tool_name,
                        "parameters": params,
                        "step_id": metadata.get("step_id", 0),
                        "state_description": state_desc,
                    })

                if actions:
                    # Determine success from metadata (default True for benchmark data)
                    success = traj_episodes[0].metadata.get("success", True) if traj_episodes[0].metadata else True
                    task_desc = ""
                    for ep in traj_episodes:
                        if ep.content and "[TASK]" in ep.content:
                            task_desc = ep.content.replace("[TASK]", "").strip()
                            break

                    await self._procedural_graph_store.ingest_trajectory(
                        trajectory_id=tid,
                        task_description=task_desc,
                        success=success,
                        actions=actions,
                        source=(ep.metadata or {}).get("benchmark", "unknown"),
                    )

        except Exception:
            # Procedural ingestion should never block episodic ingestion
            logger.exception("Procedural graph ingestion failed (non-blocking)")

    async def close(self) -> None:
        """
        Decrement the reference count and close the underlying memory stores.

        When the reference count is zero, it closes the memory stores and
        notifies the manager to remove this instance from its registry.
        """
        self._closed = True
        if not self._enabled:
            return
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.close())
        if self._long_term_memory:
            tasks.append(self._long_term_memory.close())
        await asyncio.gather(*tasks)

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        """Delete episodes by UID."""
        if not self._enabled:
            return

        uids = list(uids)

        delete_episodes_coroutines: list[Coroutine] = []
        if self._short_term_memory:
            delete_episodes_coroutines.extend(
                self._short_term_memory.delete_episode(uid) for uid in uids
            )
        if self._long_term_memory:
            delete_episodes_coroutines.append(
                self._long_term_memory.delete_episodes(uids)
            )
        await asyncio.gather(*delete_episodes_coroutines)

    async def delete_session_episodes(self) -> None:
        """Delete all data from both session and declarative memory for this context."""
        if not self._enabled:
            return
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.clear_memory())
        if self._long_term_memory:
            tasks.append(self._long_term_memory.drop_session_partition())
        await asyncio.gather(*tasks)

    class QueryResponse(BaseModel):
        """Aggregated search results from both long- and short-term memory."""

        class ShortTermMemoryResponse(BaseModel):
            """Aggregated search results from short-term memory."""

            episodes: list[EpisodeResponse]
            episode_summary: list[str]

        class LongTermMemoryResponse(BaseModel):
            """Aggregated search results from long-term memory."""

            episodes: list[EpisodeResponse]

        long_term_memory: LongTermMemoryResponse
        short_term_memory: ShortTermMemoryResponse

    class QueryMode(StrEnum):
        """Controls which memory backends are queried."""

        BOTH = "both"
        LONG_TERM_ONLY = "long_term_only"
        SHORT_TERM_ONLY = "short_term_only"

    async def _query_short_term_memory(
        self,
        *,
        query: str,
        limit: int,
        property_filter: FilterExpr | None,
    ) -> tuple[list[Episode], str]:
        """Query short-term memory or return an empty result if unavailable."""
        if self._short_term_memory is None:
            return [], ""

        return await self._short_term_memory.get_short_term_memory_context(
            query,
            limit=limit,
            filters=property_filter,
        )

    async def _query_long_term_memory(
        self,
        *,
        query: str,
        limit: int,
        expand_context: int,
        score_threshold: float,
        property_filter: FilterExpr | None,
    ) -> list[tuple[float, Episode]]:
        """Query long-term memory or return an empty result if unavailable."""
        if self._long_term_memory is None:
            return []

        return await self._long_term_memory.search_scored(
            query,
            num_episodes_limit=limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            property_filter=property_filter,
        )

    async def query_memory(
        self,
        query: str,
        *,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
        mode: QueryMode = QueryMode.BOTH,
    ) -> QueryResponse | None:
        """
        Retrieve relevant context for a given query from all memory stores.

        It fetches episodes from both short-term (session) and long-term
        (declarative) memory, deduplicates them, and returns them along with
        any available summary.

        Args:
            query: The query string to find context for.
            limit: The maximum number of episodes to return. The limit is
                   applied to both short and long term memories. The default
                   value is 20.
            expand_context: The number of additional episodes to include
                            around each matched episode from long term memory.
            score_threshold: Minimum score to consider a match.
            property_filter: Properties to filter declarative memory searches.
            mode: Which memory backends to query.

        Returns:
            A tuple containing a list of short term memory Episode objects,
            a list of long term memory Episode objects, and a
            list of summary strings.

        """
        if not self._enabled:
            return None
        start_time = time.monotonic_ns()
        search_limit = limit if limit is not None else 20

        short_episode: list[Episode] = []
        short_summary = ""
        scored_long_episodes: list[tuple[float, Episode]] = []

        if mode is EpisodicMemory.QueryMode.BOTH:
            if self._short_term_memory is None:
                scored_long_episodes = await self._query_long_term_memory(
                    query=query,
                    limit=search_limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    property_filter=property_filter,
                )
            elif self._long_term_memory is None:
                short_episode, short_summary = await self._query_short_term_memory(
                    query=query,
                    limit=search_limit,
                    property_filter=property_filter,
                )
            else:
                # Concurrently search both memory stores.
                session_result, scored_long_episodes = await asyncio.gather(
                    self._query_short_term_memory(
                        query=query,
                        limit=search_limit,
                        property_filter=property_filter,
                    ),
                    self._query_long_term_memory(
                        query=query,
                        limit=search_limit,
                        expand_context=expand_context,
                        score_threshold=score_threshold,
                        property_filter=property_filter,
                    ),
                )
                short_episode, short_summary = session_result
        elif mode is EpisodicMemory.QueryMode.LONG_TERM_ONLY:
            scored_long_episodes = await self._query_long_term_memory(
                query=query,
                limit=search_limit,
                expand_context=expand_context,
                score_threshold=score_threshold,
                property_filter=property_filter,
            )
        elif mode is EpisodicMemory.QueryMode.SHORT_TERM_ONLY:
            short_episode, short_summary = await self._query_short_term_memory(
                query=query,
                limit=search_limit,
                property_filter=property_filter,
            )
        else:
            raise ValueError(
                "Unsupported query mode."
                f" Expected one of {[item.value for item in EpisodicMemory.QueryMode]},"
                f" got {mode!r}."
            )

        # Deduplicate episodes from both memory stores, prioritizing
        # short-term memory
        episode_uid_set = {episode.uid for episode in short_episode}

        unique_scored_long_episodes = []
        for score, episode in scored_long_episodes:
            if episode.uid not in episode_uid_set:
                episode_uid_set.add(episode.uid)
                unique_scored_long_episodes.append((score, episode))

        end_time = time.monotonic_ns()
        delta = (end_time - start_time) / 1000000
        self._query_latency_summary.observe(delta)
        self._query_counter.increment()

        return EpisodicMemory.QueryResponse(
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[
                    EpisodeResponse(**episode.model_dump()) for episode in short_episode
                ],
                episode_summary=[short_summary],
            ),
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[
                    EpisodeResponse(score=score, **episode.model_dump())
                    for score, episode in unique_scored_long_episodes
                ],
            ),
        )

    async def formalize_query_with_context(
        self,
        query: str,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
    ) -> str:
        """
        Construct a finalized query string that includes context from memory.

        The context (summary and recent episodes) is prepended to the original
        query, formatted with XML-like tags for the language model to parse.

        Args:
            query: The original query string.
            limit: The maximum number of episodes to include in the context.
            expand_context: The number of additional episodes to include
                            around each matched episode from long term memory.
            score_threshold: Minimum score to include in the context.
            property_filter: Properties to filter the search.

        Returns:
            A new query string enriched with context.

        """
        query_result = await self.query_memory(
            query,
            limit=limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            property_filter=property_filter,
        )
        if query_result is None:
            logger.warning("Query result is None in formalize_query_with_context")
            return query

        episodes = sorted(
            query_result.short_term_memory.episodes
            + query_result.long_term_memory.episodes,
            key=lambda x: cast(datetime.datetime, x.created_at),
        )

        finalized_query = ""

        # Add summary if it exists
        if (
            query_result.short_term_memory.episode_summary
            and len(query_result.short_term_memory.episode_summary) > 0
        ):
            total_summary = ""
            for summ in query_result.short_term_memory.episode_summary:
                if not summ:
                    continue
                total_summary = total_summary + summ + "\n"
            total_summary = total_summary.strip()
            if total_summary:
                finalized_query += "<Summary>\n"
                finalized_query += total_summary
                finalized_query += "\n</Summary>\n"

        # Add episodes if they exist
        if episodes and len(episodes) > 0:
            finalized_query += "<Episodes>\n"
            finalized_query += episodes_to_string(
                episodes,
            )
            finalized_query += "</Episodes>\n"

        # Append the original query
        finalized_query += f"<Query>\n{query}\n</Query>"

        return finalized_query
