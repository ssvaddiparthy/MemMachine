"""Factory helpers for wiring episodic memory components."""

from typing import Any

from pydantic import InstanceOf

from memmachine_server.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine_server.common.resource_manager import CommonResourceManager

from .episodic_memory import EpisodicMemoryParams
from .long_term_memory.long_term_memory import LongTermMemory
from .long_term_memory.service_locator import (
    long_term_memory_params_from_config,
)
from .short_term_memory.service_locator import (
    short_term_memory_params_from_config,
)
from .short_term_memory.short_term_memory import ShortTermMemory


async def episodic_memory_params_from_config(
    config: EpisodicMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
    procedural_graph_store: Any | None = None,
    procedural_llm_name: str | None = None,
) -> EpisodicMemoryParams:
    """Create EpisodicMemoryParams from configuration and resource manager.

    Args:
        config: Episodic memory configuration.
        resource_manager: Shared resource manager (embedder, LTM store, etc.).
        procedural_graph_store: Optional Neo4j procedure graph store.
        procedural_llm_name: Name of the language model to use for procedure
            extraction. If None, procedural LLM extraction is disabled.
    """
    long_term_memory: LongTermMemory | None = None
    if config.long_term_memory and config.long_term_memory_enabled:
        long_term_memory_params = await long_term_memory_params_from_config(
            config.long_term_memory,
            resource_manager,
        )
        long_term_memory = LongTermMemory(long_term_memory_params)

    short_term_memory: ShortTermMemory | None = None
    if config.short_term_memory and config.short_term_memory_enabled:
        short_term_memory_params = await short_term_memory_params_from_config(
            config.short_term_memory,
            resource_manager,
        )
        short_term_memory = await ShortTermMemory.create(short_term_memory_params)

    metrics_factory_id = config.metrics_factory_id or "prometheus"

    # Resolve procedural LLM if a model name was provided
    procedural_llm: Any | None = None
    if procedural_llm_name:
        try:
            procedural_llm = await resource_manager.get_language_model(
                procedural_llm_name, validate=True
            )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "episodic_memory_params_from_config: could not load procedural LLM "
                "'%s' -- procedure extraction will be disabled",
                procedural_llm_name,
            )

    return EpisodicMemoryParams(
        session_key=config.session_key,
        metrics_factory=await resource_manager.get_metrics_factory(
            metrics_factory_id,
        ),
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        enabled=config.enabled,
        procedural_graph_store=procedural_graph_store,
        procedural_llm=procedural_llm,
    )
