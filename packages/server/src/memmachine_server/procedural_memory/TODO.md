# Procedural Memory — TODO

## ✅ P0: Remove benchmark-specific assumptions from procedural ingestion

**Resolved.** `_ingest_to_procedural_graph()` (per-batch, marker-dependent)
has been removed from `add_memory_episodes()`.

**Replacement**: `ingest_session_to_procedural_graph(session_id)` on
`EpisodicMemory` reconstructs the full trajectory from LTM (ordered by
`created_at`, no eviction risk) and calls the LLM-based
`ProcedureExtractor.extract_from_session_episodes()`. No benchmark markers,
no `trajectory_id`, no `step_id` dependencies.

**New endpoint**: `POST /api/v2/procedural/ingest-session` triggers this
from any agent framework session-close hook.

**Benchmark path unchanged**: `extract_from_benchmark_trajectory()` in
`procedure_extractor.py` still handles ALFWorld eval scripts directly.

---

## ✅ P1: Wire LLM into ProcedureExtractor

**Resolved.** Added `procedural_llm`, `procedural_community_detector`, and
`procedural_community_detection_threshold` fields to `EpisodicMemoryParams`.
`service_locator.py::episodic_memory_params_from_config()` now accepts
`procedural_llm_name` and resolves it via `resource_manager.get_language_model()`.
Falls back to `None` with a warning if the model is unavailable -- procedure
extraction is disabled but episodic memory continues unaffected.

## P2: session-close trigger in OpenClaw integration

`POST /api/v2/procedural/ingest-session` needs to be called when a session
ends. OpenClaw's `agent_end` hook fires after each turn (not true session end).
Options:
- Add an `agent_session_end` hook call if OpenClaw supports it
- Use a TTL-based trigger: N minutes of inactivity → fire ingest
- Explicit API call from the user's agent loop

## P3: Failure trajectory generation

The ALFWorld benchmark contains only successful trajectories. The v1 success
gate (`success=True` required to ingest) means failure data is moot for now.
For v2 (weighted edges), need failure data. Options:
- Run a weaker model on ALFWorld to generate real failures
- Synthetically corrupt successful trajectories
- Treat low-scoring retrievals as soft failures

## ✅ P4: Community detection trigger after session ingest

**Resolved.** `ingest_session_to_procedural_graph` now increments
`_procedural_trajectories_since_louvain` after each successful ingest and
calls `CommunityDetector.detect_communities()` when the count reaches
`procedural_community_detection_threshold` (default 10). Counter resets after
each Louvain run. Failures are non-blocking. Mirrors semantic memory's
threshold-consolidation pattern.

## ✅ P5: Composition validation via LLM

**Resolved.** `ProceduralAgent.do_query()` now calls `_validate_with_llm()`
after Steiner tree composition. The LLM receives the query + composed procedure
and returns `{valid, confidence, reason}`. If `valid=false`, the agent returns
`[]` triggering the existing episodic fallback. On LLM failure the procedure
passes through (non-blocking). The LLM client is injected via
`extra_params["procedural_llm"]`.

## ✅ P6: ALFWorld verb heuristic removed from ToolSelectAgent

**Resolved.** Removed `"clean"`, `"heat"`, `"cool"`, `"put"`, `"place"` from
`procedural_signals` in `tool_select_agent.py`. The heuristic now only matches
generic "how to" / "steps to" / "workflow for" language. Domain-specific verbs
rely on the LLM router and the reactive empty-result fallback instead.
