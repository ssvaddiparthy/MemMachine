"""Offline tests for the Stage-1 deterministic arms. No LLM, no DB.

Run from the repo root:
    python -m pytest evaluation/procedural_memory/test_stage1_align.py -q
or standalone:
    python evaluation/procedural_memory/test_stage1_align.py
"""

from __future__ import annotations

from evaluation.procedural_memory.stage1_align import (
    TypedStep,
    bag_cost,
    ordered_align_cost,
    parse_action,
    query_to_subgoals,
)


def test_parse_core_actions():
    assert parse_action("take soapbar 1 from countertop 1") == TypedStep("take", "soapbar", "countertop")
    assert parse_action("put laptop 1 in/on bed 1") == TypedStep("put", "laptop", "bed")
    assert parse_action("clean soapbar 2 with sinkbasin 1") == TypedStep("clean", "soapbar", "sinkbasin")
    assert parse_action("heat apple 1 with microwave 1") == TypedStep("heat", "apple", "microwave")
    assert parse_action("go to cabinet 2") == TypedStep("go_to", "cabinet", None)
    assert parse_action("open drawer 1") == TypedStep("open", "drawer", None)
    assert parse_action("use desklamp 1") == TypedStep("use", "desklamp", None)


def test_parse_unknown_returns_none():
    assert parse_action("") is None
    assert parse_action("ponder the universe") is None


def test_query_subgoals_placement():
    q = {"extracted_keywords": {"objects": ["soapbar", "cabinet"],
                                "verbs": ["place", "put", "move"]}}
    sg = query_to_subgoals(q)
    assert sg == [TypedStep("put", "soapbar", "cabinet")]


def test_query_subgoals_clean_then_put():
    q = {"extracted_keywords": {"objects": ["soapbar", "cabinet"],
                                "verbs": ["clean", "put"]}}
    sg = query_to_subgoals(q)
    assert sg == [TypedStep("clean", "soapbar", None), TypedStep("put", "soapbar", "cabinet")]


def test_order_matters_for_align_not_bag():
    # Two sub-goals clean -> put
    subgoals = [TypedStep("clean", "soapbar", None), TypedStep("put", "soapbar", "cabinet")]
    right = [TypedStep("clean", "soapbar", "sinkbasin"), TypedStep("put", "soapbar", "cabinet")]
    wrong = [TypedStep("put", "soapbar", "cabinet"), TypedStep("clean", "soapbar", "sinkbasin")]
    # ordered alignment should prefer the correctly-ordered trajectory
    assert ordered_align_cost(subgoals, right) < ordered_align_cost(subgoals, wrong)
    # bag-of-subgoals is blind to order: both score the same
    assert bag_cost(subgoals, right) == bag_cost(subgoals, wrong)


def test_align_rewards_coverage():
    subgoals = [TypedStep("clean", "soapbar", None), TypedStep("put", "soapbar", "cabinet")]
    covers_both = [TypedStep("take", "soapbar", "countertop"),
                   TypedStep("clean", "soapbar", "sinkbasin"),
                   TypedStep("go_to", "cabinet", None),
                   TypedStep("put", "soapbar", "cabinet")]
    missing_clean = [TypedStep("take", "soapbar", "countertop"),
                     TypedStep("put", "soapbar", "cabinet")]
    assert ordered_align_cost(subgoals, covers_both) < ordered_align_cost(subgoals, missing_clean)


def test_real_data_smoke():
    """End-to-end on the actual benchmark files if present."""
    import os
    from evaluation.procedural_memory.stage1_align import (
        load_queries, load_typed_trajectories, rank, relevant_ids,
    )
    base = os.path.join(os.path.dirname(__file__), "..", "data", "proced_mem_bench")
    qp = os.path.join(base, "queries.json")
    tp = os.path.join(base, "trajectories.json")
    if not (os.path.exists(qp) and os.path.exists(tp)):
        print("SKIP real_data_smoke (data not found)")
        return
    queries = load_queries(qp)
    typed = load_typed_trajectories(tp)
    assert len(queries) == 40
    assert len(typed) == 336
    # every trajectory parsed at least some steps
    parsed = sum(1 for s in typed.values() if s)
    assert parsed > 300, f"only {parsed}/336 trajectories parsed any steps"
    # ranking returns something for the first query
    res = rank(queries[0], typed, scorer="align", top_k=10)
    assert len(res.ranked_ids) == 10
    print(f"  real_data_smoke OK: {parsed}/336 parsed, q0 top-1={res.ranked_ids[0]}, "
          f"relevant={sorted(relevant_ids(queries[0]))[:3]}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
