"""Offline tests for the Stage-2 edit-script aligner. No LLM, no DB."""

from __future__ import annotations

from evaluation.procedural_memory.stage1_align import TypedStep
from evaluation.procedural_memory.stage2_gap import (
    _delete_subgoal,
    align_edit_script,
    gaps,
)


def test_covered_when_all_present():
    sg = [TypedStep("clean", "soapbar", None), TypedStep("put", "soapbar", "cabinet")]
    acts = [TypedStep("take", "soapbar", "countertop"),
            TypedStep("clean", "soapbar", "sinkbasin"),
            TypedStep("go_to", "cabinet", None),
            TypedStep("put", "soapbar", "cabinet")]
    script = align_edit_script(sg, acts)
    assert gaps(script) == [], f"expected no gaps, got {gaps(script)}"


def test_gap_detected_when_step_deleted():
    sg = [TypedStep("clean", "soapbar", None), TypedStep("put", "soapbar", "cabinet")]
    acts = [TypedStep("take", "soapbar", "countertop"),
            TypedStep("clean", "soapbar", "sinkbasin"),
            TypedStep("put", "soapbar", "cabinet")]
    ablated = _delete_subgoal(acts, TypedStep("clean", "soapbar", None))
    assert all(a.verb != "clean" for a in ablated)
    script = align_edit_script(sg, ablated)
    detected = gaps(script)
    assert any(g.verb == "clean" for g in detected), f"clean gap not found: {detected}"


def test_rebind_on_wrong_target():
    sg = [TypedStep("put", "soapbar", "cabinet")]
    acts = [TypedStep("take", "soapbar", "countertop"),
            TypedStep("put", "soapbar", "drawer")]  # wrong target
    script = align_edit_script(sg, acts)
    ops = [o.op for o in script]
    assert "REBIND" in ops, f"expected REBIND, got {ops}"


def test_delete_only_targeted_verb():
    acts = [TypedStep("clean", "soapbar", "sinkbasin"),
            TypedStep("heat", "apple", "microwave"),
            TypedStep("put", "soapbar", "cabinet")]
    ablated = _delete_subgoal(acts, TypedStep("clean", "soapbar", None))
    assert len(ablated) == 2
    assert any(a.verb == "heat" for a in ablated)   # unrelated step kept
    assert any(a.verb == "put" for a in ablated)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
