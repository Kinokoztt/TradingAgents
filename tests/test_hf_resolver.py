"""Tests for the live HF resolver's pure logic (ranking + term derivation)."""

from __future__ import annotations

import pytest

from tradingagents.llm_clients.hf_resolver import (
    Candidate,
    effective_match_terms,
    effective_prefer_terms,
    effective_query,
    rank_candidates,
)
from tradingagents.llm_clients.local_models import LocalModelSpec

pytestmark = pytest.mark.unit


def _spec(**kw) -> LocalModelSpec:
    base = dict(served_name="qwen3-32b", hf_repo="Qwen/Qwen3-32B-AWQ", params="32B",
                quant="AWQ-INT4", approx_vram_gb=20)
    base.update(kw)
    return LocalModelSpec(**base)


def test_effective_terms_derived_from_served_name_and_quant():
    spec = _spec()
    assert effective_query(spec) == "qwen3 32b"
    assert effective_match_terms(spec) == ("qwen3", "32b")  # quant token dropped
    assert set(effective_prefer_terms(spec)) == {"awq", "int4"}


def test_effective_terms_explicit_override():
    spec = _spec(hf_query="Qwen3 32B AWQ", match_terms=("Qwen3", "32B"), prefer_terms=("AWQ",))
    assert effective_query(spec) == "Qwen3 32B AWQ"
    assert effective_match_terms(spec) == ("qwen3", "32b")
    assert effective_prefer_terms(spec) == ("awq",)


def test_rank_filters_by_match_terms():
    cands = [
        Candidate("Qwen/Qwen3-32B-AWQ", downloads=100),
        Candidate("someone/llama-3-8b", downloads=999),  # missing match terms
    ]
    ranked = rank_candidates(cands, ("qwen3", "32b"), ("awq",))
    assert [c.repo_id for c in ranked] == ["Qwen/Qwen3-32B-AWQ"]


def test_rank_prefers_quant_then_downloads():
    cands = [
        Candidate("a/Qwen3-32B", downloads=500),          # no quant token
        Candidate("b/Qwen3-32B-AWQ", downloads=10),        # quant match wins despite fewer dl
        Candidate("c/Qwen3-32B-AWQ-v2", downloads=300),    # quant match + more dl
    ]
    ranked = rank_candidates(cands, ("qwen3", "32b"), ("awq",))
    assert ranked[0].repo_id == "c/Qwen3-32B-AWQ-v2"  # prefer hit + highest downloads
    assert ranked[1].repo_id == "b/Qwen3-32B-AWQ"
    assert ranked[-1].repo_id == "a/Qwen3-32B"        # no quant token -> last


def test_rank_empty_when_nothing_matches():
    assert rank_candidates([Candidate("x/y-7b")], ("qwen3",), ()) == []
