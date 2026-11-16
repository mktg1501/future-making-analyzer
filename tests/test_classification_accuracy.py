"""
Regression check against the paper's labeled examples. Calls OpenAI.

Run with:
    RUN_LLM_TESTS=1 python -m pytest -s tests/test_classification_accuracy.py
"""
import os

import pytest

from motor import analyze_comment, PF_EV, DEMO_EXAMPLES

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="Set RUN_LLM_TESTS=1 to run tests that call OpenAI.",
)


@pytest.mark.parametrize("ex", DEMO_EXAMPLES, ids=[e["id"] for e in DEMO_EXAMPLES])
def test_paper_example(ex):
    ref = ex["referencia"]
    result = analyze_comment(
        PF_EV,
        ex["comment"],
        ex["context"],
        ex["context_type"],
        False,   # is_consultation
        True,    # allow_secondary
    )
    assert result["main_activity"] == ref["activity"]
    assert result["activity_subtype"] == ref["subtype"]
    assert result["main_orientation"] == ref["orientation"]
