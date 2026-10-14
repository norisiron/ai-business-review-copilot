"""Run: pytest -q"""
import json

import numpy as np
import pytest

from wbr import metrics as m
from wbr.data import generate
from wbr.facts import build_facts, inr
from wbr.narrator import LLMNarrator, RuleNarrator, ungrounded_numbers
from wbr.report import slack_payload


@pytest.fixture(scope="module")
def weekly():
    return m.to_weekly(generate())


@pytest.fixture(scope="module")
def facts(weekly):
    return build_facts(weekly)


def test_gmv_bridge_adds_up_exactly(weekly):
    b = m.gmv_bridge(weekly)
    assert np.allclose(b.orders_effect + b.aov_effect, b.gmv_change, rtol=1e-9, atol=1e-3)


def test_planted_incidents_are_detected(weekly):
    hits = {(a.city, a.metric) for a in m.detect_anomalies(weekly)}
    assert ("Mumbai", "aov") in hits
    assert ("Kolkata", "cancel_rate") in hits
    assert ("Bengaluru", "promo_per_order") in hits


def test_no_high_severity_false_alarm_in_clean_city(weekly):
    assert not [a for a in m.detect_anomalies(weekly) if a.city == "Delhi NCR" and a.severity == "high"]


def test_issues_ranked_by_rupee_impact(facts):
    assert facts["issues"][0]["city"] == "Mumbai"


def test_rule_narrative_is_fully_grounded(facts):
    assert ungrounded_numbers(RuleNarrator().write(facts), facts) == []


def test_guard_accepts_grounded_llm_draft(facts):
    h = facts["headline"]["gmv"]
    draft = f"GMV landed at {h['display']} ({h['wow_display']} WoW). Top 3 issues below."
    n = LLMNarrator(lambda s, u: draft)
    assert n.write(facts) == draft
    assert n.last_rejection == []


def test_guard_rejects_hallucinated_number(facts):
    n = LLMNarrator(lambda s, u: "GMV grew 42.7% to ₹99.9 Cr on the back of a new campaign.")
    out = n.write(facts)
    assert "rejected by number guard" in out
    assert "42.7%" in n.last_rejection


def test_facts_and_slack_payload_are_serialisable(facts):
    json.dumps(facts, ensure_ascii=False)
    p = slack_payload(facts, "x")
    assert p["blocks"][0]["type"] == "header"
    assert "hooks.slack.com" not in json.dumps(p)


@pytest.mark.parametrize("x,expected", [(1.5e7, "₹1.50 Cr"), (2.3e5, "₹2.3 L"), (950, "₹950"), (-3e7, "-₹3.00 Cr")])
def test_inr_formatting(x, expected):
    assert inr(x) == expected
