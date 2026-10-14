"""
Turn KPI facts into a Weekly Business Review narrative.

Two narrators share one interface:
  * RuleNarrator - deterministic templates, zero dependencies, always available.
  * LLMNarrator  - any chat model through LangChain, wrapped in a *number guard*:
                   every figure in the LLM's draft must trace back to the facts
                   payload, otherwise the draft is rejected and we fall back.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol

SYSTEM_PROMPT = """You are a senior business analyst writing the Weekly Business Review for a \
multi-city food-delivery business. Audience: city heads and the CXO team.

Rules:
- Use ONLY numbers that appear in the FACTS JSON. Never compute new numbers or estimate.
- Quote numbers exactly as formatted in the JSON `display` fields.
- Lead with the headline, then the top 3 issues ranked by business impact, then actions.
- For each issue give: what moved, the likely driver (from the bridge / related anomalies), \
and one concrete next step with an owner team (Ops, Growth, Category, Product).
- Plain markdown. Max 220 words. No preamble."""

USER_PROMPT = "FACTS:\n```json\n{facts}\n```\nWrite the WBR."

NUM_RE = re.compile(r"(?<![\w.])[-+−]?₹?\d[\d,]*(?:\.\d+)?\s?(?:%|pp|Cr|L|k|min)?")


class Narrator(Protocol):
    name: str

    def write(self, facts: dict) -> str: ...


# --------------------------------------------------------------------------- #
# Number guard
# --------------------------------------------------------------------------- #
def _norm(tok: str) -> str:
    return re.sub(r"[\s₹,+−-]", "", tok).lower()


def allowed_numbers(facts: dict) -> set[str]:
    """Every number that appears anywhere in the facts payload's display strings."""
    blob = json.dumps(facts, ensure_ascii=False)
    return {_norm(m) for m in NUM_RE.findall(blob)}


def ungrounded_numbers(text: str, facts: dict) -> list[str]:
    ok = allowed_numbers(facts)
    bad = []
    for tok in NUM_RE.findall(text):
        n = _norm(tok)
        if not n or re.fullmatch(r"\d", n):   # list markers, "top 3", etc.
            continue
        if n not in ok:
            bad.append(tok.strip())
    return bad


# --------------------------------------------------------------------------- #
# Narrators
# --------------------------------------------------------------------------- #
class RuleNarrator:
    name = "rules"

    def write(self, facts: dict) -> str:
        h = facts["headline"]
        lines = [
            f"**Headline:** GMV {h['gmv']['display']} ({h['gmv']['wow_display']} WoW), "
            f"orders {h['orders']['display']} ({h['orders']['wow_display']}), "
            f"AOV {h['aov']['display']} ({h['aov']['wow_display']}).",
            "",
            "**Top issues**",
        ]
        for i, a in enumerate(facts["issues"][:3], 1):
            lines.append(f"{i}. **{a['city']} – {a['metric_label']}** {a['direction_word']} to "
                         f"{a['current_display']} vs {a['baseline_display']} baseline "
                         f"({a['change_display']}, impact {a['impact_display']}). {a['driver']} → *{a['action']}*")
        if not facts["issues"]:
            lines.append("No metric breached its anomaly threshold this week.")
        lines += ["", "**GMV bridge (WoW)**"]
        for b in facts["gmv_bridge"]:
            lines.append(f"- {b['city']}: {b['gmv_change_display']} "
                         f"(orders {b['orders_effect_display']}, AOV {b['aov_effect_display']})")
        return "\n".join(lines)


class LLMNarrator:
    """
    `llm` is any callable (system, user) -> str. Use `LLMNarrator.from_langchain()`
    for a real model, or pass a fake in tests.
    """
    name = "llm"

    def __init__(self, llm: Callable[[str, str], str], fallback: Narrator | None = None):
        self.llm = llm
        self.fallback = fallback or RuleNarrator()
        self.last_rejection: list[str] = []

    @classmethod
    def from_langchain(cls) -> "LLMNarrator":
        """
        Build from environment:
          LLM_PROVIDER = openai | anthropic   (default: openai)
          LLM_MODEL    = model name           (provider default if unset)
          and the provider's usual API key env var.
        """
        from langchain_core.prompts import ChatPromptTemplate

        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model_name = os.getenv("LLM_MODEL")
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(model=model_name or "claude-sonnet-4-5", temperature=0)
        else:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])
        chain = prompt | model

        def call(system: str, user: str) -> str:
            return chain.invoke({"system": system, "user": user}).content

        return cls(call)

    def write(self, facts: dict) -> str:
        draft = self.llm(SYSTEM_PROMPT, USER_PROMPT.format(facts=json.dumps(facts, ensure_ascii=False, indent=1)))
        self.last_rejection = ungrounded_numbers(draft, facts)
        if self.last_rejection:
            return self.fallback.write(facts) + (
                "\n\n> _LLM draft rejected by number guard (ungrounded: "
                + ", ".join(self.last_rejection[:5]) + "). Showing rule-based narrative._")
        return draft


def pick_narrator(mode: str = "auto") -> Narrator:
    """auto: use the LLM if LangChain and an API key are available, else rules."""
    if mode == "rules":
        return RuleNarrator()
    try:
        has_key = any(os.getenv(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))
        if mode == "llm" or has_key:
            return LLMNarrator.from_langchain()
    except ImportError:
        if mode == "llm":
            raise
    return RuleNarrator()
