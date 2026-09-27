"""Shared helper for parsing JSON out of LLM responses, which are often
wrapped in markdown code fences or preceded by chatty text."""

import json
import re


def parse_llm_json(raw: str) -> dict | None:
    """Extract and parse a JSON object from an LLM response. Returns
    None if no valid JSON object could be found."""
    text = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
