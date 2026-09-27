"""Unit tests for tools/enrichment.py — the mocked lookup tool."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.enrichment import search_company_info


def test_known_email_returns_full_record():
    result = search_company_info("aditi.rao@brightwave.io")
    assert result["found"] is True
    assert result["name"] == "Aditi Rao"
    assert result["company"] == "Brightwave Analytics"
    assert result["title"] == "VP of Engineering"
    assert "raw_signal" in result


def test_unknown_email_returns_not_found():
    result = search_company_info("nobody@nowhere.example")
    assert result["found"] is False
    assert result["email"] == "nobody@nowhere.example"
    # Must NOT contain lead fields when not found — agent's observe_step
    # relies on this shape to decide whether to skip scoring.
    assert "name" not in result
    assert "company" not in result


def test_all_seed_leads_are_individually_lookupable():
    import json

    seed_path = os.path.join(os.path.dirname(__file__), "..", "data", "seed_leads.json")
    with open(seed_path, encoding="utf-8") as f:
        seed_leads = json.load(f)

    assert len(seed_leads) >= 1
    for lead in seed_leads:
        result = search_company_info(lead["email"])
        assert result["found"] is True
        assert result["email"] == lead["email"]


def test_return_shape_matches_contract():
    """The swap point contract: {name, company, title, raw_signal, found}."""
    result = search_company_info("aditi.rao@brightwave.io")
    required_keys = {"name", "company", "title", "raw_signal", "found"}
    assert required_keys.issubset(result.keys())


if __name__ == "__main__":
    test_known_email_returns_full_record()
    test_unknown_email_returns_not_found()
    test_all_seed_leads_are_individually_lookupable()
    test_return_shape_matches_contract()
    print("All enrichment tests passed.")
