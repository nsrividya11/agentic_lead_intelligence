"""
Enrichment tool the agent calls during the 'act' step of its loop.

This is stubbed with local mock data so the whole pipeline runs with
zero external dependencies and zero cost. To go to production, replace
the body of `search_company_info` with a call to a real provider
(Clearbit, Apollo.io, People Data Labs, LinkedIn API, etc.) — the
function signature and return shape stay the same, so nothing else
in the agent needs to change.
"""

import json
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_leads.json")

with open(_DATA_PATH, encoding="utf-8") as f:
    _SEED_LEADS = {lead["email"]: lead for lead in json.load(f)}


def search_company_info(email: str) -> dict:
    """
    Look up firmographic + signal data for a lead by email.

    SWAP POINT: replace this lookup with a real enrichment API call.
    Keep the return shape: {name, company, title, raw_signal, found}.
    """
    lead = _SEED_LEADS.get(email)
    if lead is None:
        return {"found": False, "email": email}
    return {**lead, "found": True}
