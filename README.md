# Lead Finder + Enrichment + Scoring Agent

**Submission for Techvruk's AI Agentic System Challenge.**

## What this is

An AI agent that takes a raw lead (name, email, company) and autonomously:

1. **Plans** what information it needs to evaluate the lead.
2. **Acts** by calling an enrichment tool to fetch firmographic and signal data.
3. **Observes** whether the tool call succeeded, and handles the failure case gracefully (skips scoring instead of crashing).
4. **Responds** by asking an LLM to score the lead against an Ideal Customer Profile (ICP) and justify the score.

It then ranks every lead it processed and prints a final, reasoned leaderboard.

## Why this problem

Techvruk's own product is an opportunity-matching platform — connecting professionals to opportunities. A lead-scoring agent is the same core capability wearing a different hat: given a person/entity and a target profile, decide how good the match is and explain why.

This project is built as **a first-pass prototype for a Techvruk-style opportunity-matching engine**, not just a generic agent demo. The `data/icp.json` config is written from Techvruk's own hiring/visibility angle, and the enrichment tool's interface is deliberately shaped so it could sit behind Techvruk's own candidate/opportunity-matching feature with no redesign — only the data source needs to change.

## Architecture

```
run_pipeline()
  for each lead:
    plan_step()     -> decide what's needed to score this lead
    act_step()       -> call search_company_info() [tool]
    observe_step()   -> check success; skip gracefully on failure
    respond_step()   -> LLM reasons about ICP fit, returns {score, justification}
  -> rank all leads by score, print leaderboard
```

- **State**: each lead's enrichment result and verdict are carried through the loop and collected into a shared `results` list — the agent's context persists across steps, not just within one call.
- **Tool use**: `tools/enrichment.py` is a real tool call in the agentic sense — swappable, single-responsibility, contract-based.
- **Error handling**: if enrichment finds nothing (`found: False`), the agent doesn't crash or hallucinate a score — it marks the lead unscored and explains why. If the LLM call is rate-limited, the client retries once, then rotates to the next free model.

## Why mock data, not a live API

The enrichment tool (`tools/enrichment.py`) is stubbed with local seed data (`data/seed_leads.json`) instead of calling a real provider (Clearbit, Apollo, LinkedIn API). This is deliberate:

- **Zero cost, zero flake risk.** A live scraping/API call could fail live during the demo for reasons unrelated to the agent's logic (rate limits, auth, network). The agent's reasoning is what's being evaluated, not a third-party API's uptime.
- **The swap point is explicit.** `search_company_info(email)` has one job and one contract: `{name, company, title, raw_signal, found}`. Point it at a real API and nothing else in the agent changes.

## LLM: OpenRouter, free-tier only

All reasoning calls go through [OpenRouter](https://openrouter.ai), using **only free-tier (`:free`) models** — no paid usage, per contest fairness guidelines.

To handle free-tier rate limits gracefully (a real constraint, not a hypothetical), `tools/llm.py`:
1. Tries the current model.
2. On a 429 (rate limit), waits briefly and retries once.
3. If it still fails, rotates to the next free model in the list.

This fallback rotation is itself part of the agent's **workflow robustness** — the same principle as retrying a flaky tool call, applied to the LLM provider itself.

> OpenRouter's free-model catalog changes over time. If a model slug in `FREE_MODELS` (`tools/llm.py`) starts 404ing, refresh it via:
> `GET https://openrouter.ai/api/v1/models` → filter for `id` ending in `:free`.

## Setup & run

```bash
cd lead-agent
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file (already present in this submission for judging convenience, gitignored otherwise):

```
OPENROUTER_API_KEY=your_key_here
```

Get a free key at [openrouter.ai](https://openrouter.ai/keys) — no payment method required for free-tier models.

Run:

```bash
python agent.py
```

## Tests

```bash
pip install pytest

# Fast, free, fully mocked unit tests (enrichment, LLM fallback logic, agent orchestration)
python -m pytest tests/ --ignore=tests/test_smoke_openrouter.py -v

# Real smoke test — hits the live OpenRouter API with real payloads.
# Run this once before recording the demo / submitting, not on every change.
python tests/test_smoke_openrouter.py
```

19 unit tests cover:
- `tools/enrichment.py`: known/unknown lookups, return-shape contract
- `tools/llm.py`: first-model success, fallback on error, 429 retry-then-fallback, all-models-exhausted, malformed responses, network exceptions — all mocked, no real API calls
- `agent.py`: each pipeline stage (plan/act/observe/respond) individually, JSON parsing (including markdown-fenced and malformed LLM output), and a full `run_pipeline()` run verifying scored leads always rank above unscored ones

The smoke test hits the real API and checks: the key authenticates, at least one free model is currently reachable, `call_llm()` works against live traffic, and a full `respond_step()` call on a real seed lead returns a valid 0–100 score.

## Sample output

```
======================================================================
Processing lead: Aditi Rao <aditi.rao@brightwave.io>

[PLAN] Score lead aditi.rao@brightwave.io against Techvruk's ICP
       1. Enrich lead with firmographic + signal data
       2. Verify enrichment succeeded (observe)
       3. Score against ICP using LLM reasoning
       4. Return structured verdict with justification
[ACT]  Calling enrichment tool for aditi.rao@brightwave.io ...
[OBSERVE] Enrichment succeeded: Brightwave Analytics / VP of Engineering
[RESPOND] Asking LLM to score lead against ICP ...
...
======================================================================
FINAL RANKED RESULTS
======================================================================
1. Aditi Rao (aditi.rao@brightwave.io) — score: 98
   Aditi Rao is a VP of Engineering, an exact target title, and her signal explicitly
   mentions scaling a 40-person engineering team and hiring interns for Q4...
2. Priya Nair (priya@loomworks.co) — score: 92
   ...
5. Marcus Chen (m.chen@fieldstonecap.com) — score: 15
   Marcus Chen is an Analyst, not a target title, and the signal only shows general
   interest in AI hiring tools...
```

## Path to production

To take this from prototype to something Techvruk could actually run:

1. Replace `tools/enrichment.py`'s mock lookup with a real provider (Apollo.io, People Data Labs, or Techvruk's own user/opportunity database).
2. Replace `data/icp.json` with a per-company or per-recruiter configurable ICP.
3. Swap free-tier OpenRouter models for a paid tier or dedicated model once volume justifies the cost — the fallback architecture in `tools/llm.py` means this is a one-line config change, not a rewrite.
4. Persist results to a database instead of stdout, and expose the ranked leaderboard via an API/dashboard.

## Files

```
lead-agent/
├── agent.py              # orchestrator: plan/act/observe/respond loop
├── tools/
│   ├── enrichment.py      # mocked enrichment tool (swap point for real API)
│   └── llm.py             # OpenRouter client with free-model fallback + retry
├── data/
│   ├── seed_leads.json    # 5 mock leads
│   └── icp.json           # Techvruk-style ideal customer profile
├── requirements.txt
└── .env                   # OpenRouter API key (gitignored in real use)
```
