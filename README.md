# Lead Intelligence Agent

**Submission for Techvruk's AI Agentic System Challenge.**

## What this is

An AI agent that takes **any free-text lead-generation goal** — e.g. *"Find 5 Indian ed-tech companies with 100+ employees that are hiring AI engineers"* — and autonomously:

1. **Understands** the goal: an LLM turns it into a structured task spec (search queries, target count, scoring criteria, weights). Nothing is hardcoded — ask for colleges, startups, people, or anything else, and the criteria are derived fresh from what you actually asked.
2. **Plans** the concrete steps needed for that specific goal.
3. **Finds** real candidates via live web search (multiple query variants, merged and deduped).
4. **Enriches** each candidate with a second, targeted real web search, then has the LLM extract structured firmographic fields from the raw results.
5. **Observes** whether each tool call succeeded, and handles failure gracefully — a candidate with no data, or an LLM call that fails, is marked unscored instead of crashing the run.
6. **Scores** each enriched candidate against the criteria derived from the *original* goal, with a reasoned justification.

It returns a ranked, reasoned leaderboard. There is no seed lead list, no static ICP file, and no mock data anywhere in the pipeline — every result comes from a live search and a live LLM call.

## Why this problem

Techvruk's own product is an opportunity-matching platform — connecting professionals to opportunities. A lead-finding-and-scoring agent is the same core capability wearing a different hat: given an open-ended goal, discover real candidates and decide how good each match is, with reasoning attached.

## Architecture

```
run_pipeline(user_goal)
  understand_step(goal)   -> LLM derives {search_queries, target_count, criteria, scoring_weights}
  plan_step(spec)          -> prints the concrete steps for THIS goal
  find_step(spec)           -> live web search (Tavily), multiple query variants, merged + deduped
  for each raw search result (until target_count reached):
    extract_candidate_step() -> LLM checks if this result names a real, relevant candidate
    enrich_step()             -> second live search + LLM extracts structured fields
    score_step()               -> LLM scores against the goal's own criteria
  -> rank by score, unscored candidates listed separately
```

- **State**: the task spec, seen-candidate set, and accumulating results list are all threaded through the loop — the agent's context persists across every step of a run.
- **Tool use**: `tools/search.py` (Tavily) is called twice per candidate (find + enrich) with different, purpose-built queries — a real, swappable tool contract.
- **Error handling**: search failures, unparseable LLM responses, and full LLM-quota exhaustion are all caught per-candidate so one failure doesn't take down the whole batch. A daily free-tier quota error is detected and skipped immediately rather than retried.

## Real data, not mock data

Both external calls are live:

- **Search**: [Tavily](https://tavily.com) — a free-tier search API purpose-built for LLM agents. `tools/search.py` calls it twice per candidate: once (via multiple query variants) to discover candidates, once per candidate to enrich it.
- **Reasoning**: [Groq](https://groq.com) as the primary provider, with [OpenRouter](https://openrouter.ai) free-tier (`:free`) models as fallback — both free, per contest fairness guidelines.

Nothing about the entity type, criteria, or scoring rubric is hardcoded — they're all derived by the LLM from whatever the user typed.

## LLM: Groq primary, OpenRouter fallback, both free-tier

`tools/llm.py` calls a two-tier provider chain in priority order — Groq models first, then OpenRouter's free models:
1. Try the current model.
2. On a **transient** rate limit (429/502/embedded provider error), wait briefly and retry once.
3. On a **daily quota exhaustion** (OpenRouter's "free-models-per-day", or Groq's "rate_limit_exceeded"), skip the retry — it won't clear within the run — and fall back immediately.
4. If a model fails either way, rotate to the next model; once every Groq model is exhausted, the chain continues into OpenRouter's free models automatically.

Groq is tried first because its free tier has substantially higher daily limits than OpenRouter's free models, so it absorbs the bulk of normal usage; OpenRouter is the safety net if Groq has an outage or its own limits are hit.

> OpenRouter's free tier caps out at roughly 50 requests/day account-wide unless a small credit balance is added (their own error message: *"Add 10 credits to unlock 1000 free model requests per day"*). Heavy testing in one day can exhaust it — this is a real constraint of the free tier, not a bug, and is exactly why Groq is the primary provider.
>
> Both providers' free-model catalogs change over time. If a model slug in `GROQ_MODELS` or `OPENROUTER_FREE_MODELS` (`tools/llm.py`) starts 404ing, refresh it via `GET https://api.groq.com/openai/v1/models` or `GET https://openrouter.ai/api/v1/models` respectively.

## Setup & run (CLI)

```bash
cd lead-agent
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
```

- Get a free Groq key at [console.groq.com/keys](https://console.groq.com/keys) — no payment method required.
- Get a free OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys) — no payment method required for free-tier models.
- Get a free Tavily key at [tavily.com](https://tavily.com) — no card required for the free tier.

Run with any goal:

```bash
python agent.py "Find 5 Indian ed-tech companies with 100+ employees that are hiring AI engineers"
```

Or run without an argument and it will prompt you:

```bash
python agent.py
```

## Run the web UI locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a browser UI: type any goal, click "Run agent," and see the live reasoning log plus the final ranked leaderboard.

## Live deployment

Deployed on [Render](https://render.com)'s free tier: **https://agentic-lead-intelligence.onrender.com**

> Free-tier services spin down after 15 minutes of inactivity — the first request after idle can take 30-50 seconds to wake up.

To deploy your own copy:
1. Push this repo to GitHub.
2. On Render, create a new Web Service from the repo — `render.yaml` in this directory configures the build/start commands automatically.
3. Add `OPENROUTER_API_KEY` and `TAVILY_API_KEY` as environment variables in Render's dashboard (never commit them — `render.yaml` deliberately marks them `sync: false` so Render prompts for them instead of reading them from the repo).

## Tests

```bash
pip install pytest

# Fast, free, fully mocked unit tests (search/LLM fallback logic, agent orchestration)
python -m pytest tests/ --ignore=tests/test_smoke_openrouter.py --ignore=tests/test_smoke_search.py -v

# Real smoke tests — hit the live APIs with real payloads.
# Run these once before recording the demo / redeploying, not on every change.
python tests/test_smoke_openrouter.py
python tests/test_smoke_search.py
```

27 unit tests cover:
- `tools/llm.py`: first-model success, fallback on error, transient-429 retry-then-fallback, daily-quota-exhaustion skip-retry, embedded-error-in-200-response detection, all-models-exhausted, malformed responses, network exceptions — all mocked
- `agent.py`: every pipeline stage individually (understand/plan/find/extract/enrich/score), multi-query merge/dedupe, per-candidate error isolation (one candidate's total LLM failure doesn't crash the batch), and full `run_pipeline()` runs

The smoke tests hit the real APIs and check: keys authenticate, live search returns relevant results, at least one free LLM model is reachable, and a full `score_step()` call against a live LLM returns a valid 0–100 score.

## Sample output

```
[UNDERSTAND] Parsing goal into a structured task spec ...
[UNDERSTAND] search_queries=['list of Indian ed-tech companies hiring AI engineers', 'top Indian education technology companies AI engineer jobs', ...], target_count=4, criteria=['Indian', 'ed-tech', '100+ employees', 'hiring AI engineers']

[PLAN] Find, enrich, and rank 4 company(s)
       1. Search the web for candidates using 4 query variant(s): ...
       2. For each candidate, run a targeted follow-up search to enrich it
       3. Extract structured fields from search results (LLM)
       4. Score each candidate against: Indian, ed-tech, 100+ employees, hiring AI engineers
       5. Rank and return the leaderboard

[FIND] Searching the web for candidates using 4 query variant(s) ...
[OBSERVE] 14 unique raw search results across all queries
[ACT]  Enriching candidate: CommLab India ...
[OBSERVE] Enriched CommLab India: industry=Rapid eLearning Solutions, size=None
[RESPOND] Scoring CommLab India ...
...

======================================================================
FINAL RANKED RESULTS
======================================================================
1. CommLab India — score: 50
   The lead is Indian and operates in the ed-tech sector, but size and AI engineer
   hiring signals are not confirmed.
2. Arivihan Technologies — score: 50
   Matches Indian and ed-tech criteria, but size is below 100+ employees and no
   hiring AI engineers signal.
3. Edmo — score: 25
   The lead matches the Indian criterion (Bangalore, IN) but fails on ed-tech
   (Software Development industry), 100+ employees (11-50 employees), and hiring
   AI engineers (no hiring signal).
```

Scores are intentionally not inflated — when real data doesn't clearly support a criterion, the agent says so.

## Path to production

1. Swap Tavily for a paid/higher-limit search provider once volume justifies the cost.
2. Add caching so re-running a similar goal doesn't re-search/re-enrich identical candidates.
3. Swap free-tier OpenRouter models for a paid tier once volume justifies the cost — the fallback architecture in `tools/llm.py` means this is a one-line config change, not a rewrite.
4. Persist results to a database instead of printing them, and expose the ranked leaderboard via a proper API/dashboard for multi-user use.

## Files

```
lead-agent/
├── agent.py              # orchestrator: understand/plan/find/enrich/score pipeline
├── app.py                 # Streamlit web UI wrapping agent.py
├── render.yaml             # Render deployment config
├── tools/
│   ├── search.py            # Tavily web search tool (find + enrich)
│   ├── llm.py                # OpenRouter client with free-model fallback + retry
│   └── json_utils.py          # shared LLM-JSON-response parsing helper
├── tests/
│   ├── test_agent.py            # mocked unit tests for the pipeline
│   ├── test_llm.py               # mocked unit tests for the LLM client
│   ├── test_smoke_openrouter.py   # real smoke test against live OpenRouter
│   └── test_smoke_search.py        # real smoke test against live Tavily
├── requirements.txt
└── .env                   # API keys (gitignored — never committed)
```
