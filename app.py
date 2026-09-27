"""
Streamlit UI for the Lead Intelligence Agent.

Lets a user type ANY free-text lead-generation goal, runs the real
understand -> plan -> find -> enrich -> score pipeline (agent.py)
against live web search + a live LLM, and shows both the agent's live
step-by-step reasoning log and the final ranked leaderboard.
"""

import contextlib
import io

import streamlit as st

from agent import run_pipeline

st.set_page_config(page_title="Lead Intelligence Agent", page_icon="🎯", layout="wide")

st.title("🎯 Lead Intelligence Agent")
st.caption(
    "Real web search + real LLM reasoning. Ask for any kind of lead — "
    "no hardcoded companies, ICPs, or criteria."
)

with st.expander("Example goals to try"):
    st.markdown(
        "- Find 5 Indian ed-tech companies with 100+ employees that are hiring AI engineers\n"
        "- Find 4 fintech startups in the US that recently raised a Series A\n"
        "- Find 3 universities in Hyderabad with a computer science department"
    )

goal = st.text_area(
    "What kind of leads are you looking for?",
    placeholder="e.g. Find 5 Indian ed-tech companies with 100+ employees that are hiring AI engineers",
    height=80,
)

run_clicked = st.button("Run agent", type="primary", disabled=not goal.strip())

if run_clicked:
    log_placeholder = st.empty()
    log_buffer = io.StringIO()

    with st.spinner("Agent is planning, searching, and scoring — this can take 1-2 minutes ..."):
        try:
            with contextlib.redirect_stdout(log_buffer):
                results = run_pipeline(goal.strip())
        except RuntimeError as exc:
            st.error(
                f"Could not complete the run: {exc}\n\n"
                "This usually means the free-tier LLM quota is exhausted for today — try again later."
            )
            results = None
        finally:
            with st.expander("Agent reasoning log (plan → find → enrich → score)", expanded=(results is None)):
                st.code(log_buffer.getvalue() or "(no output captured)", language=None)

    if results:
        st.subheader("Ranked results")
        scored = [r for r in results if r.get("score") is not None]
        unscored = [r for r in results if r.get("score") is None]

        if scored:
            st.dataframe(
                [
                    {"Rank": i + 1, "Name": r["name"], "Score": r["score"], "Justification": r["justification"]}
                    for i, r in enumerate(scored)
                ],
                use_container_width=True,
                hide_index=True,
            )
        if unscored:
            st.caption("Unscored (enrichment or scoring failed for these):")
            for r in unscored:
                st.write(f"- **{r['name']}** — {r['justification']}")
    elif results == []:
        st.warning("No candidates found for this goal. Try rephrasing or broadening it.")
