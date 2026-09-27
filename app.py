"""
Streamlit UI for the Lead Intelligence Agent.

Lets a user type ANY free-text lead-generation goal, runs the real
understand -> plan -> find -> enrich -> score pipeline (agent.py)
against live web search + a live LLM, and shows LIVE progress as each
phase runs (via agent.run_pipeline's on_phase callback) plus the full
reasoning log and final ranked leaderboard.
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

# Icons for each checklist line's state.
_ICONS = {"pending": "⚪", "active": "🔵", "done": "✅", "skip": "⏭️", "error": "⚠️"}

_PHASE_LABELS = {
    "understand": "Understand goal — deriving search strategy and scoring criteria",
    "plan": "Plan — deciding the concrete steps for this goal",
    "find": "Find — searching the web for candidate leads",
}


def _render_checklist(placeholder, phase_state, candidate_lines):
    """Redraw the whole live checklist from current state."""
    lines = []
    for phase in ("understand", "plan", "find"):
        state = phase_state.get(phase, "pending")
        icon = _ICONS.get(state, "⚪")
        detail = phase_state.get(f"{phase}_detail", "")
        suffix = f" — {detail}" if detail else ""
        lines.append(f"{icon} **{_PHASE_LABELS[phase]}**{suffix}")

    if candidate_lines:
        lines.append("")
        lines.append("**Enrich & score candidates:**")
        for name, (state, detail) in candidate_lines.items():
            icon = _ICONS.get(state, "⚪")
            suffix = f" — {detail}" if detail else ""
            lines.append(f"{icon} {name}{suffix}")

    placeholder.markdown("\n\n".join(lines))


if run_clicked:
    checklist_placeholder = st.empty()
    log_buffer = io.StringIO()

    phase_state = {"understand": "active"}
    candidate_lines = {}
    _render_checklist(checklist_placeholder, phase_state, candidate_lines)

    def on_phase(phase, status, detail="", candidate_name=None):
        if phase == "candidate":
            candidate_lines[candidate_name] = (status, detail if status != "start" else "")
        else:
            if status == "start":
                phase_state[phase] = "active"
            elif status == "done":
                phase_state[phase] = "done"
                phase_state[f"{phase}_detail"] = detail
                # Move to the next phase's "active" state immediately for
                # a smoother feel instead of a flat moment between phases.
                next_phase = {"understand": "plan", "plan": "find"}.get(phase)
                if next_phase:
                    phase_state[next_phase] = "active"

        _render_checklist(checklist_placeholder, phase_state, candidate_lines)

    with st.spinner("Agent is working — this can take 1-2 minutes ..."):
        try:
            with contextlib.redirect_stdout(log_buffer):
                results = run_pipeline(goal.strip(), on_phase=on_phase)
        except RuntimeError as exc:
            st.error(
                f"Could not complete the run: {exc}\n\n"
                "This usually means the free-tier LLM quota is exhausted for today — try again later."
            )
            results = None
        finally:
            with st.expander("Full agent reasoning log (plan → find → enrich → score)", expanded=(results is None)):
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
