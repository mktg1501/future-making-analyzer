"""
Consumer Future-Making Analyzer - user interface.
Analysis logic lives in motor.py.
"""
import os
import hmac
import html
import json

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Consumer Future-Making Analyzer",
    page_icon="FM",
    layout="wide",
)

from motor import (
    HOMEPAGE_DESCRIPTION, INTERPRETIVE_USE_NOTE, PAPER_TITLE, PAPER_URL,
    CONTEXT_TYPE_CHOICES, ORIENTATIONS, ACTIVITY_META, CHALLENGE_DEFINITIONS,
    PF_EV, POLICY_ROADMAP_STEPS, MANAGER_ROADMAP_STEPS,
    POLICY_ORIENTATION_GUIDANCE, MANAGER_ORIENTATION_GUIDANCE,
    CHALLENGE_POLICY_GUIDANCE, CHALLENGE_MANAGER_GUIDANCE,
    MANAGER_MESSAGING_NOTE, CROSS_ORIENTATION_NOTE,
    analyze_comment, analyze_document, index_threads, build_context,
    build_comment_records_from_csv, build_comment_records_from_paragraphs,
    build_comment_records_from_user_labels, build_comment_records_from_consultation,
    compute_dominant_distributions, compute_challenge_review_candidates,
    compute_evenly_spaced_sample_indices,
    run_consistency_suite, run_framework_validation, _clean_enum,
    has_comment_text_column, detect_user_labeled_exchange,
)

MAX_COMMENTS = int(os.environ.get("MAX_COMENTARIOS", "300"))
EST_COST_PER_COMMENT = 0.00075


# ─────────────────────────────────────────
# PASSWORD GATE
# ─────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("Consumer Future-Making Analyzer")
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        st.error("The server is missing the APP_PASSWORD variable. Ask the administrator to set it.")
        st.stop()
    entered = st.text_input("Access password", type="password")
    if st.button("Enter"):
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


# ─────────────────────────────────────────
# SINGLE RESULT DISPLAY
# ─────────────────────────────────────────
def show_result(r: dict):
    ori = _clean_enum(r.get("main_orientation", "")).upper()
    act = _clean_enum(r.get("main_activity", "")).upper()
    sub = _clean_enum(r.get("activity_subtype", "")).upper()
    meta = ORIENTATIONS.get(ori, {})
    color = meta.get("color", "#333333")
    bg = meta.get("bg", "#FFFFFF")

    st.markdown(
        f"<div style='background:{bg};border-left:6px solid {color};"
        f"padding:12px 16px;border-radius:6px'>"
        f"<b>Orientation:</b> {html.escape(ori)} &nbsp;|&nbsp; "
        f"<b>Dominant activity:</b> {html.escape(act)} / {html.escape(sub)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Activity rationale**")
        st.write(str(r.get("activity_rationale", "")))
        st.markdown("**Orientation rationale**")
        st.write(str(r.get("orientation_rationale", "")))
    with col2:
        st.markdown("**Narrative identified:** " + str(r.get("narrative_identified", "")))
        st.markdown("**Emotions:** " + str(r.get("emotions_identified", "")))
        st.markdown("**Temporality:** " + str(r.get("temporality_identified", "")))
        st.markdown("**Conditions of adoption:** " + str(r.get("notable_conditions_of_adoption", "")))
        st.markdown("**Supporting text:** " + str(r.get("supporting_text", "")))

    secondary = r.get("secondary_classifications") or []
    if secondary:
        st.markdown("**Secondary classifications**")
        for s in secondary:
            if isinstance(s, dict):
                st.markdown(
                    f"- {s.get('activity', '')} / {s.get('activity_subtype', '')} "
                    f"({s.get('orientation', '')}): {s.get('rationale', '')}"
                )

    if r.get("context_note"):
        st.markdown("**Context note:** " + str(r.get("context_note")))
    if r.get("input_scope_warning"):
        st.warning("Input-scope note: " + str(r.get("input_scope_warning")))

    if ori in POLICY_ORIENTATION_GUIDANCE:
        with st.expander("Policy guidance for this orientation (Steps 2 and 4)"):
            g = POLICY_ORIENTATION_GUIDANCE[ori]
            st.write(g["lens"])
            st.write(g["implications"])
            st.markdown("**What to monitor:**")
            for m in g["monitor"]:
                st.markdown(f"- {m}")
            st.markdown(f"**Objective:** {g['objective']}")

        with st.expander("Managerial guidance for this orientation (Steps 2 and 4)"):
            g = MANAGER_ORIENTATION_GUIDANCE[ori]
            st.write(g["lens"])
            st.write(g["implications"])
            st.markdown("**What to monitor:**")
            for m in g["monitor"]:
                st.markdown(f"- {m}")
            st.markdown(f"**Objective:** {g['objective']}")
            st.markdown(f"**Avoid:** {g['avoid']}")

    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# TAB 1: SINGLE COMMENT
# ─────────────────────────────────────────
def tab_single():
    st.subheader("Analyze a single comment")
    pf = st.text_area("Prescribed future", value=PF_EV, height=100, key="pf_single")

    labels = [label for label, _, _ in CONTEXT_TYPE_CHOICES]
    label = st.selectbox("Context type", labels, key="ctx_single")
    ctype = next(code for lab, code, _ in CONTEXT_TYPE_CHOICES if lab == label)
    description = next(desc for lab, _, desc in CONTEXT_TYPE_CHOICES if lab == label)
    st.caption(description)

    context = st.text_area("Context (optional)", height=120, key="context_single")
    comment = st.text_area("Focal comment to classify", height=160, key="comment_single")

    if st.button("Classify", type="primary", key="btn_single"):
        if not comment.strip():
            st.warning("Please type or paste a comment before classifying.")
            return
        effective_ctype = ctype if context.strip() else "NONE"
        is_consultation = ctype == "CONSULTATION_PROMPT"
        with st.spinner("Analyzing the comment..."):
            try:
                r = analyze_comment(
                    pf.strip(), comment.strip(), context.strip(),
                    effective_ctype, is_consultation, None, False,
                )
            except Exception as e:
                st.error(f"The analysis could not be completed: {e}")
                return
        show_result(r)
        st.download_button(
            "Download result (JSON)",
            data=json.dumps(r, ensure_ascii=False, indent=2),
            file_name="result.json",
            mime="application/json",
        )


# ─────────────────────────────────────────
# TAB 2: MULTIPLE COMMENTS
# ─────────────────────────────────────────
def tab_multiple():
    st.subheader("Map orientations across several comments")
    pf = st.text_area("Prescribed future", value=PF_EV, height=100, key="pf_multi")

    source = st.radio(
        "Source of the comments",
        [
            "CSV file (column comment_text, text, or comment)",
            "Pasted text (comments separated by a blank line)",
            "Conversation with labels 'User 1:', 'User 2:', etc.",
        ],
        key="source_multi",
    )
    is_consultation = st.checkbox("These are responses to a public consultation", key="is_consult")
    consult_question = ""
    if is_consultation:
        consult_question = st.text_area(
            "Consultation question or proposal (used as context for every response)",
            height=100, key="consult_question",
        )

    records = []
    if source.startswith("CSV"):
        uploaded = st.file_uploader("Upload the CSV file", type=["csv"], key="csv_upload")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if not has_comment_text_column(df):
                    st.error("The CSV needs a column named comment_text, text, or comment.")
                else:
                    records = build_comment_records_from_csv(df)
            except Exception as e:
                st.error(f"The CSV could not be read: {e}")
    else:
        text = st.text_area("Paste the comments here", height=250, key="text_multi")
        if text.strip():
            if source.startswith("Conversation"):
                records = build_comment_records_from_user_labels(text)
                if not records:
                    st.warning("Found fewer than two 'User N:' labels. Check the format.")
            elif is_consultation:
                records = build_comment_records_from_consultation(text)
            else:
                records = build_comment_records_from_paragraphs(text)

    if not records:
        st.info("No comments to analyze yet.")
        return

    st.write(f"Comments detected: **{len(records)}**")
    if len(records) > MAX_COMMENTS:
        st.warning(
            f"Only the first {MAX_COMMENTS} comments will be analyzed, to control API costs. "
            "Split the file into parts if you need more."
        )
        records = records[:MAX_COMMENTS]

    with st.expander("Preview the comments"):
        st.dataframe(pd.DataFrame([
            {"ID": r["comment_id"], "Author": r.get("author", ""), "Comment": r["comment_text"]}
            for r in records
        ]), use_container_width=True)

    if not st.button("Analyze comments", type="primary", key="btn_multi"):
        return

    by_id, order = index_threads(records)
    for rec in records:
        ctx, ctype, _ = build_context(rec, by_id, order, consult_question, is_consultation)
        rec["context_text"] = ctx
        rec["context_type"] = ctype
        rec["is_consultation"] = is_consultation

    progress = st.progress(0, text="Starting analysis...")
    try:
        results = analyze_document(records, pf.strip(), None, progress)
    except Exception as e:
        st.error(f"An error occurred during the analysis: {e}")
        return
    progress.empty()

    author_by_id = {r["comment_id"]: r.get("author", "") for r in records}
    rows = []
    for r in results:
        rows.append({
            "ID": r.get("_comment_id", ""),
            "Author": author_by_id.get(r.get("_comment_id"), ""),
            "Comment": r.get("_comment_text", ""),
            "Orientation": _clean_enum(r.get("main_orientation", "")),
            "Activity": _clean_enum(r.get("main_activity", "")),
            "Subtype": _clean_enum(r.get("activity_subtype", "")),
            "Supporting text": r.get("supporting_text", ""),
            "Error": r.get("_error", ""),
        })
    df_result = pd.DataFrame(rows)

    st.success(f"Analysis complete: {len(results)} comments.")
    st.dataframe(df_result, use_container_width=True)
    st.download_button(
        "Download results (CSV)",
        data=df_result.to_csv(index=False).encode("utf-8-sig"),
        file_name="orientation_results.csv",
        mime="text/csv",
    )

    dist = compute_dominant_distributions(results)
    st.markdown("### Distribution")
    if dist["orientation_counts"]:
        st.bar_chart(pd.Series(dist["orientation_counts"], name="Comments"))
    if dist["activity_counts"]:
        st.write("Activities:", dist["activity_counts"])

    st.markdown("### Human review of future-making challenges")
    st.caption(
        "These are leads for manual review, not automatic diagnoses. Challenges are "
        "only interpreted across linked comments."
    )
    candidates = compute_challenge_review_candidates(results)
    for key, meta in CHALLENGE_DEFINITIONS.items():
        n = len(candidates.get(key, []))
        with st.expander(f"{meta['label']}: {n} linked pair(s) to review"):
            st.write(CHALLENGE_POLICY_GUIDANCE[key]["diagnostic_question"])
            st.write(CHALLENGE_POLICY_GUIDANCE[key]["action"])

    with st.expander("Interpretation notes"):
        st.write(MANAGER_MESSAGING_NOTE)
        st.write(CROSS_ORIENTATION_NOTE)
    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# TAB 3: ROADMAPS
# ─────────────────────────────────────────
def tab_roadmaps():
    st.subheader("Policymaking Roadmap (7 steps)")
    for num, title, text in POLICY_ROADMAP_STEPS:
        st.markdown(f"**Step {num}. {title}**")
        st.write(text)

    st.subheader("Managerial Roadmap (6 steps)")
    for num, title, text in MANAGER_ROADMAP_STEPS:
        st.markdown(f"**Step {num}. {title}**")
        st.write(text)

    st.caption(f'"{PAPER_TITLE}" | Read the paper: {PAPER_URL}')
    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# TAB 4: DEVELOPER TOOLS (password protected)
# ─────────────────────────────────────────
def tab_developer():
    dev_password = os.environ.get("DEV_PASSWORD", "")
    if not dev_password:
        st.info("Developer tools are disabled. The DEV_PASSWORD variable is not set.")
        return
    if not st.session_state.get("dev_ok"):
        entered = st.text_input("Developer password", type="password", key="dev_pw")
        if st.button("Unlock", key="dev_unlock"):
            if hmac.compare_digest(entered.encode("utf-8"), dev_password.encode("utf-8")):
                st.session_state["dev_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    st.subheader("Framework structure tests (no API calls)")
    if st.button("Run structure tests", key="dev_structure"):
        tests = run_framework_validation()
        passed = sum(1 for _, ok, _ in tests if ok)
        st.metric("Tests passed", f"{passed} / {len(tests)}")
        for name, ok, detail in tests:
            st.write(f"{'✅' if ok else '❌'} {name} — {detail}")

    st.subheader("Consistency check against reference examples (uses the API)")
    st.caption(
        "Internal check against illustrative examples. It is not empirical validation "
        "and does not measure intercoder reliability."
    )
    if st.button("Run consistency check", key="dev_consistency"):
        with st.spinner("Running the reference examples..."):
            report = run_consistency_suite(None)
        st.write(f"Overall exact-match agreement: {report['overall_agreement']:.0%}")
        st.dataframe(pd.DataFrame([
            {
                "Example": r["example"],
                "Expected": " / ".join(str(x) for x in r["expected"]),
                "Predicted": " / ".join(str(x) for x in r["predicted"]),
                "Match": r["match"],
                "Error": r.get("error", ""),
            }
            for r in report["results"]
        ]), use_container_width=True)


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
check_password()

st.title("Consumer Future-Making Analyzer")
with st.expander("About this tool"):
    st.markdown(HOMEPAGE_DESCRIPTION)
    st.caption(INTERPRETIVE_USE_NOTE)

tab1, tab2, tab3, tab4 = st.tabs([
    "Single comment",
    "Multiple comments",
    "Roadmaps",
    "Developer",
])
with tab1:
    tab_single()
with tab2:
    tab_multiple()
with tab3:
    tab_roadmaps()
with tab4:
    tab_developer()
