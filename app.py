"""
Consumer Future-Making Analyzer: interface (Streamlit).
The analysis logic lives in motor.py.
"""
import hashlib
import hmac
import html
import json
import logging
import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Consumer Future-Making Analyzer",
    page_icon="FM",
    layout="wide",
)

from motor import (
    CHALLENGE_DEFINITIONS, CHALLENGE_POLICY_GUIDANCE, CONTEXT_TYPE_CHOICES,
    CROSS_ORIENTATION_NOTE, DEFAULT_THREAD, DEMO_BY_ID, DEMO_CONVERSATION,
    DEMO_CONVERSATION_REFERENCE, DEMO_EXAMPLES, EST_COST_PER_COMMENT,
    EXAMPLE_DISCLAIMER, HOMEPAGE_DESCRIPTION, INTERPRETIVE_USE_NOTE,
    LABELED_THREAD, MANAGER_MESSAGING_NOTE, MANAGER_ORIENTATION_GUIDANCE,
    MAX_COMMENTS, ORIENTATION_ORDER, ORIENTATIONS, PAPER_TITLE, PAPER_URL, PF_EV,
    POLICY_ORIENTATION_GUIDANCE, analyze_comment, analyze_document,
    build_comment_records_from_consultation, build_comment_records_from_csv,
    build_comment_records_from_paragraphs, build_comment_records_from_user_labels,
    compute_challenge_review_candidates, compute_dominant_distributions,
    ensure_unique_ids, has_comment_text_column, orientation_coexistence,
    orientations_by_thread, prepare_records, pseudonymize, select_within_limit,
    _clean_enum,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("future_making_app")

CTX_CODE = {label: code for label, code, _ in CONTEXT_TYPE_CHOICES}
CTX_LABEL = {code: label for label, code, _ in CONTEXT_TYPE_CHOICES}
CTX_DESC = {code: desc for _, code, desc in CONTEXT_TYPE_CHOICES}

SOURCE_CSV = "CSV file (column comment_text, text, or comment)"
SOURCE_PASTE = "Pasted text (comments separated by a blank line)"
SOURCE_CONV = "Conversation with labels 'User 1:', 'User 2:'"
SOURCE_DEMO = "Paper example: forum conversation (4 users)"

ORI_LABEL = {o: o.title() for o in ORIENTATION_ORDER}


# ─────────────────────────────────────────
# PASSWORD
# ─────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("Consumer Future-Making Analyzer")
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        st.error("The APP_PASSWORD variable is missing on the server. Ask the administrator to set it.")
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
# SHARED COMPONENTS
# ─────────────────────────────────────────
def _guidance_expander(title, g, with_avoid=False):
    with st.expander(title):
        st.write(g["lens"])
        st.write(g["implications"])
        st.markdown("**What to monitor:**")
        for m in g["monitor"]:
            st.markdown(f"- {m}")
        st.markdown(f"**Objective:** {g['objective']}")
        if with_avoid and g.get("avoid"):
            st.markdown(f"**Avoid:** {g['avoid']}")


def show_result(r: dict, reference: dict = None):
    ori = _clean_enum(r.get("main_orientation", "")).upper()
    act = _clean_enum(r.get("main_activity", "")).upper()
    sub = _clean_enum(r.get("activity_subtype", "")).upper()
    meta = ORIENTATIONS.get(ori, {})
    color = html.escape(meta.get("color", "#333333"))
    bg = html.escape(meta.get("bg", "#FFFFFF"))

    st.markdown(
        f"<div style='background:{bg};border-left:6px solid {color};"
        f"padding:12px 16px;border-radius:6px'>"
        f"<b>Orientation:</b> {html.escape(ori)} &nbsp;|&nbsp; "
        f"<b>Dominant activity:</b> {html.escape(act)} / {html.escape(sub)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if reference:
        match = (act, sub, ori) == (reference["activity"], reference["subtype"], reference["orientation"])
        verdict = "Matches" if match else "Does not match"
        st.info(
            f"{verdict} the paper's reference: {reference['orientation'].title()} · "
            f"{reference['activity'].title()} / {reference['subtype'].title()}. {EXAMPLE_DISCLAIMER}"
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
        _guidance_expander("Policy guidance for this orientation (Steps 2 and 4)",
                           POLICY_ORIENTATION_GUIDANCE[ori])
        _guidance_expander("Managerial guidance for this orientation (Steps 2 and 4)",
                           MANAGER_ORIENTATION_GUIDANCE[ori], with_avoid=True)

    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# TAB 1: ONE COMMENT
# ─────────────────────────────────────────
def _demo_label(demo_id: str) -> str:
    if not demo_id:
        return "— Write my own comment —"
    return DEMO_BY_ID[demo_id]["titulo"]


def _load_single_demo():
    demo_id = st.session_state.get("demo_single", "")
    st.session_state.pop("single_result", None)
    ex = DEMO_BY_ID.get(demo_id)
    if ex:
        st.session_state["pf_single"] = PF_EV
        st.session_state["ctx_single"] = CTX_LABEL[ex["context_type"]]
        st.session_state["context_single"] = ex["context"]
        st.session_state["comment_single"] = ex["comment"]


def tab_single():
    st.subheader("Analyze a single comment")
    st.caption("Write your own comment or load one of the paper's examples to see how the tool works.")

    st.selectbox(
        "Paper example (optional)",
        [""] + list(DEMO_BY_ID),
        format_func=_demo_label,
        key="demo_single",
        on_change=_load_single_demo,
    )
    pf = st.text_area("Prescribed future", value=PF_EV, height=100, key="pf_single")
    label = st.selectbox("Context type", list(CTX_CODE), key="ctx_single")
    ctype = CTX_CODE[label]
    st.caption(CTX_DESC[ctype])
    context = st.text_area("Context (optional)", height=120, key="context_single")
    comment = st.text_area("Comment to classify", height=160, key="comment_single")
    with_secondary = st.checkbox(
        "Allow secondary classifications (if the comment contains a separable second practice)",
        key="secondary_single",
    )

    demo = DEMO_BY_ID.get(st.session_state.get("demo_single", ""))
    reference = demo["referencia"] if demo and comment.strip() == demo["comment"] else None
    if demo and reference:
        with st.expander("Paper's reference classification (no model call)", expanded=True):
            st.markdown(
                f"**Orientation:** {reference['orientation'].title()} · "
                f"**Activity:** {reference['activity'].title()} / {reference['subtype'].title()}"
            )
            st.write(demo["nota_paper"])
            st.caption(EXAMPLE_DISCLAIMER)

    if st.button("Classify", type="primary", key="btn_single"):
        if not pf.strip():
            st.warning("Enter the prescribed future before classifying.")
            return
        if not comment.strip():
            st.warning("Enter or load a comment before classifying.")
            return
        effective_ctype = ctype if context.strip() else "NONE"
        with st.spinner("Analyzing the comment..."):
            try:
                r = analyze_comment(
                    pf.strip(), comment.strip(), context.strip(), effective_ctype,
                    ctype == "CONSULTATION_PROMPT", allow_secondary=with_secondary,
                )
            except (ValueError, RuntimeError) as e:
                st.error(str(e))
                return
            except Exception:
                logger.exception("Error while classifying a single comment")
                st.error("The analysis could not be completed. Please try again.")
                return
        st.session_state["single_result"] = {
            "r": r,
            "key": (pf.strip(), comment.strip(), context.strip(), ctype, with_secondary),
        }

    stored = st.session_state.get("single_result")
    if not stored:
        return
    current_key = (pf.strip(), comment.strip(), context.strip(), ctype, with_secondary)
    if stored["key"] != current_key:
        st.info("The input changed since the last classification. Classify again to update the result.")
        return
    show_result(stored["r"], reference)
    st.download_button(
        "Download result (JSON)",
        data=json.dumps(stored["r"], ensure_ascii=False, indent=2),
        file_name="result.json",
        mime="application/json",
    )


# ─────────────────────────────────────────
# TAB 2: SEVERAL COMMENTS AND DASHBOARD
# ─────────────────────────────────────────
def _read_records(source, is_consult, uploaded, text):
    """Returns (records, messages[(level, text)])."""
    msgs = []
    if source == SOURCE_DEMO:
        return build_comment_records_from_user_labels(DEMO_CONVERSATION), msgs

    if source == SOURCE_CSV:
        if uploaded is None:
            return [], msgs
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            logger.exception("The CSV could not be read")
            msgs.append(("error", "The CSV could not be read. Check that it is a valid CSV file."))
            return [], msgs
        if not has_comment_text_column(df):
            msgs.append(("error", "The CSV needs a column named comment_text, text, or comment."))
            return [], msgs
        return build_comment_records_from_csv(df), msgs

    if not text.strip():
        return [], msgs

    if source == SOURCE_CONV:
        records = build_comment_records_from_user_labels(text)
        if not records:
            msgs.append(("warning", "Found fewer than two 'User N:' labels. Check the format."))
        return records, msgs

    if is_consult:
        msgs.append(("info", "For consultations, a CSV with comment_id is more reliable than pasted text."))
        return build_comment_records_from_consultation(text), msgs

    records, discarded = build_comment_records_from_paragraphs(text)
    if discarded:
        msgs.append(("info", f"Discarded {discarded} block(s) with fewer than two words."))
    return records, msgs


def _signature(pf, consult_q, is_consult, records):
    payload = json.dumps({
        "pf": pf,
        "cq": consult_q if is_consult else "",
        "recs": [(r["comment_id"], r["comment_text"], r["thread_id"], r.get("parent_comment_id"))
                 for r in records],
    }, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _author_label(r: dict, pseudo: bool) -> str:
    author = r.get("_author", "")
    return pseudonymize(author) if pseudo else author


def _pair_label(r: dict) -> str:
    return (f"{_clean_enum(r.get('main_orientation', '')).title()} · "
            f"{_clean_enum(r.get('activity_subtype', '')).title()}")


def _render_pie(coex: dict):
    values = [{
        "orientation": ORI_LABEL[o],
        "comments": coex["counts"][o],
        "color": ORIENTATIONS[o]["color"],
    } for o in coex["present"]]
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Distribution of future-making orientations across the analyzed comments.",
        "data": {"values": values},
        "mark": {"type": "arc", "tooltip": True},
        "encoding": {
            "theta": {"field": "comments", "type": "quantitative"},
            "color": {
                "field": "orientation", "type": "nominal",
                "scale": {"domain": [v["orientation"] for v in values],
                          "range": [v["color"] for v in values]},
                "legend": {"title": "Orientation"},
            },
            "tooltip": [
                {"field": "orientation", "type": "nominal", "title": "Orientation"},
                {"field": "comments", "type": "quantitative", "title": "Comments"},
            ],
        },
        "width": 320,
        "height": 320,
    }
    st.vega_lite_chart(spec)


def _coexistence_sentence(coex: dict) -> str:
    if coex["n_present"] == 0:
        return "No orientations were classified."
    if coex["n_present"] == 1:
        return (f"Only one orientation ({ORI_LABEL[coex['present'][0]]}) appears in this set, "
                "so no coexistence is observed.")
    names = ", ".join(ORI_LABEL[o] for o in coex["present"])
    return (f"{coex['n_present']} of 4 orientations coexist in this set: {names}. "
            "Coexistence describes the pattern in the corpus; it does not classify people.")


def _render_demo_comparison(valid: list):
    rows = []
    for r in valid:
        ref = DEMO_CONVERSATION_REFERENCE.get(r["_comment_id"])
        if not ref:
            continue
        pred = (_clean_enum(r.get("main_activity", "")).upper(),
                _clean_enum(r.get("activity_subtype", "")).upper(),
                _clean_enum(r.get("main_orientation", "")).upper())
        exp = (ref["activity"], ref["subtype"], ref["orientation"])
        rows.append({
            "Participant": r["_comment_id"],
            "Paper reference": f"{exp[2].title()} · {exp[0].title()} / {exp[1].title()}",
            "Model": f"{pred[2].title()} · {pred[0].title()} / {pred[1].title()}",
            "Match": "Yes" if pred == exp else "No",
        })
    if rows:
        st.markdown("**Comparison with the paper's reference**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(EXAMPLE_DISCLAIMER)


def _render_pairs(pairs, pseudo: bool):
    for pair in pairs:
        p, r = pair["parent"], pair["reply"]
        st.markdown(
            f"**{_author_label(p, pseudo) or 'Previous comment'} ({_pair_label(p)}) → "
            f"{_author_label(r, pseudo) or 'Reply'} ({_pair_label(r)})**"
        )
        st.markdown("> " + p["_comment_text"].replace("\n", " "))
        st.markdown("> " + r["_comment_text"].replace("\n", " "))
        st.divider()


def _show_dashboard(stored: dict, pseudo: bool):
    results = stored["results"]
    valid = [r for r in results if "_error" not in r]
    errors = len(results) - len(valid)

    st.success(f"Analysis complete: {len(valid)} of {len(results)} comments classified.")
    if errors:
        st.warning(f"{errors} comment(s) with errors. They are not counted in the distribution. "
                   "See the Error column in the table.")
    if stored.get("omitted"):
        st.warning(f"{stored['omitted']} comments were omitted because of the {MAX_COMMENTS}-comment limit.")
    if not valid:
        return

    st.markdown("### Coexistence of orientations")
    coex = orientation_coexistence(valid)
    col_chart, col_text = st.columns([1, 1])
    with col_chart:
        _render_pie(coex)
    with col_text:
        st.write(_coexistence_sentence(coex))
        table = pd.DataFrame([{
            "Orientation": ORI_LABEL[o],
            "Comments": coex["counts"][o],
            "Share": f"{coex['shares'][o]:.0%}",
        } for o in ORIENTATION_ORDER])
        st.dataframe(table, hide_index=True)
        st.caption("The chart shows the same information as the table, so it can be read without color.")

    by_thread = orientations_by_thread(valid)
    if by_thread:
        st.markdown("**Coexistence within each conversation**")
        rows = []
        for tid, info in by_thread.items():
            present = [o for o in ORIENTATION_ORDER if info["counts"][o]]
            rows.append({
                "Conversation": "Labeled conversation" if tid == LABELED_THREAD else str(tid),
                "Comments": info["n"],
                "Orientations present": f"{len(present)} of 4: " + ", ".join(ORI_LABEL[o] for o in present),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("### Activity distribution")
    dist = compute_dominant_distributions(valid)
    if dist["activity_counts"]:
        st.dataframe(pd.DataFrame(list(dist["activity_counts"].items()),
                                  columns=["Activity", "Comments"]), hide_index=True)
    if dist["subtype_counts"]:
        st.dataframe(pd.DataFrame(list(dist["subtype_counts"].items()),
                                  columns=["Activity / subtype", "Comments"]), hide_index=True)

    if stored.get("demo"):
        _render_demo_comparison(valid)

    st.markdown("### Human review of future-making challenges")
    st.caption("These are leads for manual review, not automatic diagnoses. Only linked pairs are interpreted.")
    candidates = compute_challenge_review_candidates(results)
    for key, meta in CHALLENGE_DEFINITIONS.items():
        pairs = candidates.get(key, [])
        with st.expander(f"{meta['label']}: {len(pairs)} linked pair(s) to review"):
            st.write(CHALLENGE_POLICY_GUIDANCE[key]["diagnostic_question"])
            st.write(CHALLENGE_POLICY_GUIDANCE[key]["action"])
            if pairs:
                st.markdown("**Linked pairs (conversation order)**")
                _render_pairs(pairs, pseudo)

    with st.expander("Interpretation notes"):
        st.write(MANAGER_MESSAGING_NOTE)
        st.write(CROSS_ORIENTATION_NOTE)

    st.markdown("### Results table")
    rows = []
    for r in results:
        rows.append({
            "ID": r.get("_comment_id", ""),
            "Author": _author_label(r, pseudo),
            "Comment": r.get("_comment_text", ""),
            "Orientation": _clean_enum(r.get("main_orientation", "")),
            "Activity": _clean_enum(r.get("main_activity", "")),
            "Subtype": _clean_enum(r.get("activity_subtype", "")),
            "Supporting text": r.get("supporting_text", ""),
            "Error": r.get("_error", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.download_button(
        "Download results (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="orientation_results.csv",
        mime="text/csv",
    )
    st.caption("Comments may contain names or other personal data. Review the file before sharing it.")


def tab_multiple():
    st.subheader("Map orientations across several comments")
    st.caption("Upload a CSV, paste comments, or use the paper's example conversation.")

    pf = st.text_area("Prescribed future", value=PF_EV, height=100, key="pf_multi")
    source = st.radio(
        "Source of the comments",
        [SOURCE_CSV, SOURCE_PASTE, SOURCE_CONV, SOURCE_DEMO],
        key="source_multi",
    )
    uploaded, text = None, ""
    if source == SOURCE_CSV:
        uploaded = st.file_uploader("Upload the CSV file", type=["csv"], key="csv_upload")
    elif source != SOURCE_DEMO:
        text = st.text_area("Paste the comments here", height=250, key="text_multi")

    is_consult = st.checkbox("These are responses to a public consultation", key="is_consult")
    consult_q = ""
    if is_consult:
        consult_q = st.text_area(
            "Consultation question or proposal (context for every response)",
            height=100, key="consult_question",
        )
    pseudo = st.checkbox("Pseudonymize authors in the table and download", value=True, key="pseudo_multi")

    if not pf.strip():
        st.warning("Enter the prescribed future before analyzing.")
        return
    if is_consult and not consult_q.strip():
        st.warning("Enter the consultation question before analyzing.")
        return
    if source == SOURCE_DEMO:
        st.info(EXAMPLE_DISCLAIMER)

    records, msgs = _read_records(source, is_consult, uploaded, text)
    for level, msg in msgs:
        getattr(st, level)(msg)
    if not records:
        st.info("No comments to analyze yet.")
        return

    renamed = ensure_unique_ids(records)
    if renamed:
        st.warning(f"{renamed} repeated identifier(s) were renamed so no comment is lost.")

    selected, omitted = select_within_limit(records, MAX_COMMENTS)
    prepare_records(selected, consult_q.strip() if is_consult else "", is_consult)

    st.write(f"Comments detected: **{len(records)}**. Will be analyzed: **{len(selected)}**.")
    if omitted:
        st.warning(f"{omitted} comments will be omitted to control cost (limit {MAX_COMMENTS}). "
                   "Complete threads are kept together.")
    st.info(f"Estimated cost: ~USD {len(selected) * EST_COST_PER_COMMENT:.2f}.")

    with st.expander("Preview the comments"):
        st.dataframe(pd.DataFrame([{
            "ID": r["comment_id"],
            "Author": pseudonymize(r.get("author", "")) if pseudo else r.get("author", ""),
            "Comment": r["comment_text"],
        } for r in selected]), hide_index=True)

    signature = _signature(pf.strip(), consult_q.strip(), is_consult, selected)

    if st.button("Analyze comments", type="primary", key="btn_multi"):
        progress = st.progress(0, text="Starting analysis...")
        try:
            results = analyze_document(
                selected, pf.strip(),
                lambda done, total: progress.progress(done / total, text=f"Analyzed {done}/{total}"),
            )
        except Exception:
            logger.exception("Error during batch analysis")
            st.error("An error occurred during the analysis. Check the server log.")
            return
        progress.empty()
        st.session_state["multi"] = {
            "signature": signature,
            "results": results,
            "omitted": omitted,
            "demo": source == SOURCE_DEMO,
        }

    stored = st.session_state.get("multi")
    if not stored:
        return
    if stored["signature"] != signature:
        st.info("The input changed since the last analysis. Analyze again to update the dashboard.")
        return
    _show_dashboard(stored, pseudo)


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
check_password()

st.title("Consumer Future-Making Analyzer")
with st.expander("About this tool"):
    st.markdown(HOMEPAGE_DESCRIPTION)
    st.caption(f'"{PAPER_TITLE}" | Read the paper: {PAPER_URL}')
    st.caption(INTERPRETIVE_USE_NOTE)

tab1, tab2 = st.tabs(["Analyze a single comment", "Several comments and dashboard"])
with tab1:
    tab_single()
with tab2:
    tab_multiple()
