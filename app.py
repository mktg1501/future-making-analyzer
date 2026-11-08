"""
Consumer Future-Making Analyzer: interface (Streamlit).
The analysis logic lives in motor.py.
"""
import hashlib
import html
import json
import logging
import re
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Consumer Future-Making Analyzer",
    page_icon="FM",
    layout="wide",
)

from motor import (
    ASSUMED_THREAD, CHALLENGE_DEFINITIONS, CONTEXT_TYPE_CHOICES, DEMO_BY_ID,
    DEMO_CONVERSATION, DEMO_CONVERSATION_REFERENCE, HOMEPAGE_DESCRIPTION,
    INTERPRETIVE_USE_NOTE, LABELED_THREAD, LINK_ASSUMED, LINK_CONFIRMED, LINK_LABELED,
    LINK_NONE, MANAGER_ORIENTATION_GUIDANCE, MANAGER_ROADMAP_STEPS, MAX_COMMENTS,
    ORIENTATION_ORDER, ORIENTATIONS, PAPER_TITLE, PAPER_URL, PF_EV,
    POLICY_ORIENTATION_GUIDANCE, POLICY_ROADMAP_STEPS, analyze_comment, analyze_document,
    apply_assumed_adjacency, build_comment_records_from_consultation,
    build_comment_records_from_csv, build_comment_records_from_paragraphs,
    build_comment_records_from_user_labels, compute_challenge_review_candidates,
    compute_dominant_distributions, ensure_unique_ids, has_comment_text_column,
    orientation_coexistence, orientations_by_thread, prepare_records, pseudonymize,
    select_within_limit, _clean_enum,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("future_making_app")

CTX_CODE = {label: code for label, code, _ in CONTEXT_TYPE_CHOICES}
CTX_LABEL = {code: label for label, code, _ in CONTEXT_TYPE_CHOICES}
CTX_DESC = {code: desc for _, code, desc in CONTEXT_TYPE_CHOICES}

ORI_LABEL = {o: o.title() for o in ORIENTATION_ORDER}

# ─────────────────────────────────────────
# WORDING
# Orientations are always described as "consumers with a ... orientation".
# Applied to all guidance and model text before display.
# ─────────────────────────────────────────
_ORIENTATION_PATTERN = re.compile(
    r"\b(resistant|catalyzer|ambivalent|expander)(?:s|\s+consumers?)\b", re.IGNORECASE
)
_ARTICLE = {"ambivalent": "an", "expander": "an"}


def _neutral(text) -> str:
    def repl(m):
        word = m.group(1).lower()
        phrase = f"consumers with {_ARTICLE.get(word, 'a')} {word} orientation"
        if m.group(0)[0].isupper():
            phrase = phrase[0].upper() + phrase[1:]
        return phrase
    return _ORIENTATION_PATTERN.sub(repl, str(text))


EXAMPLE_NOTE = (
    "These examples show how the tool works. The same comments are included in the model's "
    "instructions, so they do not measure the tool's accuracy."
)

PRIVACY_NOTICE = (
    "**Privacy notice.** Do not upload or paste Personally Identifiable Information (PII), such as "
    "names, email addresses, phone numbers, home addresses, account handles, or identification numbers. "
    "Removing PII is a standard rule designed to protect user data and comply with privacy laws. "
    "Comments are sent to an external AI service (OpenAI) for classification."
)

PII_CONSENT_LABEL = "I confirm that this text contains no Personally Identifiable Information (PII)."

SRC_CONSUMER = "Consumer-generated text (interviews, social media, forums, public consultation submissions)"
SRC_INSTITUTIONAL = "Institutional or media text (news, archival, policy documents)"

ROADMAP_STATUS = {
    "generated": "Generated from the data; check against the full comments",
    "review": "Leads for human review; not a diagnosis",
    "decision": "Needs a decision by your team",
    "input": "Defined by you",
}

FRAGILITY_NOTE = "These are observable indicators, not a measure of fragility."

TABLE_WE_SOURCE = (
    "Source: manuscript Web Appendix Tables WE1 (policymaking) and WE2 (managerial), which apply the "
    "framework to AI-integrated healthcare. The actions are the authors'; the wording is adapted here "
    "from patients to consumers, and the healthcare examples are not shown."
)

PAIR_NOTE = (
    "Each pair is linked, shares an activity, and differs in orientation or subtype. A pair is a lead "
    "for review, not evidence that a challenge occurred. Links are shown with their basis: confirmed "
    "in the data, given by the order of User N labels, or assumed by the tool. Pairs with assumed links "
    "are hidden unless you include them."
)

DECISION_OPTIONS = ["Not reviewed", "Shows the challenge", "Does not show it", "Unsure"]

LINK_BASIS_LABEL = {
    LINK_CONFIRMED: "Confirmed in the data",
    LINK_LABELED: "Order given by the User N labels (not verified as replies)",
    LINK_ASSUMED: "Assumed by the tool",
    LINK_NONE: "No link",
}

THREAD_LABEL = {
    LABELED_THREAD: "Pasted conversation (User N labels)",
    ASSUMED_THREAD: "Pasted text with assumed links",
}

LINK_CHOICE_DATA = "Use only the links in the data"
LINK_CHOICE_ASSUMED = "Assume each comment replies to the one above it"

# ─────────────────────────────────────────
# GUIDANCE FROM TABLES WE1 AND WE2 (adapted to consumers)
# ─────────────────────────────────────────
WE_ORIENTATION_GUIDANCE = {
    "CATALYZER": {
        "policy_action": (
            "Monitor urgency and inevitability narratives, simplified evaluations, advocacy, and early use. "
            "Do not use early uptake to frame wider adoption as inevitable."
        ),
        "manager_action": (
            "Use early adopters' experiences and testimonials, and document the conditions and limitations "
            "that made their adoption possible."
        ),
        "manager_caution": (
            "Do not present their experiences as representative or as evidence that adoption is inevitable."
        ),
    },
    "AMBIVALENT": {
        "policy_action": (
            "Identify the particular condition behind the hesitation (for example charging, price, servicing, "
            "or battery life), and direct support toward that condition rather than general information about EVs."
        ),
        "manager_action": (
            "Monitor conditional language, repeated comparison, requests for evidence, trials without continuation, "
            "and abandoned purchases. Identify the specific unresolved condition."
        ),
        "manager_caution": "Avoid pressure, artificial urgency, and framing hesitation as ignorance.",
    },
    "RESISTANT": {
        "policy_action": (
            "Monitor narratives of control and loss of choice, refusals, complaints, and continued reliance on "
            "current vehicles. Identify what the resistance seeks to protect before choosing a response."
        ),
        "manager_action": (
            "Provide clear disclosure, meaningful opportunities to ask questions, human assistance, and workable "
            "complaint or escalation routes. Make alternatives visible where they exist, and monitor refusals to "
            "understand what they protect."
        ),
        "manager_caution": (
            "Avoid \"there is no alternative\", ridicule, and hidden automation of decisions."
        ),
    },
    "EXPANDER": {
        "policy_action": (
            "Examine the alternative problem definitions and practices consumers propose, such as public transport, "
            "cycling, or denser cities. Check whether the intervention is presented as a complete solution when it "
            "may be one component of a broader strategy."
        ),
        "manager_action": (
            "Monitor claims that the offer does not solve the underlying problem. Invite consumers to describe broader "
            "preferred futures, and assess whether the offer should be complemented by other pathways."
        ),
        "manager_caution": (
            "Avoid presenting the offer as a complete solution, and avoid dismissing systemic critique as outside "
            "the organization's responsibility."
        ),
    },
}

WE_MESSAGING_RULE = (
    "Communicate achievements alongside uncertainty, trade-offs, limitations, enabling conditions, and "
    "distributional effects. Avoid inevitability and universal-benefit claims. (Table WE2, step 5.)"
)

WE_POLICY_STEP_ACTIONS = {
    1: "Define the practices the intervention changes. Check whether the prescribed future depends on conditions (money, equipment, skills, access) that are not evenly distributed.",
    2: "Combine consultations, complaints, feedback, and accounts of planned or actual practice. Do not infer a consumer's orientation toward one measure from their response to another.",
    3: "Treat evaluation, negotiation, and enactment as interdependent. Identify whether the main difficulty lies in how the future is evaluated, negotiated, or enacted.",
    4: "Match support to the orientations and challenges present. The objective is responsible acceleration, not uniform acceleration.",
    5: "Locate infrastructure and capability support where practices change: purchase, home charging, servicing, and resale.",
    6: "Measure fairness, understandability, and access alongside adoption. Adoption alone does not show that the prescribed future is stabilizing.",
    7: "If a measure raises technical performance but intensifies conflict, reconsider how it is framed and supported, then repeat the diagnostic cycle.",
}

WE_MANAGER_STEP_ACTIONS = {
    1: "Map the changes each offer asks consumers to make across the customer journey, including whether they must use a product or only change how they access a service.",
    2: "Use interviews, service data, complaints, opt-outs, and actual use. Locate orientations at specific offers and touchpoints, not permanently at individuals.",
    3: "Identify which challenge a pattern indicates, rather than treating every divergence as generic resistance.",
    4: "Choose responses by orientation and challenge. Check whether responding to one orientation intensifies a challenge elsewhere.",
    5: "Communicate achievements together with uncertainty, trade-offs, and enabling conditions. Avoid inevitability and universal-benefit claims.",
    6: "Support consumers where they must adjust practice: purchase, charging, servicing, resale, and complaints. Make it easy to pause, reverse, or modify adoption.",
}

# ─────────────────────────────────────────
# EXAMPLES
# ─────────────────────────────────────────
SINGLE_EXAMPLE_IDS = ["joe", "clara"]

CONV2_TEXT = """User 1: I bought my EV because it was faster than the comparable ICE car, more spacious, has better range... at least $600 cheaper per quarter on "fuel," requires almost no maintenance or servicing... and gets regular improvements via software updates. It's the most brutally fast and best handling car I ever had!

User 2: In the building where I live there are 150 underground carparks. All full. There is no reserve capacity in the building to install charging points for 1% of that. Nor any in the feed from the street, nor any in the feed to the suburb... I have no objection to eventually driving an EV, but it's just not happening any time soon. The infrastructure is decades away.

User 3: I won't be getting one, I'll stick to my V8 and my other diesel 4x4...

User 4: Maybe some reliable public transport would be an answer."""

CONV2_REFERENCE = {
    "user1": {"activity": "ENACTMENT", "subtype": "ACCELERATE", "orientation": "CATALYZER"},
    "user2": {"activity": "ENACTMENT", "subtype": "DELAY", "orientation": "AMBIVALENT"},
    "user3": {"activity": "ENACTMENT", "subtype": "PREVENT", "orientation": "RESISTANT"},
    "user4": {"activity": "ENACTMENT", "subtype": "REROUTE", "orientation": "EXPANDER"},
}

CONVERSATION_EXAMPLES = {
    "conv1": {
        "label": "Example 1: forum exchange on the adoption curve (4 users)",
        "text": DEMO_CONVERSATION,
        "reference": DEMO_CONVERSATION_REFERENCE,
        "source_note": (
            "Forum exchange quoted in the manuscript. Reference labels are those the manuscript "
            "assigns to each user. The order follows the User N labels."
        ),
    },
    "conv2": {
        "label": "Example 2: social media thread on competing enactments (4 users)",
        "text": CONV2_TEXT,
        "reference": CONV2_REFERENCE,
        "source_note": (
            "Social media thread from the manuscript's competing enactments example. The manuscript labels "
            "this thread by orientation only; the activity labels are a proposed coding that researchers "
            "should check before use."
        ),
    },
}
CONVERSATION_BY_LABEL = {v["label"]: v for v in CONVERSATION_EXAMPLES.values()}

# Manuscript Table 2: orientation definitions.
TABLE2 = {
    "CATALYZER": {
        "narrative": "Urgency narrative",
        "goal": "Accelerate change toward the prescribed future",
        "temporality": "The future is close (change is happening now)",
        "condition": "High degree of alignment between current practices and the prescribed future",
    },
    "AMBIVALENT": {
        "narrative": "Pragmatic narrative",
        "goal": "Slow down change, delay decisions, balance risks and benefits",
        "temporality": "The future is contingent (change is uncertain)",
        "condition": "Limited resources to support change in current practices",
    },
    "RESISTANT": {
        "narrative": "Control narrative",
        "goal": "Contest the prescribed future, protect the status quo",
        "temporality": "The future is distant (no change)",
        "condition": "Low degree of alignment between current practices and the prescribed future",
    },
    "EXPANDER": {
        "narrative": "Bigger picture narrative",
        "goal": "Expand the prescribed future, propose new pathways and alternative futures",
        "temporality": "The future is distant (broader change)",
        "condition": "Mismatch among current practices, normative practices, and those directed by the prescribed future",
    },
}


# ─────────────────────────────────────────
# SHARED COMPONENTS
# ─────────────────────────────────────────
def show_result(r: dict, reference: dict = None):
    ori = _clean_enum(r.get("main_orientation", "")).upper()
    act = _clean_enum(r.get("main_activity", "")).upper()
    sub = _clean_enum(r.get("activity_subtype", "")).upper()
    color = html.escape(ORIENTATIONS.get(ori, {}).get("color", "#333333"))
    bg = html.escape(ORIENTATIONS.get(ori, {}).get("bg", "#FFFFFF"))

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
            f"{verdict} the reference classification: {reference['orientation'].title()} · "
            f"{reference['activity'].title()} / {reference['subtype'].title()}. {EXAMPLE_NOTE}"
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Activity rationale**")
        st.write(_neutral(r.get("activity_rationale", "")))
        st.markdown("**Orientation rationale**")
        st.write(_neutral(r.get("orientation_rationale", "")))
    with col2:
        st.markdown("**Narrative identified:** " + _neutral(r.get("narrative_identified", "")))
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
                    f"({s.get('orientation', '')}): {_neutral(s.get('rationale', ''))}"
                )

    if r.get("context_note"):
        st.markdown("**Context note:** " + str(r.get("context_note")))
    if r.get("input_scope_warning"):
        st.warning("Input-scope note: " + str(r.get("input_scope_warning")))

    if ori in WE_ORIENTATION_GUIDANCE:
        we = WE_ORIENTATION_GUIDANCE[ori]
        with st.expander("Policy guidance for this orientation"):
            st.write(_neutral(POLICY_ORIENTATION_GUIDANCE[ori]["implications"]))
            st.markdown(f"**Recommended action:** {_neutral(we['policy_action'])}")
            st.caption(TABLE_WE_SOURCE)
        with st.expander("Managerial guidance for this orientation"):
            st.write(_neutral(MANAGER_ORIENTATION_GUIDANCE[ori]["implications"]))
            st.markdown(f"**Intervention:** {_neutral(we['manager_action'])}")
            st.markdown(f"**Caution:** {_neutral(we['manager_caution'])}")
            st.caption(TABLE_WE_SOURCE)

    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# TAB 1: ONE COMMENT
# ─────────────────────────────────────────
def _single_example_label(value: str) -> str:
    if not value:
        return "Write my own comment"
    number = SINGLE_EXAMPLE_IDS.index(value) + 1
    return f"Example {number}: {DEMO_BY_ID[value]['titulo'].split(':')[0]}"


def _load_single_demo():
    demo_id = st.session_state.get("demo_single", "")
    st.session_state.pop("single_result", None)
    ex = DEMO_BY_ID.get(demo_id)
    if ex:
        st.session_state["pf_single"] = PF_EV
        st.session_state["ctx_single"] = CTX_LABEL[ex["context_type"]]
        st.session_state["context_single"] = ex["context"]
        st.session_state["comment_single"] = ex["comment"]


def _render_example_intro(ex: dict):
    number = SINGLE_EXAMPLE_IDS.index(ex["id"]) + 1
    source = ex["titulo"].split(":")[0]
    st.markdown(f"**Example {number}: {source}**")
    st.markdown(f"> {ex['comment']}")
    ctype = ex["context_type"]
    st.markdown(f"**Context type:** {CTX_LABEL[ctype]}")
    st.caption(CTX_DESC[ctype])
    st.markdown(f"**Context provided:** {ex['context'] or 'None. The comment is interpreted on its own.'}")
    st.caption(f"Why this reference: {ex['nota_paper']}")


@st.cache_data(show_spinner=False, max_entries=64)
def _classify_example(pf: str, comment: str, context: str, ctype: str, allow_secondary: bool) -> dict:
    """Examples are public text from the manuscript, so caching them is safe."""
    return analyze_comment(pf, comment, context, ctype, False, allow_secondary=allow_secondary)


def _show_example_analysis(demo: dict, pf: str, with_secondary: bool):
    st.markdown("### Analysis")
    with st.spinner("Analyzing the example..."):
        try:
            r = _classify_example(pf, demo["comment"], "", "NONE", with_secondary)
        except (ValueError, RuntimeError) as e:
            st.error(str(e))
            return
        except Exception:
            logger.exception("Error while analyzing an example comment")
            st.error("The analysis could not be completed. Please try again.")
            return
    show_result(r, demo["referencia"])


def _custom_single_flow(pf, ctype, context, comment, with_secondary):
    st.markdown("### Analysis")
    consent = st.checkbox(PII_CONSENT_LABEL, key="pii_single")
    if st.button("Classify", type="primary", key="btn_single", disabled=not consent):
        if not pf.strip():
            st.warning("Enter the prescribed future before classifying.")
            return
        if not comment.strip():
            st.warning("Enter a comment before classifying.")
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
    show_result(stored["r"])
    st.download_button(
        "Download result (JSON)",
        data=json.dumps(stored["r"], ensure_ascii=False, indent=2),
        file_name="result.json",
        mime="application/json",
    )


def tab_single():
    st.subheader("Analyze a single comment")
    st.caption("Write your own comment, or choose one of the two examples to see the analysis.")
    st.info(PRIVACY_NOTICE)

    st.session_state.setdefault("pf_single", PF_EV)
    st.session_state.setdefault("ctx_single", next(iter(CTX_CODE)))
    st.session_state.setdefault("context_single", "")
    st.session_state.setdefault("comment_single", "")

    st.selectbox(
        "Example (optional)",
        [""] + SINGLE_EXAMPLE_IDS,
        format_func=_single_example_label,
        key="demo_single",
        on_change=_load_single_demo,
    )

    demo = DEMO_BY_ID.get(st.session_state.get("demo_single", ""))
    if demo:
        _render_example_intro(demo)

    pf = st.text_area("Prescribed future", height=100, key="pf_single")
    label = st.selectbox("Context type", list(CTX_CODE), key="ctx_single")
    ctype = CTX_CODE[label]
    st.caption(CTX_DESC[ctype])
    context = st.text_area("Context (optional)", height=120, key="context_single")
    comment = st.text_area("Comment to classify", height=160, key="comment_single")
    with_secondary = st.checkbox(
        "Allow secondary classifications (if the comment contains a separable second practice)",
        key="secondary_single",
    )

    is_example = (
        demo is not None
        and comment.strip() == demo["comment"]
        and pf.strip() == PF_EV.strip()
        and not context.strip()
    )
    if is_example:
        _show_example_analysis(demo, pf.strip(), with_secondary)
        return
    _custom_single_flow(pf, ctype, context, comment, with_secondary)


# ─────────────────────────────────────────
# TAB 2: MAP ORIENTATIONS ACROSS SEVERAL COMMENTS
# ─────────────────────────────────────────
SOURCE_CSV = "CSV file (column comment_text, text, or comment)"
SOURCE_PASTE = "Pasted text (comments separated by a blank line)"
SOURCE_CONV = "Pasted conversation with labels 'User 1:', 'User 2:'"
SOURCE_OPTIONS = [SOURCE_CSV, SOURCE_PASTE, SOURCE_CONV] + [v["label"] for v in CONVERSATION_EXAMPLES.values()]


def _read_records(source, is_consult, uploaded, text):
    """Returns (records, messages[(level, text)])."""
    msgs = []
    if source in CONVERSATION_BY_LABEL:
        return build_comment_records_from_user_labels(CONVERSATION_BY_LABEL[source]["text"]), msgs

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


def _label3(r: dict) -> str:
    return (f"{_clean_enum(r.get('main_orientation', '')).title()} · "
            f"{_clean_enum(r.get('main_activity', '')).title()} / "
            f"{_clean_enum(r.get('activity_subtype', '')).title()}")


def _link_basis(source: str) -> str:
    return LINK_BASIS_LABEL.get(source or LINK_NONE, LINK_BASIS_LABEL[LINK_NONE])


def _pair_id(pair: dict) -> str:
    return f"{pair['parent']['_comment_id']}->{pair['reply']['_comment_id']}"


def _one_line(text: str) -> str:
    return str(text).replace("\n", " ")


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
    st.vega_lite_chart(spec=spec)


def _coexistence_sentence(coex: dict) -> str:
    if coex["n_present"] == 0:
        return "No orientations were classified."
    if coex["n_present"] == 1:
        return (f"Only one orientation ({ORI_LABEL[coex['present'][0]]}) appears in this set, "
                "so no coexistence is observed.")
    names = ", ".join(ORI_LABEL[o] for o in coex["present"])
    return (f"{coex['n_present']} of 4 orientations coexist in this set: {names}. "
            "This describes the pattern in these comments; it does not classify people.")


def _adapted(valid: list, ori: str) -> dict:
    """Summarizes what the model found for one orientation in THIS dataset."""
    rows = [r for r in valid if _clean_enum(r.get("main_orientation", "")).upper() == ori]
    narratives = Counter(_neutral(str(r.get("narrative_identified", "")).strip())
                         for r in rows if str(r.get("narrative_identified", "")).strip())
    conditions = [str(r.get("notable_conditions_of_adoption", "")).strip() for r in rows
                  if str(r.get("notable_conditions_of_adoption", "")).strip()]
    quotes = [str(r.get("supporting_text", "")).strip() for r in rows
              if str(r.get("supporting_text", "")).strip()]
    return {"n": len(rows), "narratives": narratives.most_common(3),
            "conditions": conditions[:3], "quotes": quotes[:2]}


def _adapted_text(a: dict) -> str:
    if a["n"] == 0:
        return ""
    parts = [f"{a['n']} comment(s) in this set."]
    if a["narratives"]:
        parts.append("Narratives the model identified: " +
                     "; ".join(f"{n} ({c})" for n, c in a["narratives"]) + ".")
    if a["conditions"]:
        parts.append("Conditions the model identified: " + " | ".join(a["conditions"]) + ".")
    return " ".join(parts)


def _render_adapted_block(valid: list, ori: str):
    a = _adapted(valid, ori)
    st.markdown("**In this dataset:** " + _adapted_text(a))
    for q in a["quotes"]:
        st.markdown("> " + q)
    st.caption("Labels come from the model. Check them against the complete comments before relying on them.")


def _render_reference_comparison(valid: list, reference: dict):
    rows = []
    for r in valid:
        ref = reference.get(r["_comment_id"])
        if not ref:
            continue
        pred = (_clean_enum(r.get("main_activity", "")).upper(),
                _clean_enum(r.get("activity_subtype", "")).upper(),
                _clean_enum(r.get("main_orientation", "")).upper())
        exp = (ref["activity"], ref["subtype"], ref["orientation"])
        rows.append({
            "Participant": r["_comment_id"],
            "Reference": f"{exp[2].title()} · {exp[0].title()} / {exp[1].title()}",
            "Model": f"{pred[2].title()} · {pred[0].title()} / {pred[1].title()}",
            "Match": "Yes" if pred == exp else "No",
        })
    if rows:
        st.markdown("**Comparison with the reference classification**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(EXAMPLE_NOTE)


def _adjacency_rows(valid: list, pseudo: bool) -> list:
    by_id = {r["_comment_id"]: r for r in valid if r.get("_comment_id")}
    rows = []
    for r in valid:
        parent = by_id.get(r.get("_parent_comment_id"))
        if not parent:
            continue
        rows.append({
            "Replying participant": _author_label(r, pseudo) or r["_comment_id"],
            "Responds to": _author_label(parent, pseudo) or parent["_comment_id"],
            "Previous comment": _label3(parent),
            "Reply": _label3(r),
            "Link basis": _link_basis(r.get("_link_source")),
        })
    return rows


def _repeat_participants(valid: list, pseudo: bool) -> list:
    groups = {}
    for r in valid:
        author = r.get("_author", "")
        if author:
            groups.setdefault(author, []).append(r)
    rows = []
    for author, rs in groups.items():
        if len(rs) < 2:
            continue
        oris = {_clean_enum(r.get("main_orientation", "")).upper() for r in rs}
        ordered = [ORI_LABEL[o] for o in ORIENTATION_ORDER if o in oris]
        rows.append({
            "Participant": pseudonymize(author) if pseudo else author,
            "Comments": len(rs),
            "Orientations used across comments": ", ".join(ordered),
        })
    return rows


ROADMAP_STATUS_NOTE = (
    "Each step of the roadmap, with what this dataset shows for it. The manuscript's roadmaps (Figures 3 and 4) "
    "are sequential in the text but recursive in practice, so you can revisit a step as new data arrive."
)


def _roadmap_integration(valid: list, results: list, pf_text: str, include_assumed: bool):
    st.markdown("#### Roadmap integration")
    st.write(ROADMAP_STATUS_NOTE)
    choice = st.radio(
        "Roadmap", ["Policymaking (7 steps)", "Managerial (6 steps)"],
        horizontal=True, key="roadmap_choice",
    )
    is_policy = choice.startswith("Policy")
    steps = POLICY_ROADMAP_STEPS if is_policy else MANAGER_ROADMAP_STEPS
    we_actions = WE_POLICY_STEP_ACTIONS if is_policy else WE_MANAGER_STEP_ACTIONS

    coex = orientation_coexistence(valid)
    candidates = compute_challenge_review_candidates(results, include_assumed=include_assumed)
    n_conv = len(candidates["CONVOLUTED_EVALUATIONS"])
    n_conf = len(candidates["CONFRONTATIONAL_NEGOTIATIONS"])
    n_comp = len(candidates["COMPETING_ENACTMENTS"])
    n_enact = sum(1 for r in valid if _clean_enum(r.get("main_activity", "")).upper() == "ENACTMENT")
    top = max(ORIENTATION_ORDER, key=lambda o: coex["counts"][o])
    shares = ", ".join(f"{ORI_LABEL[o]} {coex['counts'][o]}" for o in ORIENTATION_ORDER if coex["counts"][o])

    if is_policy:
        content = {
            1: (f"Prescribed future: {pf_text[:200]}", "input"),
            2: (f"Orientations in this set: {shares} ({coex['n_present']} of 4 present).", "generated"),
            3: (f"Linked pairs to review: convoluted evaluations {n_conv}, "
                f"confrontational negotiations {n_conf}, competing enactments {n_comp}.", "review"),
            4: (f"Largest group: {ORI_LABEL[top]} ({coex['counts'][top]} comments). "
                "Choose support measures for each orientation present (see the policy implications).", "decision"),
            5: (f"Comments reporting enactment: {n_enact} of {len(valid)}. Decide which infrastructure "
                "and capabilities consumers need.", "decision"),
            6: ("A single run is one snapshot. Measuring progress needs a dated log of repeated runs.", "decision"),
            7: ("Revision depends on steps 4 to 6 and on the measurements in step 6.", "decision"),
        }
    else:
        content = {
            1: (f"Prescribed future: {pf_text[:200]}", "input"),
            2: (f"Orientations in this set: {shares} ({coex['n_present']} of 4 present).", "generated"),
            3: (f"Linked pairs to review: convoluted evaluations {n_conv}, "
                f"confrontational negotiations {n_conf}, competing enactments {n_comp}.", "review"),
            4: (f"Largest group: {ORI_LABEL[top]} ({coex['counts'][top]} comments). "
                "Choose support strategies sensitive to the orientations present.", "decision"),
            5: ("Check whether current messages mobilize one orientation while alienating others.", "decision"),
            6: (f"Comments reporting enactment: {n_enact} of {len(valid)}. Decide where consumers need support "
                "at each touchpoint.", "decision"),
        }

    rows = []
    for num, title, _ in steps:
        n = int(num)
        text, status = content[n]
        rows.append({
            "Step": f"{num}. {title}",
            "What this dataset shows": text,
            "Table WE action (adapted)": we_actions.get(n, ""),
            "Status": ROADMAP_STATUS[status],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "Only steps 1 to 3 can be partly generated from the data. The other steps depend on decisions "
        "that the manuscript assigns to policymakers or managers. " + TABLE_WE_SOURCE
    )


def _fragility_indicators(valid: list, results: list, include_assumed: bool):
    st.markdown("#### Fragile futures: observable indicators")
    st.write(FRAGILITY_NOTE)

    coex = orientation_coexistence(valid)
    candidates = compute_challenge_review_candidates(results, include_assumed=include_assumed)
    enact = [r for r in valid if _clean_enum(r.get("main_activity", "")).upper() == "ENACTMENT"]
    enact_subs = {_clean_enum(r.get("activity_subtype", "")).upper() for r in enact} - {""}
    accelerate = sum(1 for r in enact if _clean_enum(r.get("activity_subtype", "")).upper() == "ACCELERATE")

    rows = [
        {"Indicator": "Orientations present (of 4)",
         "Observed in this set": f"{coex['n_present']} of 4",
         "Paper's concept": "Multiple preferred futures"},
        {"Indicator": "Linked pairs flagged: confrontational negotiation",
         "Observed in this set": f"{len(candidates['CONFRONTATIONAL_NEGOTIATIONS'])} pair(s) for review",
         "Paper's concept": "Confrontational negotiations"},
        {"Indicator": "Linked pairs flagged: convoluted evaluation",
         "Observed in this set": f"{len(candidates['CONVOLUTED_EVALUATIONS'])} pair(s) for review",
         "Paper's concept": "Convoluted evaluations"},
        {"Indicator": "Enactment subtypes present (of 4)",
         "Observed in this set": f"{len(enact_subs)} of 4" if enact else "No enactment comments",
         "Paper's concept": "Competing enactments"},
        {"Indicator": "Enactment comments that accelerate the prescribed future",
         "Observed in this set": f"{accelerate} of {len(enact)}" if enact else "No enactment comments",
         "Paper's concept": "Acceleration of the prescribed future"},
        {"Indicator": "Volatility over time",
         "Observed in this set": "Not observable from one run",
         "Paper's concept": "Volatility"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Keep these indicators separate. Do not add them into a single index.")


def _render_analysis(valid: list, results: list, pseudo: bool, reference, pf_text: str, include_assumed: bool):
    coex = orientation_coexistence(valid)
    present = coex["present"]
    dist = compute_dominant_distributions(valid)

    st.markdown("### Analysis and implications")
    st.caption("Based on the manuscript and its Web Appendix. Each comment is counted once, by its dominant activity and orientation.")

    st.markdown("#### 1. Analysis of the results")
    st.write(_coexistence_sentence(coex))
    st.write(
        "The manuscript describes the four orientations as socially patterned ways of performing future-making. "
        "Each has its own narrative, goal, temporality, and notable condition of adoption (Table 2)."
    )
    st.dataframe(pd.DataFrame([{
        "Orientation": ORI_LABEL[o],
        "Narrative": TABLE2[o]["narrative"],
        "Goal": TABLE2[o]["goal"],
        "Temporality": TABLE2[o]["temporality"],
        "Notable condition of adoption": TABLE2[o]["condition"],
    } for o in present]), hide_index=True)
    st.caption("Source: manuscript, Table 2.")

    for o in present:
        with st.expander(f"{ORI_LABEL[o]}: {coex['counts'][o]} comment(s), {coex['shares'][o]:.0%} of the set"):
            _render_adapted_block(valid, o)

    if dist["activity_counts"]:
        st.write(
            "Evaluation, negotiation, and enactment are interdependent in the manuscript. This tool records only the "
            "dominant activity of each comment, so these counts describe each comment's main function, not everything it contains."
        )
        st.dataframe(pd.DataFrame(list(dist["activity_counts"].items()),
                                  columns=["Activity", "Comments"]), hide_index=True)

    _roadmap_integration(valid, results, pf_text, include_assumed)

    _fragility_indicators(valid, results, include_assumed)

    adjacency = _adjacency_rows(valid, pseudo)
    if adjacency:
        st.markdown("**Replies compared with the comment they answer**")
        st.write(
            "This table is descriptive. It shows which orientation and activity each reply has, next to the "
            "comment it responds to, and the basis of each link. It does not show that one performance caused the other."
        )
        st.dataframe(pd.DataFrame(adjacency), hide_index=True)

    repeats = _repeat_participants(valid, pseudo)
    if repeats:
        st.markdown("**Participants with more than one comment**")
        st.write(
            "Where the same participant appears more than once, this shows which orientations they used. The data "
            "show orientations in use across comments; they do not show how or why a participant changed."
        )
        st.dataframe(pd.DataFrame(repeats), hide_index=True)

    if reference:
        _render_reference_comparison(valid, reference)

    st.markdown("#### 2. Managerial implications")
    st.caption(TABLE_WE_SOURCE)
    for o in present:
        mg = MANAGER_ORIENTATION_GUIDANCE[o]
        we = WE_ORIENTATION_GUIDANCE[o]
        with st.expander(f"{ORI_LABEL[o]}: managerial implications"):
            st.write(_neutral(mg["implications"]))
            _render_adapted_block(valid, o)
            st.markdown(f"**Intervention:** {_neutral(we['manager_action'])}")
            st.markdown(f"**Caution:** {_neutral(we['manager_caution'])}")
            st.markdown(f"**Objective:** {_neutral(mg['objective'])}")
    st.markdown("**Messaging (managerial roadmap, step 5)**")
    st.write(WE_MESSAGING_RULE)

    st.markdown("#### 3. Public policy implications")
    st.caption(TABLE_WE_SOURCE)
    for o in present:
        pg = POLICY_ORIENTATION_GUIDANCE[o]
        we = WE_ORIENTATION_GUIDANCE[o]
        with st.expander(f"{ORI_LABEL[o]}: policy implications"):
            st.write(_neutral(pg["implications"]))
            _render_adapted_block(valid, o)
            st.markdown(f"**Recommended action:** {_neutral(we['policy_action'])}")
            st.markdown("**What to monitor:**")
            for m in pg["monitor"]:
                st.markdown(f"- {_neutral(m)}")
            st.markdown(f"**Objective:** {_neutral(pg['objective'])}")


def _linked_pairs_section(results: list, pseudo: bool, include_assumed: bool):
    st.markdown("### Linked pairs for human review")
    st.caption(PAIR_NOTE)
    candidates = compute_challenge_review_candidates(results, include_assumed=include_assumed)
    if not any(candidates.values()):
        st.info("No linked pairs to review under the current link setting.")
        return

    for key, meta in CHALLENGE_DEFINITIONS.items():
        pairs = candidates.get(key, [])
        if not pairs:
            continue
        st.markdown(f"**{meta['label']}: {len(pairs)} pair(s)**")
        st.caption(meta["definition"])
        for pair in pairs:
            p, rp = pair["parent"], pair["reply"]
            st.markdown(
                f"**{_author_label(p, pseudo) or p['_comment_id']} ({_label3(p)}) → "
                f"{_author_label(rp, pseudo) or rp['_comment_id']} ({_label3(rp)})**"
            )
            st.markdown("> " + _one_line(p["_comment_text"]))
            st.markdown("> " + _one_line(rp["_comment_text"]))
            st.caption("Link basis: " + _link_basis(pair.get("link_source")))
            st.selectbox("Researcher decision", DECISION_OPTIONS, key=f"decision_{_pair_id(pair)}")
            st.divider()

    decision_rows = []
    for key, meta in CHALLENGE_DEFINITIONS.items():
        for pair in candidates.get(key, []):
            pid = _pair_id(pair)
            decision_rows.append({
                "Pair": pid,
                "Challenge (for review)": meta["label"],
                "Link basis": _link_basis(pair.get("link_source")),
                "Researcher decision": st.session_state.get(f"decision_{pid}", "Not reviewed"),
            })
    if decision_rows:
        df = pd.DataFrame(decision_rows)
        st.download_button(
            "Download review decisions (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="review_decisions.csv",
            mime="text/csv",
        )


def _show_dashboard(stored: dict, pseudo: bool, include_assumed: bool):
    results = stored["results"]
    valid = [r for r in results if "_error" not in r]
    errors = len(results) - len(valid)
    pf_text = stored.get("pf", "")

    st.success(f"Analysis complete: {len(valid)} of {len(results)} comments classified.")
    st.markdown(f"**Prescribed future:** {pf_text}")
    if errors:
        st.warning(f"{errors} comment(s) with errors. They are not counted in the results. See the Error column in the table.")
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
        st.dataframe(pd.DataFrame([{
            "Orientation": ORI_LABEL[o],
            "Comments": coex["counts"][o],
            "Share": f"{coex['shares'][o]:.0%}",
        } for o in ORIENTATION_ORDER]), hide_index=True)
        st.caption("The chart shows the same information as the table, so it can be read without color.")

    by_thread = orientations_by_thread(valid)
    if by_thread:
        st.markdown("**Coexistence within each conversation**")
        rows = []
        for tid, info in by_thread.items():
            present = [o for o in ORIENTATION_ORDER if info["counts"][o]]
            rows.append({
                "Conversation": THREAD_LABEL.get(tid, str(tid)),
                "Comments": info["n"],
                "Orientations present": f"{len(present)} of 4: " + ", ".join(ORI_LABEL[o] for o in present),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    _render_analysis(valid, results, pseudo, stored.get("reference"), pf_text, include_assumed)

    _linked_pairs_section(results, pseudo, include_assumed)

    with st.expander("How this follows the manuscript's method"):
        st.markdown(
            "- **Sources.** Consumer texts (interviews, social media, forums, public consultations) are classified. "
            "Institutional and media texts are used to define the prescribed future, not classified (Table 1).\n"
            "- **Dominant activity.** Each comment is classified by its dominant activity: evaluation, negotiation, or "
            "enactment. A comment is negotiation when its overall function is to advocate, question, reject, or contest a "
            "collective trajectory.\n"
            "- **Orientations.** Orientations are assigned from the full configuration of narrative, goal, emotion, "
            "temporality, and practice, not from sentiment or keywords alone (Section C of the prompt).\n"
            "- **Links.** Each link states its basis. Links assumed by the tool are excluded from challenge review "
            "unless the user includes them.\n"
            "- **Challenges.** The three future-making challenges arise across linked comments and are not assigned to a single comment. "
            "Fragile futures are a theoretical condition and are not scored by this tool.\n"
            "- **Limits.** The tool is a demonstration. Its results need interpretation with the full comments and context."
        )

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
            "Link basis": _link_basis(r.get("_link_source")),
            "Activity rationale": r.get("activity_rationale", ""),
            "Orientation rationale": r.get("orientation_rationale", ""),
            "Narrative": r.get("narrative_identified", ""),
            "Temporality": r.get("temporality_identified", ""),
            "Condition of adoption": r.get("notable_conditions_of_adoption", ""),
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
    st.caption("Upload a CSV, paste comments, or choose one of the two example conversations.")
    st.info(PRIVACY_NOTICE)

    st.session_state.setdefault("pf_multi", PF_EV)
    pf = st.text_area("Prescribed future", height=100, key="pf_multi")

    source_type = st.radio("Type of text", [SRC_CONSUMER, SRC_INSTITUTIONAL], key="source_type_multi")
    if source_type == SRC_INSTITUTIONAL:
        st.info(
            "Institutional and media texts are not classified as consumer orientations. The manuscript uses them "
            "to define the prescribed future and its framing. Paste them into the 'Prescribed future' box, or analyze "
            "consumer comments that respond to them."
        )
        return

    source = st.radio("Source of the comments", SOURCE_OPTIONS, key="source_multi")
    example = CONVERSATION_BY_LABEL.get(source)

    uploaded, text = None, ""
    if example:
        st.markdown(f"**Prescribed future for this example:** {PF_EV}")
        st.markdown(
            "**Context type:** Parent comment. Each user is read with the previous user's comment as context; "
            "User 1 has no context."
        )
        with st.expander("Show the conversation", expanded=True):
            for block in example["text"].split("\n\n"):
                st.markdown("> " + block)
        st.caption(example["source_note"])
    elif source == SOURCE_CSV:
        uploaded = st.file_uploader("Upload the CSV file", type=["csv"], key="csv_upload")
    else:
        text = st.text_area("Paste the comments here", height=250, key="text_multi")

    is_consult, consult_q, consent = False, "", True
    if not example:
        is_consult = st.checkbox("These are responses to a public consultation", key="is_consult")
        if is_consult:
            consult_q = st.text_area(
                "Consultation question or proposal (context for every response)",
                height=100, key="consult_question",
            )
        consent = st.checkbox(PII_CONSENT_LABEL, key="pii_multi")
    pseudo = st.checkbox("Pseudonymize authors in the table and download", value=True, key="pseudo_multi")

    link_mode = "data"
    if not example:
        link_choice = st.radio(
            "How should comments be linked?",
            [LINK_CHOICE_DATA, LINK_CHOICE_ASSUMED],
            key="link_mode_multi",
            help="Use the data's own reply links when they exist. The assumed option treats each "
                 "comment as a reply to the one above it, and labels every such link as assumed.",
        )
        link_mode = "assumed" if link_choice == LINK_CHOICE_ASSUMED else "data"

    include_assumed = False

    if not pf.strip():
        st.warning("Enter the prescribed future before analyzing.")
        return
    if is_consult and not consult_q.strip():
        st.warning("Enter the consultation question before analyzing.")
        return

    records, msgs = _read_records(source, is_consult, uploaded, text)
    for level, msg in msgs:
        getattr(st, level)(msg)
    if not records:
        st.info("No comments to analyze yet.")
        return

    renamed = ensure_unique_ids(records)
    if renamed:
        st.warning(f"{renamed} repeated identifier(s) were renamed so no comment is lost.")

    if link_mode == "assumed":
        records = apply_assumed_adjacency(records)
        st.warning(
            "Assumed links are not evidence of interaction. Each link created here is labeled as assumed "
            "in the results."
        )
        include_assumed = st.checkbox(
            "Include pairs with assumed links in the review section",
            value=False, key="include_assumed",
        )

    selected, omitted = select_within_limit(records, MAX_COMMENTS)
    prepare_records(selected, consult_q.strip() if is_consult else "", is_consult)

    st.write(f"Comments detected: **{len(records)}**. Will be analyzed: **{len(selected)}**.")
    if omitted:
        st.warning(f"{omitted} comments will be omitted to control cost (limit {MAX_COMMENTS}). "
                   "Complete threads are kept together.")

    with st.expander("Preview the comments"):
        st.dataframe(pd.DataFrame([{
            "ID": r["comment_id"],
            "Author": pseudonymize(r.get("author", "")) if pseudo else r.get("author", ""),
            "Link basis": _link_basis(r.get("link_source")),
            "Comment": r["comment_text"],
        } for r in selected]), hide_index=True)

    signature = _signature(pf.strip(), consult_q.strip(), is_consult, selected)

    if st.button("Analyze comments", type="primary", key="btn_multi", disabled=not consent):
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
            "pf": pf.strip(),
            "reference": example["reference"] if example else None,
        }

    stored = st.session_state.get("multi")
    if not stored:
        return
    if stored["signature"] != signature:
        st.info("The input changed since the last analysis. Analyze again to update the results.")
        return
    _show_dashboard(stored, pseudo, include_assumed)


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
st.title("Consumer Future-Making Analyzer")
with st.expander("About this tool"):
    st.markdown(HOMEPAGE_DESCRIPTION)
    st.info(PRIVACY_NOTICE)
    st.caption(f'"{PAPER_TITLE}" | Read the paper: {PAPER_URL}')
    st.caption(INTERPRETIVE_USE_NOTE)

tab1, tab2 = st.tabs(["Analyze a single comment", "Map orientations across several comments"])
with tab1:
    tab_single()
with tab2:
    tab_multiple()
