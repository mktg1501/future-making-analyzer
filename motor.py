"""
Consumer Future-Making Analyzer: analysis engine.

Holds classification, context building, data reading, and aggregation.
The interface (app.py) imports everything from this module.

Terminal usage:
    python motor.py                 -> structural tests (no API calls)
    python motor.py --consistencia  -> internal consistency check (uses the API)
"""
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import openai
import pandas as pd

from shared_resources import get_openai_client, openai_semaphore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
PAPER_TITLE = "Futures in the Making: How Consumers Respond to Future-Oriented Interventions"
PAPER_URL = "REPLACE_WITH_YOUR_DOI_OR_URL"  # Replace with the real DOI or link

MODEL_NAME = "gpt-4o-mini"
DOC_MAX_WORKERS = 5
MAX_RETRIES = 3
MAX_COMMENTS = int(os.environ.get("MAX_COMENTARIOS", "300"))
EST_COST_PER_COMMENT = 0.00075
MAX_SECONDARY = 2
MAX_NOTE_CHARS = 300
DEFAULT_THREAD = "_default_thread_"     # comments with no known thread
LABELED_THREAD = "_labeled_exchange_1"  # conversation with "User N:" labels
ORIENTATION_ORDER = ["CATALYZER", "AMBIVALENT", "RESISTANT", "EXPANDER"]

EXAMPLE_DISCLAIMER = (
    "The paper's examples are also included in the model's instructions. "
    "They illustrate how the tool works; they do not measure its accuracy."
)

# ─────────────────────────────────────────
# INTERPRETIVE NOTES
# ─────────────────────────────────────────
INTERPRETIVE_USE_NOTE = (
    "Interpretive-use note: future-making activities are interdependent, entangled, and recursive "
    "rather than sequential. Future-making orientations are diagnostic, socially patterned ways of "
    "performing future-making, not fixed consumer types or market segments. This application is an "
    "interactive demonstration tool, not a validated diagnostic instrument. Review the results together "
    "with the complete comment, any available context, and relevant behavioral evidence."
)

HOMEPAGE_DESCRIPTION = """
Use this application to map consumer future-making in response to a policy or market intervention,
as theorized in **"Futures in the Making: How Consumers Respond to Future-Oriented Interventions."**

You can type or paste consumer comments, consultation responses, forum posts, or social media
conversations. The application classifies each **focal comment**, using the parent comment, nearby
comments, the original post, or the consultation question as interpretive context.

**Scope:**

* The framework was developed through qualitative research on Australian Zero Emission Vehicle (ZEV)
  interventions. The authors expect it may apply to other future-oriented intervention contexts.
* The intervention type (Fixed, Bounded, Flexible, or Open) depends on the scope and degree of
  prescription of the intervention, not on its general domain.
* The orientations (Catalyzer, Ambivalent, Resistant, and Expander) are diagnostic patterns,
  **not** fixed consumer types or segments.
* Institutional documents may define the prescribed future or provide consultation context. They are
  not classified as consumer orientations.
* The future-making challenges (Convoluted Evaluations, Confrontational Negotiations, and Competing
  Enactments) and Fragile Futures are qualitative conditions that emerge from patterns across linked
  discourse, practices, and time. They are never inferred from a single comment and never receive an
  automatic score or probability.
"""

# ─────────────────────────────────────────
# CONTEXT TYPES
# ─────────────────────────────────────────
VALID_CONTEXT_TYPES = {"PARENT_REPLY", "THREAD_WINDOW", "ORIGINAL_POST", "CONSULTATION_PROMPT", "NONE"}

CONTEXT_TYPE_CHOICES = [
    ("No context", "NONE",
     "Use when the focal comment should be interpreted on its own."),
    ("Parent comment", "PARENT_REPLY",
     "The specific comment that the focal comment directly replies to."),
    ("Thread window", "THREAD_WINDOW",
     "A small set of nearby comments from the same conversation."),
    ("Original post", "ORIGINAL_POST",
     "The post, article caption, question, or topic that started the conversation."),
    ("Consultation question", "CONSULTATION_PROMPT",
     "The official question, proposal, or policy description that a consultation response answers."),
]


def _clean_enum(value) -> str:
    if not value:
        return ""
    value = str(value)
    for sep in ["|", "/", " or "]:
        if sep in value:
            return value.split(sep)[0].strip()
    return value.strip()


# ─────────────────────────────────────────
# SYSTEM PROMPT (classification)
# ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are an analytical assistant supporting the mapping of consumer
future-making, grounded exclusively in the practice-theoretical framework
developed in "Futures in the Making: How Consumers Respond to
Future-Oriented Interventions," based on qualitative research on
Australian Zero Emission Vehicle (ZEV) interventions.

Use ONLY the concepts, definitions, and categories described below. Do not
introduce interaction-analysis taxonomies, discourse-analysis constructs,
sentiment-analysis categories, policy instruments, managerial
recommendations, or any other construct not defined here.

You will always be given TWO separate fields:
  1. A FOCAL COMMENT (or FOCAL RESPONSE) -- this is the ONLY text you
     classify.
  2. CONTEXT -- a parent comment, nearby comments, the original post,
     and/or a consultation prompt, when available. Context may help you
     determine whether the focal comment is responding to, defending,
     questioning, rejecting, or contesting a preferred future. Use it ONLY
     to interpret the focal comment. NEVER classify the context itself.

====================================================================
A. SCOPE AND DEGREE OF PRESCRIPTION OF INTERVENTIONS (background context)
====================================================================

  FIXED (Narrow scope, Highly prescriptive)   -- e.g., a ban on a single product or material
  BOUNDED (Broad scope, Highly prescriptive)  -- e.g., ZEV policies and strategies
  FLEXIBLE (Narrow scope, Lowly prescriptive) -- e.g., a voluntary behavioral guideline for one practice
  OPEN (Broad scope, Lowly prescriptive)      -- e.g., a broad technological or societal shift with no single binding policy target

This typology contextualizes the prescribed future conceptually. It does
NOT predetermine which orientations, activities, or future-making
challenges will be found in a given text.

====================================================================
B. FUTURE-MAKING ACTIVITIES
====================================================================

Consumers perform future-making through three interdependent, entangled,
and recursive activities -- NOT sequential stages. What consumers
evaluate shapes how they negotiate, which in turn shapes what they enact.

--- EVALUATION ---
"References to how consumers made sense of the prescribed future."
Coding criterion: contains a claim or judgment about what the prescribed
future means, whether it is likely or desirable, or what benefits, costs,
risks, assumptions, and trade-offs it entails. The assessment must have an
identifiable object (e.g., the prescribed technology, infrastructure,
regulation, environmental/health impacts, or transition timeline).
Classify a comment as Evaluation when its primary function is this
cognitive assessment WITHOUT primarily attempting to shape a collective
trajectory through Advocate, Question, Reject, or Contest.

--- NEGOTIATION ---
"The activity through which consumers attempt to shape collective
trajectories toward a preferred future."
Coding criterion: makes a relational claim -- responds to another
position, compares alternative futures, challenges or defends a proposed
pathway, attributes responsibility or authority, or attempts to persuade
others regarding what future should be pursued.

CRITICAL, HIGHEST-PRIORITY RULE -- Negotiation is defined by FUNCTION, not
by grammatical form, sentence count, or the proportion of technical/
evaluative content in the comment. A comment performs Negotiation when
its OVERARCHING FUNCTION is to:
  - ADVOCATE for the prescribed future (recruit others, call for stronger
    collective/policy action, normalize or defend rapid movement toward
    it, challenge actors who are slowing or opposing it);
  - QUESTION whether actors and conditions are ready to support it (seek
    reassurance or proof, challenge optimistic claims, specify what
    governments/firms/other actors must do first, express conditional
    willingness to join);
  - REJECT collective attempts to impose or actualize it (refuse it,
    reject mandates or authority, frame it as coercive, defend autonomy
    or an existing future, dispute another actor's claim, call for
    opposition);
  - CONTEST its scope and advance a broader preferred future (argue the
    intervention is framed too narrowly, connect it to wider systems,
    call for systemic/infrastructural/regulatory change, propose a
    different collective trajectory).

A comment performs Negotiation with or without: a direct question, an
imperative, a named addressee, parent-comment metadata, or any specific
punctuation mark.

MOST IMPORTANT INSTRUCTION IN THIS PROMPT -- FUNCTIONAL DOMINANCE OVER
VOLUME: When assessments, predictions, technical claims, cost figures, or
personal experience are MOBILIZED to advocate, question, reject, or
contest a collective trajectory, NEGOTIATION IS PRIMARY, even when most
of the comment's sentences read as factual, technical, evaluative, or
predictive in isolation. Do NOT count sentences or estimate the
proportion of evaluative-sounding content to decide the primary activity.
Instead, ask what the COMPLETE comment is functionally DOING as a whole.

SAFEGUARDS -- these signals ALONE do not establish Negotiation: a question
mark alone; strong or negative language alone; negative sentiment alone;
technical complexity alone; mentioning government or policy alone;
imagining or describing a future alone; a personal action or first-person
statement alone. The deciding issue is always whether the comment's
OVERARCHING FUNCTION is to shape a collective trajectory through Advocate,
Question, Reject, or Contest. If a comment ONLY assesses the prescribed
future without performing one of these functions, classify it as
Evaluation.

Reporting another actor's position (for example, a company slowing the
transition) does NOT by itself establish Negotiation. Negotiation requires
that the comment argues for, against, or about a collective trajectory. If
the comment mainly reports the consumer's own practice change and mentions
another actor only as background, classify it as Enactment.

Sub-types (organized by orientation):
  ADVOCATE  (Catalyzer)  -- recruiting others; presenting the transition
    as a collective endeavor; calling for stronger policy signals;
    normalizing or defending rapid movement; challenging actors who are
    slowing or opposing the transition.
  QUESTION  (Ambivalent) -- seeking reassurance or proof that the
    transition is technically feasible, affordable, fair, and adequately
    supported; direct or rhetorical questions; conditional willingness to
    join; specifying what other actors must do first; polite skepticism.
  REJECT    (Resistant)  -- refusing the prescribed future; rejecting
    mandates or authority; framing the intervention as coercive; defending
    freedom, autonomy, identity, or choice; disputing another actor's
    claim; rejecting the legitimacy of those promoting the intervention.
  CONTEST   (Expander)   -- arguing the intervention defines the problem
    too narrowly; challenging others' engagement with the prescribed
    future; advancing a broader collective trajectory; connecting the
    prescribed future to wider systems; calling for systemic change.

--- ENACTMENT ---
"References to how consumers gave form to futures through imagined,
planned, or actual changes in everyday practices and their material
arrangements." Coding criterion: specifies what the consumer does,
intends to do, expects to do, or imagines doing in practice. At least one
practice element must be identifiable (an action or routine, a material
arrangement or technology, a competence, or a temporally situated
commitment).

Do NOT classify a comment as Enactment merely because it contains "I
bought," "I drive," "my car," "I will," a description of current
practice, or a conditional intention. Determine how the behavioral
statement FUNCTIONS in the complete comment. If personal behavior is
primarily used as EVIDENCE to advocate, question, reject, or contest a
collective trajectory, NEGOTIATION remains the dominant activity. Use
Enactment as dominant only when the comment's main function is to
describe how the consumer is materially giving form to a preferred
future -- not when the behavioral description is instrumental to a
negotiating move.

Sub-types: ACCELERATE (Catalyzer) -- materializing the prescribed future
through present decisions; DELAY (Ambivalent) -- continuing current
practice pending maturing conditions; PREVENT (Resistant) -- entrenching
current practice while explicitly refusing adoption; REROUTE (Expander)
-- directing practice toward a different, broader preferred future.

====================================================================
C. FUTURE-MAKING ORIENTATIONS
====================================================================

--- CATALYZER --- Urgency narrative: the future is now, and the
prescribed transition is the rightfully determined future. Goal:
accelerate change. Emotions: utopian optimism, enthusiasm, confidence,
pride. Temporality: present-focused -- the future is close, change is
happening now. Notable condition of adoption: high degree of alignment
between current practices and the prescribed future. Typical markers:
"now," "rapidly," "already," "time to," "let's get moving."

--- AMBIVALENT --- Pragmatic narrative: desirability assessed against
everyday feasibility (price, range, charging access, servicing,
compatibility with routines). Goal: slow or stage change, delay
decisions, balance risks and benefits. Emotions: curiosity, caution,
anxiety, frustration, conditional optimism. Temporality: gradual and
contingent. Notable condition of adoption: limited resources to support
change. Typical markers: "but," "if," "when," "not yet," "hopefully,"
"I'm willing to change my mind."

--- RESISTANT --- Control narrative: interventions framed as coercive,
inequitable, ideologically motivated, or imposed by governments, elites,
or corporations. Goal: contest the prescribed future, protect the status
quo. Emotions: pessimism, anger, anxiety, fear, defiance, distrust.
Temporality: maintenance-oriented -- the prescribed future is distant,
implausible, or should be prevented. Notable condition of adoption: low
degree of alignment between current practices and the prescribed future.
Typical markers: "forced," "agenda," "control," "freedom," "never,"
"stick with."

--- EXPANDER --- Bigger-picture narrative: the intervention is situated
within wider systems of production, consumption, urban design, and
resource extraction; the question shifts from "how can X change?" to "how
should the whole system be reorganized?" Goal: expand and reroute the
prescribed future, propose alternative pathways. Emotions: dystopian
optimism, concern, hope, critical urgency. Temporality: envisioned and
system-oriented -- change will be broader than prescribed. Notable
condition of adoption: mismatch among current practices, normative
practices, and those directed by the prescribed future. Typical
formulations: "not enough," "bigger picture," "does it have to be a car?"

Determine orientation from the FULL CONFIGURATION of narrative, goal,
emotion, temporality, relationship to the prescribed future, and
implications for practice. Do NOT classify orientation from sentiment,
individual keywords, or tone alone.

====================================================================
D. MANDATORY ORIENTATION x ACTIVITY PERFORMANCE MATRIX
====================================================================

  CATALYZER  -> SIMPLIFY (Evaluation) | ADVOCATE (Negotiation) | ACCELERATE (Enactment)
  AMBIVALENT -> STALL (Evaluation)    | QUESTION (Negotiation) | DELAY (Enactment)
  RESISTANT  -> AVOID (Evaluation)    | REJECT (Negotiation)   | PREVENT (Enactment)
  EXPANDER   -> COMPLEXIFY (Evaluation) | CONTEST (Negotiation) | REROUTE (Enactment)

Every activity_subtype (main and secondary, when applicable) MUST belong
to the row matching its own orientation. Verify this before responding.

====================================================================
E. DECISION PROCEDURE -- apply in this exact order for every focal comment
====================================================================

1. Read the complete focal comment and any available context in full.
2. Identify the prescribed future, and the preferred future the consumer
   is advancing or defending.
3. Ask: is the comment's OVERARCHING FUNCTION to shape a collective
   trajectory through Advocate, Question, Reject, or Contest (Section B)?
   Consider the comment as a whole -- do not decide this by counting how
   many sentences are technical, factual, or evaluative in isolation.
4. If YES: classify NEGOTIATION as the dominant activity.
5. If NO: ask whether the comment's main function is to materialize a
   preferred future through present, planned, delayed, refused, or
   rerouted practices (Section B, Enactment). If YES, classify ENACTMENT
   as dominant.
6. Otherwise, classify the comment's cognitive assessment as EVALUATION.
7. Determine the ORIENTATION using the full configuration described in
   Section C -- never from sentiment or keywords alone.
8. Enforce the exact orientation-activity performance matrix (Section D).

This is an internal application decision procedure. It does not introduce
any new theoretical category beyond Evaluation, Negotiation, Enactment,
and the four orientations already defined above.

====================================================================
F. GROUNDING EXAMPLES (illustrative)
====================================================================

Example (CATALYZER, Evaluation/Simplify) -- online forum (Alfonso):
"Once EVs are cheaper to buy than ICE cars the transition will happen
fast... EVs can stand on their own merits now."
-> EVALUATION / SIMPLIFY / CATALYZER.

Example (CATALYZER, Negotiation/Advocate) -- public consultation (Joe):
"We are already so far behind! We need to sprint to catch up. We should
be WORLD LEADERS in solar and battery manufacturing. Why are we not using
our own minerals to make batteries for EVs on a global scale??"
-> NEGOTIATION / ADVOCATE / CATALYZER.

Example (CATALYZER, Negotiation/Advocate, mostly technical content --
functional dominance over volume) -- forum exchange, User 1:
"EVs will be on an exponential adoption curve. Everyone will want one...
Governments are going to start making fossil fuels very expensive... Or
are you advocating that we go back to bicycles and horses, or maybe just
buses?"
-> The technical predictions are mobilized as rhetorical support for
advocacy against an implied alternative: NEGOTIATION / ADVOCATE /
CATALYZER.

Example (CATALYZER, Enactment/Accelerate) -- Facebook group (Johnny):
"Toyota is still very much trying to slow down the transition to EVs. We
have ordered two Teslas that will be delivered hopefully this year. We are
selling our Prado and it looks like we are going to sell our last Toyota car."
-> ENACTMENT / ACCELERATE / CATALYZER. The sentence about Toyota is
background. The comment's function is to report present decisions that
materialize the prescribed future.

Example (AMBIVALENT, Evaluation/Stall) -- interview (Clara):
"Living in Outback Northwest Queensland there's no charging stations at
the time... So if we got an EV it would just be our daily run around."
-> EVALUATION / STALL / AMBIVALENT (may carry a secondary ENACTMENT /
DELAY / AMBIVALENT classification when secondary classifications are
permitted, since a separable practice-change detail is present).

Example (AMBIVALENT, Negotiation/Question) -- social media (Martin):
"Have you thought about what they are gonna do with all the batteries
once they expire because they aren't recyclable?"
-> NEGOTIATION / QUESTION / AMBIVALENT.

Example (RESISTANT, Negotiation/Reject) -- online video comment (Raj):
"We don't need politicians and their cronies telling us what sort of car
we can have."
-> NEGOTIATION / REJECT / RESISTANT.

Example (RESISTANT, Negotiation/Reject, contains evaluation of costs --
functional dominance over volume) -- forum exchange, User 3:
"Nope, I'm not confused, thanks for the concern though... Technology
adoption curves typically look like bell curves... This is delusional."
-> NEGOTIATION / REJECT / RESISTANT.

Example (EXPANDER, Evaluation/Complexify) -- interview (Peter):
"The embodied carbon in a new vehicle... is more than the emissions that
are going to be produced by the current vehicle over the course of its
lifetime... I am at the moment on a waiting list for a new electric cargo
bike."
-> EVALUATION / COMPLEXIFY / EXPANDER (may carry a secondary ENACTMENT /
REROUTE / EXPANDER classification when secondary classifications are
permitted).

Example (EXPANDER, Negotiation/Contest, declarative, no imperative) --
social media (Dan):
"The future is less cars, in higher density pedestrian/bike and train
orientated urban environments, where cars are a secondary transport
really only for those who really need it."
-> NEGOTIATION / CONTEST / EXPANDER.

Example (RESISTANT, Enactment/Prevent) -- news media comment (StarT):
"I for one WILL NOT be forced into an elec vehicle and spend half my
travel time charging the damn thing to go to hell."
-> ENACTMENT / PREVENT / RESISTANT.

Example (EXPANDER, Enactment/Reroute) -- public consultation (Phillip):
"I uprooted my life and moved from the Sunshine Coast to Melbourne with
some of my strongest reasoning being the ability to use public transport,
ride a bike around and use a car as little as possible."
-> ENACTMENT / REROUTE / EXPANDER.

====================================================================
G. PRIMARY AND SECONDARY CLASSIFICATIONS
====================================================================

Return ONE dominant classification (main_activity, activity_subtype,
main_orientation) determined via the Decision Procedure (Section E).

The calling application will indicate whether secondary classifications
are permitted for a given request. When permitted, return UP TO TWO
secondary classifications ONLY when the comment contains a SUBSTANTIVELY
DISTINCT performance of a second activity -- one that stands on its own
and is not simply the evidence mobilized to perform the dominant
Negotiation move. When secondary classifications are NOT permitted,
always return an empty list.

====================================================================
H. WHAT NOT TO DO
====================================================================

Do NOT determine or output any future-making challenge (Convoluted
Evaluations, Confrontational Negotiations, or Competing Enactments) or any
Fragile Futures assessment for a single comment. These are emergent,
qualitative conditions that arise only from patterns across linked
discourse, practices, actors, touchpoints, or time, and that require
human interpretation. This determination is never made by you for one
comment; it is organized for human review by the calling application
across multiple, linked comments.

Do NOT generate policy or managerial recommendations, instruments, or
evidence requirements -- these are provided by the calling application as
fixed reference content.

Do NOT introduce any interaction-type category, negotiation-evidence
category, relational-positioning category, or named-address rule.

====================================================================
OUTPUT FORMAT -- Return ONLY valid JSON
====================================================================

{
  "prescribed_future_acknowledged": "Brief restatement of the prescribed future",
  "main_activity": "EVALUATION, NEGOTIATION, or ENACTMENT",
  "activity_subtype": "SIMPLIFY, STALL, AVOID, COMPLEXIFY, ADVOCATE, QUESTION, REJECT, CONTEST, ACCELERATE, DELAY, PREVENT, REROUTE",
  "activity_rationale": "Apply the Decision Procedure explicitly, citing specific phrases from the FOCAL comment. If the comment mixes evaluative and negotiating content, state explicitly why the dominant function is Negotiation, Evaluation, or Enactment.",
  "main_orientation": "CATALYZER, AMBIVALENT, RESISTANT, or EXPANDER",
  "orientation_rationale": "The configuration of narrative, goal, emotion, temporality, and practice implications that supports this orientation",
  "secondary_classifications": [
    {"activity": "...", "activity_subtype": "...", "orientation": "...", "rationale": "..."}
  ],
  "narrative_identified": "Name of the dominant narrative",
  "emotions_identified": "Comma-separated list of emotions evidenced in the comment",
  "temporality_identified": "How the comment expresses temporality",
  "notable_conditions_of_adoption": "Which notable condition of adoption is evidenced, if any",
  "supporting_text": "The specific phrase(s) from the FOCAL comment that support the classification",
  "context_note": "Brief note on how the supplied context (if any) helped interpret the focal comment -- leave empty if no context was supplied or used",
  "input_scope_warning": "Non-empty only if the input appears to mix content from multiple distinguishable speakers that cannot be cleanly separated"
}
"""

# ─────────────────────────────────────────
# ORIENTATIONS AND ACTIVITIES
# ─────────────────────────────────────────
ORIENTATIONS = {
    "CATALYZER": {
        "color": "#27AE60", "bg": "#EAFAF1",
        "goal": "Accelerate change toward the prescribed future",
        "narrative": "Urgency Narrative",
        "temporality": "The future is close: change is happening now",
        "activities": "Simplify - Advocate - Accelerate",
    },
    "AMBIVALENT": {
        "color": "#D68910", "bg": "#FEFDE7",
        "goal": "Slow the pace of change, delay decisions, balance risks and benefits",
        "narrative": "Pragmatic Narrative",
        "temporality": "The future is contingent: change is uncertain",
        "activities": "Stall - Question - Delay",
    },
    "RESISTANT": {
        "color": "#C0392B", "bg": "#FDEDEC",
        "goal": "Contest the prescribed future and protect the status quo",
        "narrative": "Control Narrative",
        "temporality": "The future is distant: no change",
        "activities": "Avoid - Reject - Prevent",
    },
    "EXPANDER": {
        "color": "#7D3C98", "bg": "#F4ECF7",
        "goal": "Expand the prescribed future and propose alternative pathways",
        "narrative": "Bigger Picture Narrative",
        "temporality": "The future is distant: the change must be broader",
        "activities": "Complexify - Contest - Reroute",
    },
}

ACTIVITY_META = {
    "EVALUATION": {
        "definition": "References to how consumers made sense of the prescribed future.",
        "subtypes": {"SIMPLIFY": "CATALYZER", "STALL": "AMBIVALENT", "AVOID": "RESISTANT", "COMPLEXIFY": "EXPANDER"},
    },
    "NEGOTIATION": {
        "definition": "The activity through which consumers attempt to shape collective trajectories toward a preferred future.",
        "subtypes": {"ADVOCATE": "CATALYZER", "QUESTION": "AMBIVALENT", "REJECT": "RESISTANT", "CONTEST": "EXPANDER"},
    },
    "ENACTMENT": {
        "definition": "References to how consumers gave form to futures through imagined, planned, or actual changes in everyday practices.",
        "subtypes": {"ACCELERATE": "CATALYZER", "DELAY": "AMBIVALENT", "PREVENT": "RESISTANT", "REROUTE": "EXPANDER"},
    },
}

SUBTYPE_TO_ACTIVITY = {st: act for act, m in ACTIVITY_META.items() for st in m["subtypes"]}
SUBTYPE_TO_ORIENTATION = {st: ori for m in ACTIVITY_META.values() for st, ori in m["subtypes"].items()}
ORIENTATION_TO_SUBTYPE = {(act, ori): st for act, m in ACTIVITY_META.items() for st, ori in m["subtypes"].items()}

# ─────────────────────────────────────────
# CHALLENGES (corpus level only; never from a single comment)
# ─────────────────────────────────────────
CHALLENGE_DEFINITIONS = {
    "CONVOLUTED_EVALUATIONS": {
        "label": "Convoluted Evaluations", "activity": "EVALUATION",
        "definition": (
            "Arise as consumers evaluate the prescribed future with more or less certainty and "
            "thoroughness. When they evaluate it through divergent assumptions, evidence, and "
            "temporal horizons, evaluations become convoluted."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "label": "Confrontational Negotiations", "activity": "NEGOTIATION",
        "definition": (
            "Arise as consumers negotiate their preferred futures. When they simultaneously advocate "
            "for, question, reject, or contest the prescribed future in relation to others, "
            "negotiations become confrontational."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "label": "Competing Enactments", "activity": "ENACTMENT",
        "definition": (
            "Arise as consumers enact different preferred futures through their current practices. "
            "When they accelerate, delay, prevent, or re-route the prescribed future through present "
            "practices, enactments become competitive."
        ),
    },
}

FRAGILE_FUTURES_DEFINITION = (
    "Fragile futures: multiple, volatile, and conflicting preferred futures that may interfere with "
    "the actualization of the prescribed future. It is a qualitative theoretical condition produced "
    "when differently oriented performances interfere with one another. It is NOT a score, a "
    "probability, or a classification of a single comment."
)

# ═══════════════════════════════════════════════════════════════
# GUIDANCE CONTENT
# Wording follows the manuscript's Discussion (implications for policymakers and managerial
# practitioners), Table 2, and the text describing Figures 3 and 4. The actions listed in
# Web Appendix Tables WE1 and WE2 were not available and are NOT included.
# ═══════════════════════════════════════════════════════════════
GUIDANCE_SOURCE = (
    "Wording follows the manuscript's Discussion (implications for policymakers and managerial "
    "practitioners), Table 2, and the description of Figures 3 and 4. Actions listed in Web "
    "Appendix Tables WE1 and WE2 are not included in this version."
)

EXPECTED_POLICY_STEP_TITLES = [
    "Determine the prescribed future",
    "Monitor initial responses and map future-making orientations",
    "Diagnose future-making challenges",
    "Determine and adjust support measures aligned with orientations",
    "Support consumers' enactment of present practices",
    "Measure progress of the intervention",
    "Continuously revise the intervention",
]

EXPECTED_MANAGER_STEP_TITLES = [
    "Determine the prescribed future",
    "Monitor initial responses and map future-making orientations",
    "Diagnose future-making challenges",
    "Select orientation-sensitive support strategies",
    "Select messaging strategies",
    "Support consumers at multiple touchpoints",
]

POLICY_ROADMAP_STEPS = [
    ("1", EXPECTED_POLICY_STEP_TITLES[0],
     "Determine the future prescribed by the intervention and what it means for existing consumer practices."),
    ("2", EXPECTED_POLICY_STEP_TITLES[1],
     "Monitor the initial responses, for example on social media and in public consultations, and map which "
     "orientations appear. Orientations are patterns of performing future-making, not segments."),
    ("3", EXPECTED_POLICY_STEP_TITLES[2],
     "Diagnose the future-making challenges that emerge when orientations interact: convoluted evaluations, "
     "confrontational negotiations, and competing enactments."),
    ("4", EXPECTED_POLICY_STEP_TITLES[3],
     "Determine support measures aligned with the orientations present, then monitor and adjust them."),
    ("5", EXPECTED_POLICY_STEP_TITLES[4],
     "Support consumers' enactment of present practices by providing the infrastructure and capabilities they "
     "need to navigate the change, responding to what the diagnostic steps show."),
    ("6", EXPECTED_POLICY_STEP_TITLES[5],
     "Measure how the intervention is progressing."),
    ("7", EXPECTED_POLICY_STEP_TITLES[6],
     "Revise the intervention continuously and provide more support as needed."),
]

MANAGER_ROADMAP_STEPS = [
    ("1", EXPECTED_MANAGER_STEP_TITLES[0],
     "Identify the future prescribed by the intervention and the practices it asks consumers to change."),
    ("2", EXPECTED_MANAGER_STEP_TITLES[1],
     "Monitor how consumers respond to the intervention, for the purpose of supporting them through the change."),
    ("3", EXPECTED_MANAGER_STEP_TITLES[2],
     "Diagnose how consumers are responding and which future-making challenges are emerging."),
    ("4", EXPECTED_MANAGER_STEP_TITLES[3],
     "Select support strategies sensitive to the orientations present. Web Appendix Table WE2 lists actions "
     "to avoid when developing support mechanisms."),
    ("5", EXPECTED_MANAGER_STEP_TITLES[4],
     "Select messaging strategies that keep consumers informed of the required change, its benefits, and the "
     "support available. Frame messages so that one orientation is not mobilized while others are alienated."),
    ("6", EXPECTED_MANAGER_STEP_TITLES[5],
     "Support consumers at multiple touchpoints along their journey with the organization, so that they have a "
     "positive experience on the pathway toward the preferred future."),
]

# Step 3 diagnostic questions. The two roadmaps use different wording on purpose.
POLICY_STEP3_QUESTIONS = {
    "CONVOLUTED_EVALUATIONS":
        "Are evaluations of the prescribed future diverging in their assumptions, evidence, or temporal horizons?",
    "CONFRONTATIONAL_NEGOTIATIONS":
        "Are advocacy, questioning, rejection, and contestation occurring at the same time, in ways that block a shared direction?",
    "COMPETING_ENACTMENTS":
        "Are practices that accelerate, delay, prevent, or reroute the change pointing toward incompatible futures?",
}

MANAGER_STEP3_QUESTIONS = {
    "CONVOLUTED_EVALUATIONS":
        "Where do consumers' evaluations of the change diverge, and which evidence or timeline does each group rely on?",
    "CONFRONTATIONAL_NEGOTIATIONS":
        "Which consumers are advocating, questioning, rejecting, or contesting the change, and are these positions escalating?",
    "COMPETING_ENACTMENTS":
        "Which present practices (adopting, delaying, refusing, or rerouting) conflict at the points where consumers must change?",
}

POLICY_ORIENTATION_GUIDANCE = {
    "CATALYZER": {
        "lens": "Consumers with this orientation see the prescribed future as the rightly determined one and the change as already happening (urgency narrative; Table 2).",
        "implications": (
            "Catalyzers can help normalize the move toward the prescribed future, and their enthusiasm can be "
            "leveraged to bring other consumers along, for example through word-of-mouth or testimonials. The "
            "manuscript's aim is not to convert all consumers to this orientation."
        ),
        "monitor": [
            "Urgency and inevitability language (Table 2: the future is close).",
            "Early adoption and advocacy, such as recommendations to others.",
        ],
        "objective": "Use this enthusiasm to normalize the change without requiring other consumers to share this orientation.",
    },
    "AMBIVALENT": {
        "lens": "Consumers with this orientation see the prescribed future as contingent: change is uncertain and conditions are not yet ready (pragmatic narrative; Table 2).",
        "implications": (
            "Ambivalence can play a constructive role. It prompts reflection, surfaces weak points, and calls for "
            "stronger infrastructure before large-scale adoption, which can make transitions more robust. These "
            "consumers also need reassurance that other actors are doing their share."
        ),
        "monitor": [
            "Conditional language and the specific conditions named, such as charging, price, servicing, or batteries.",
            "Continued use of current vehicles while the market is monitored.",
        ],
        "objective": "Turn the conditions these consumers name into specific questions the intervention can answer.",
    },
    "RESISTANT": {
        "lens": "Consumers with this orientation frame the prescribed future through a control narrative, as coercive and inequitable, and seek to protect the status quo (Table 2).",
        "implications": (
            "Examining this orientation can provide valuable insight into consumers' reasons for pushback. The "
            "manuscript notes that these consumers feel they have no say in designing the future."
        ),
        "monitor": [
            "Language about coercion, imposition, and loss of choice (control narrative).",
            "Refusals to adopt and intentions to keep current vehicles.",
        ],
        "objective": "Understand the reasons for pushback and give these consumers a way to be heard in designing the future.",
    },
    "EXPANDER": {
        "lens": "Consumers with this orientation situate the intervention within wider systems and see the change as needing to be broader (bigger picture narrative; Table 2).",
        "implications": (
            "Expander critique broadens the scope of the envisioned change and encourages systemic innovation beyond "
            "immediate next steps. It highlights issues the intervention may not have considered, such as cities "
            "designed around cars."
        ),
        "monitor": [
            "Claims that the intervention does not address the underlying problem.",
            "Calls for alternatives such as public transport, cycling, or denser cities.",
        ],
        "objective": "Use the broader scope to identify complementary pathways and issues the intervention has not yet considered.",
    },
}

MANAGER_ORIENTATION_GUIDANCE = {
    "CATALYZER": {
        "lens": POLICY_ORIENTATION_GUIDANCE["CATALYZER"]["lens"],
        "implications": (
            "Catalyzers simplify their evaluation and accelerate enactment. They can normalize the change through "
            "word-of-mouth or testimonials, but messages should not mobilize them at the cost of alienating other orientations."
        ),
        "monitor": POLICY_ORIENTATION_GUIDANCE["CATALYZER"]["monitor"],
        "objective": "Use their experience to normalize the change and bring other consumers along.",
        "avoid": None,  # Web Appendix Table WE2 not yet provided; the interface hides this line when empty.
    },
    "AMBIVALENT": {
        "lens": POLICY_ORIENTATION_GUIDANCE["AMBIVALENT"]["lens"],
        "implications": (
            "Ambivalent consumers stall their evaluation and delay enactment. Their named conditions can be addressed "
            "at specific touchpoints, and the manuscript notes that avoiding rushed decisions reduces the risk of unstable adoption."
        ),
        "monitor": POLICY_ORIENTATION_GUIDANCE["AMBIVALENT"]["monitor"],
        "objective": "Address the specific conditions these consumers name, and avoid rushing their decisions.",
        "avoid": None,
    },
    "RESISTANT": {
        "lens": POLICY_ORIENTATION_GUIDANCE["RESISTANT"]["lens"],
        "implications": (
            "Resistant consumers avoid evaluation, reject the prescribed future, and seek to prevent enactment. Their "
            "reasons for pushback are useful information for the support design, and messages should not alienate them."
        ),
        "monitor": POLICY_ORIENTATION_GUIDANCE["RESISTANT"]["monitor"],
        "objective": "Understand the reasons for pushback and make sure messaging does not alienate these consumers.",
        "avoid": None,
    },
    "EXPANDER": {
        "lens": POLICY_ORIENTATION_GUIDANCE["EXPANDER"]["lens"],
        "implications": (
            "Expanders complexify their evaluation, contest the prescribed future, and reroute practice toward broader "
            "alternatives. They reveal broader value propositions and complementary pathways."
        ),
        "monitor": POLICY_ORIENTATION_GUIDANCE["EXPANDER"]["monitor"],
        "objective": "Consider the systemic issues these consumers raise and identify complementary or alternative pathways.",
        "avoid": None,
    },
}

# Challenge guidance: wording follows the manuscript's descriptions of each challenge and the Discussion.
CHALLENGE_POLICY_GUIDANCE = {
    "CONVOLUTED_EVALUATIONS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["CONVOLUTED_EVALUATIONS"],
        "action": (
            "Divergent assumptions, evidence, and temporal horizons delay coherent sensemaking and can interfere with "
            "desirable practice change. Anticipate this when designing the intervention, and plan for the evidence and "
            "timelines consumers rely on."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["CONFRONTATIONAL_NEGOTIATIONS"],
        "action": (
            "Confrontation makes it harder to accommodate change and to bring a future into being collectively. Accommodating "
            "diverse preferred futures creates opportunities for multiple pathways, rather than enforcing only the prescribed change."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["COMPETING_ENACTMENTS"],
        "action": (
            "Divergent enactments create volatility and hinder progress toward the prescribed future and its stabilization. "
            "Support multiple viable pathways, and keep the prescribed future from becoming the only accepted route."
        ),
    },
}

CHALLENGE_MANAGER_GUIDANCE = {
    "CONVOLUTED_EVALUATIONS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["CONVOLUTED_EVALUATIONS"],
        "action": (
            "Where evaluations diverge at a touchpoint, address the evidence or timeline consumers rely on, so that the support "
            "offered fits their situation."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["CONFRONTATIONAL_NEGOTIATIONS"],
        "action": (
            "Check that messages do not mobilize one orientation while alienating others, because confrontation makes "
            "collective change harder to achieve."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["COMPETING_ENACTMENTS"],
        "action": (
            "Support consumers through the enactment they are attempting at each touchpoint, including delayed, refused, or "
            "rerouted practices, rather than only the prescribed change."
        ),
    },
}

MANAGER_MESSAGING_NOTE = (
    "Message framing should ensure that one orientation is not mobilized while others are alienated. "
    "(Manuscript, managerial roadmap, step 5.)"
)

# ─────────────────────────────────────────
# PAPER EXAMPLES (demonstration and internal check)
# ─────────────────────────────────────────
PF_EV = (
    "Transition all vehicles to Zero Emission Vehicles (ZEV), including electric vehicles, to achieve "
    "Australia's net-zero emissions targets, as prescribed by Australia's National Electric Vehicle Strategy (2023)"
)

DEMO_EXAMPLES = [
    {
        "id": "joe", "titulo": "Joe (public consultation): Negotiation · Advocate · Catalyzer",
        "comment": (
            "We are already so far behind! We need to sprint to catch up. We should be WORLD LEADERS in "
            "solar and battery manufacturing. Why are we not using our own minerals to make batteries for "
            "EVs on a global scale??"
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "NEGOTIATION", "subtype": "ADVOCATE", "orientation": "CATALYZER"},
        "nota_paper": "Calls for acceleration and recruits others. Negotiation seeks to mobilize a collective trajectory.",
    },
    {
        "id": "clara", "titulo": "Clara (interview): Evaluation · Stall · Ambivalent",
        "comment": (
            "Living in Outback Northwest Queensland there's no charging stations at the time. I did like the "
            "appeal of an electric vehicle mainly because you don't have to put fuel in it, which is great. "
            "But just at the time I went and bought a fairly decent car for five and a half grand. If we went "
            "on a driving holiday, we would take our big car. So if we got an EV it would just be our daily run around."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "EVALUATION", "subtype": "STALL", "orientation": "AMBIVALENT"},
        "nota_paper": "Evaluates the EV in terms of feasibility (charging, long trips) without redesigning her routines: an evaluation that stalls.",
    },
    {
        "id": "martin", "titulo": "Martin (social media): Negotiation · Question · Ambivalent",
        "comment": "Have you thought about what they are gonna do with all the batteries once they expire because they aren't recyclable?",
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "NEGOTIATION", "subtype": "QUESTION", "orientation": "AMBIVALENT"},
        "nota_paper": "Asks whether the system is ready. The question sets a condition that must be met before moving forward.",
    },
    {
        "id": "raj", "titulo": "Raj (video): Negotiation · Reject · Resistant",
        "comment": "We don't need politicians and their cronies telling us what sort of car we can have.",
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT"},
        "nota_paper": "Rejects the authority behind the intervention and defends freedom of choice.",
    },
    {
        "id": "startt", "titulo": "StarT (news): Enactment · Prevent · Resistant",
        "comment": (
            "I for one WILL NOT be forced into an elec vehicle and spend half my travel time charging the damn "
            "thing to go to hell."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "ENACTMENT", "subtype": "PREVENT", "orientation": "RESISTANT"},
        "nota_paper": "Declares that they will not adopt an EV, that is, they keep their current practice and refuse to change.",
    },
    {
        "id": "johnny", "titulo": "Johnny (Facebook): Enactment · Accelerate · Catalyzer",
        "comment": (
            "Toyota is still very much trying to slow down the transition to EVs. We have ordered two Teslas "
            "that will be delivered hopefully this year. We are selling our Prado and it looks like we are "
            "going to sell our last Toyota car."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "ENACTMENT", "subtype": "ACCELERATE", "orientation": "CATALYZER"},
        "nota_paper": "Buys EVs and sells combustion vehicles: materializes the prescribed future.",
    },
    {
        "id": "peter", "titulo": "Peter (interview): Evaluation · Complexify · Expander",
        "comment": (
            "The embodied carbon in a new vehicle is more than the emissions that are going to be produced by "
            "the current vehicle over the course of its lifetime until it falls apart. So that's the plan: to "
            "extract maximum value out of that current vehicle until it is no longer functional. I am at the "
            "moment on a waiting list for a new electric cargo bike because my current electric cargo bike is "
            "about seven years old."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "EVALUATION", "subtype": "COMPLEXIFY", "orientation": "EXPANDER"},
        "nota_paper": (
            "Evaluates embodied emissions and decides to keep his car. The cargo bike is a secondary reroute practice: "
            "when secondary classifications are enabled, the tool can detect it."
        ),
    },
    {
        "id": "phillip", "titulo": "Phillip (public consultation): Enactment · Reroute · Expander",
        "comment": (
            "I uprooted my life and moved from the Sunshine Coast to Melbourne with some of my strongest "
            "reasoning being the ability to use public transport, ride a bike around and use a car as little as possible."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "ENACTMENT", "subtype": "REROUTE", "orientation": "EXPANDER"},
        "nota_paper": "Moved home to depend less on the car: redirects his practice toward a broader future.",
    },
]

DEMO_BY_ID = {ex["id"]: ex for ex in DEMO_EXAMPLES}

# Paper conversation (forum). Links are adjacency: each user replies to the previous one.
DEMO_CONVERSATION = """User 1: EVs will be on an exponential adoption curve. Everyone will want one... Nobody will want an expensive 2nd hand ICE... Globally, governments are going to start making fossil fuels very expensive. T-A-X-E-S will be levied on this foul, polluting rubbish we are all burning today... Or are you advocating that we go back to bicycles and horses, or maybe just buses?

User 2: Better tell that to the Prius owners replacing their batteries. My car is now 13 years old... Batteries wear over time... so what magic bullet have you discovered that defies physics...? Once someone like me can get a used EV for <$10k, and have the battery replaced cheaply, then I'll agree with you... I'm not anti EV, I'm just realistic about costs and time frames.

User 3: Nope, I'm not confused, thanks for the concern though... If he bought an ICE car now, he will get 6-10 years use out of it and sell it for scrap value and registration. Not even close to the financial ruin you are trying to peddle... Technology adoption curves typically look like bell curves... not what you are suggesting... This is delusional.

User 4: I fully get what you're saying, it's not rocket science, but that's not what I'm on about... I simply object to being told I'm an idiot... I'd like to see passenger cars filled with passengers, less cars on the road, less money spent on new roads!... Where people may simply drive less."""

DEMO_CONVERSATION_REFERENCE = {
    "user1": {"activity": "NEGOTIATION", "subtype": "ADVOCATE", "orientation": "CATALYZER"},
    "user2": {"activity": "NEGOTIATION", "subtype": "QUESTION", "orientation": "AMBIVALENT"},
    "user3": {"activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT"},
    "user4": {"activity": "NEGOTIATION", "subtype": "CONTEST", "orientation": "EXPANDER"},
}

# ─────────────────────────────────────────
# MODEL OUTPUT AND CONSISTENCY
# ─────────────────────────────────────────
EXPECTED_OUTPUT_SCHEMA_KEYS = [
    "prescribed_future_acknowledged", "main_activity", "activity_subtype",
    "activity_rationale", "main_orientation", "orientation_rationale",
    "secondary_classifications", "narrative_identified", "emotions_identified",
    "temporality_identified", "notable_conditions_of_adoption",
    "supporting_text", "context_note", "input_scope_warning",
]


def fill_schema(result: dict) -> dict:
    for key in EXPECTED_OUTPUT_SCHEMA_KEYS:
        if key not in result:
            result[key] = [] if key == "secondary_classifications" else ""
    return result


def get_secondary_classifications(result: dict) -> list:
    items = result.get("secondary_classifications") or []
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if isinstance(item, dict):
            cleaned.append({
                "activity": _clean_enum(item.get("activity", "")).upper(),
                "activity_subtype": _clean_enum(item.get("activity_subtype", "")).upper(),
                "orientation": _clean_enum(item.get("orientation", "")).upper(),
                "rationale": str(item.get("rationale", "")),
            })
    return cleaned


def normalize_classification(activity: str, subtype: str, orientation: str):
    """Corrects an activity/subtype/orientation combination using the Section D matrix.
    Returns (activity, subtype, orientation, notes)."""
    notes = []
    activity = activity if activity in ACTIVITY_META else ""
    subtype = subtype if subtype in SUBTYPE_TO_ACTIVITY else ""
    orientation = orientation if orientation in ORIENTATIONS else ""

    if subtype and activity and SUBTYPE_TO_ACTIVITY[subtype] != activity:
        notes.append(f"activity corrected from {activity} to {SUBTYPE_TO_ACTIVITY[subtype]}")
        activity = SUBTYPE_TO_ACTIVITY[subtype]
    if subtype and not activity:
        activity = SUBTYPE_TO_ACTIVITY[subtype]
    if subtype and not orientation:
        orientation = SUBTYPE_TO_ORIENTATION[subtype]
        notes.append(f"orientation derived from subtype {subtype}")
    if activity and orientation:
        expected = ORIENTATION_TO_SUBTYPE[(activity, orientation)]
        if subtype != expected:
            notes.append(f"subtype adjusted from {subtype or 'empty'} to {expected}")
            subtype = expected
    return activity, subtype, orientation, notes


def enforce_consistency(result: dict) -> dict:
    notes = []
    act, sub, ori, n = normalize_classification(
        _clean_enum(result.get("main_activity", "")).upper(),
        _clean_enum(result.get("activity_subtype", "")).upper(),
        _clean_enum(result.get("main_orientation", "")).upper(),
    )
    notes += n
    result["main_activity"], result["activity_subtype"], result["main_orientation"] = act, sub, ori
    main = (act, sub, ori)

    secondary = []
    for i, item in enumerate(get_secondary_classifications(result)):
        a, s, o, n = normalize_classification(item["activity"], item["activity_subtype"], item["orientation"])
        if not (a and s and o):
            notes.append(f"secondary {i + 1} discarded (incomplete)")
            continue
        if (a, s, o) == main or any(
            (a, s, o) == (x["activity"], x["activity_subtype"], x["orientation"]) for x in secondary
        ):
            continue
        notes += [f"secondary {i + 1}: {x}" for x in n]
        item.update(activity=a, activity_subtype=s, orientation=o)
        secondary.append(item)
    result["secondary_classifications"] = secondary[:MAX_SECONDARY]
    if notes:
        result["_consistency_note"] = "; ".join(notes)[:MAX_NOTE_CHARS]
    return result


def enforce_context_metadata(result: dict, context_available: bool, context_type: str) -> dict:
    result["context_used"] = bool(context_available)
    result["context_type"] = context_type if context_type in VALID_CONTEXT_TYPES else "NONE"
    if not context_available:
        result["context_note"] = ""
    return result


# ─────────────────────────────────────────
# MODEL CALL WITH RETRIES
# ─────────────────────────────────────────
def _call_model(system_prompt: str, user_message: str) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with openai_semaphore:
                response = get_openai_client().chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            return response.choices[0].message.content or ""
        except (openai.RateLimitError, openai.APITimeoutError,
                openai.APIConnectionError, openai.InternalServerError) as e:
            last_error = e
            logger.warning("Attempt %d failed (%s); retrying.", attempt + 1, type(e).__name__)
            time.sleep(2 ** attempt)
    raise RuntimeError(
        "The analysis service did not respond after several attempts. Please try again in a few minutes."
    ) from last_error


def _parse_json(content: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError("The model response was not valid JSON. Please try again.") from e
    if not isinstance(data, dict):
        raise ValueError("The model response does not have the expected structure.")
    return data


def analyze_comment(prescribed_future: str, focal_text: str, context_text: str = "",
                    context_type: str = "NONE", is_consultation: bool = False,
                    allow_secondary: bool = True) -> dict:
    focal_text = (focal_text or "").strip()
    if not focal_text:
        raise ValueError("The comment is empty.")

    if is_consultation:
        focal_label = "FOCAL RESPONSE TO CLASSIFY"
        context_label = "CONSULTATION/POLICY CONTEXT -- USE FOR INTERPRETATION BUT DO NOT CLASSIFY"
    else:
        focal_label = "FOCAL COMMENT TO CLASSIFY"
        context_label = ("CONTEXT (parent comment / nearby comments / original post) -- "
                         "USE FOR INTERPRETATION BUT DO NOT CLASSIFY")

    context_available = bool((context_text or "").strip())
    context_block = context_text.strip() if context_available else "No context is available for this comment."
    secondary_instruction = (
        "You MAY return up to two secondary classifications if substantively and separably supported, per Section G."
        if allow_secondary else
        "Do NOT return any secondary classifications for this request. secondary_classifications must be an empty list."
    )

    user_message = f"""
PRESCRIBED FUTURE:
{prescribed_future}

{focal_label}:
{focal_text}

{context_label}:
{context_block}

SECONDARY CLASSIFICATIONS FOR THIS REQUEST: {secondary_instruction}

Apply the Decision Procedure in Section E in order. FIRST ask whether the focal comment's overarching function is to Advocate, Question, Reject, or Contest a collective trajectory, BEFORE considering how much of the comment reads as technical, factual, or evaluative in isolation. Verify that activity_subtype belongs to the matrix row of its own orientation (Section D). Do not produce any future-making challenge, Fragile Futures assessment, or policy/managerial recommendation.
"""
    content = _call_model(SYSTEM_PROMPT, user_message)
    parsed = fill_schema(_parse_json(content))
    parsed = enforce_consistency(parsed)
    if not (parsed["main_activity"] and parsed["activity_subtype"] and parsed["main_orientation"]):
        raise ValueError("The model did not return a complete classification. Please try again.")
    if not allow_secondary:
        parsed["secondary_classifications"] = []
    return enforce_context_metadata(parsed, context_available, context_type)


def _safe_analyze(prescribed_future: str, rec: dict) -> dict:
    """Never raises: returns {'_error': message} if anything fails."""
    try:
        return analyze_comment(
            prescribed_future, rec["comment_text"], rec.get("context_text", ""),
            rec.get("context_type", "NONE"), rec.get("is_consultation", False), True,
        )
    except (ValueError, RuntimeError) as e:
        return {"_error": str(e)}
    except Exception:
        logger.exception("Unexpected error while analyzing comment %s", rec.get("comment_id"))
        return {"_error": "Unexpected error while analyzing this comment (see the server log)."}


def analyze_document(records: list, prescribed_future: str, progress_callback=None) -> list:
    total = len(records)
    if total == 0:
        return []
    results = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOC_MAX_WORKERS) as executor:
        futures = {executor.submit(_safe_analyze, prescribed_future, rec): pos
                   for pos, rec in enumerate(records)}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            pos = futures[future]
            rec = records[pos]
            r = future.result()
            r.update({
                "_comment_id": rec["comment_id"],
                "_thread_id": rec["thread_id"],
                "_parent_comment_id": rec.get("parent_comment_id"),
                "_comment_text": rec["comment_text"],
                "_author": rec.get("author", ""),
                "_original_index": rec["original_index"],
            })
            results[pos] = r
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)
    return sorted(results, key=lambda r: r["_original_index"])


# ─────────────────────────────────────────
# DATA READING AND THREAD STRUCTURE
# ─────────────────────────────────────────
def has_comment_text_column(df: pd.DataFrame) -> bool:
    cols = {str(c).lower().strip() for c in df.columns}
    return bool(cols & {"comment_text", "text", "comment"})


def strip_trailing_orientation_label(text: str) -> str:
    return re.sub(
        r'\(\s*(Catalyzer|Ambivalent|Resistant|Expander)\s+orientation\s*\)\.?\s*$',
        '', text, flags=re.IGNORECASE,
    ).strip()


USER_LABEL_PATTERN = re.compile(r'(?im)^\s*User\s*\d+\s*:\s*')


def build_comment_records_from_user_labels(text: str) -> list:
    matches = list(USER_LABEL_PATTERN.finditer(text or ""))
    if len(matches) < 2:
        return []
    kept = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        cleaned = re.sub(r'\s+', ' ', strip_trailing_orientation_label(text[m.end():end])).strip()
        if cleaned:
            kept.append((i + 1, cleaned))
    records = []
    for pos, (num, cleaned) in enumerate(kept):
        records.append({
            "comment_id": f"user{num}",
            "thread_id": LABELED_THREAD,
            # Adjacency: each user replies to the previous one (explicit assumption, not verified).
            "parent_comment_id": f"user{kept[pos - 1][0]}" if pos > 0 else None,
            "author": f"User {num}", "timestamp": "",
            "comment_text": cleaned, "original_index": pos,
        })
    return records


def build_comment_records_from_paragraphs(text: str):
    """Returns (records, number_discarded)."""
    blocks = [re.sub(r'\s+', ' ', p).strip()
              for p in re.split(r'\n\s*\n+', (text or "").strip()) if p.strip()]
    records, discarded = [], 0
    for item in blocks:
        item = strip_trailing_orientation_label(item)
        if len(item.split()) < 2:
            discarded += 1
            continue
        records.append({
            "comment_id": f"c{len(records)}", "thread_id": DEFAULT_THREAD,
            "parent_comment_id": None, "author": "", "timestamp": "",
            "comment_text": item, "original_index": len(records),
        })
    return records, discarded


def build_comment_records_from_csv(df: pd.DataFrame) -> list:
    cols = {str(c).lower().strip(): c for c in df.columns}
    text_col = cols.get("comment_text") or cols.get("text") or cols.get("comment")
    if not text_col:
        return []

    def safe_get(row, name, default=""):
        c = cols.get(name)
        if c and pd.notna(row.get(c)):
            return str(row.get(c)).strip()
        return default

    records = []
    for i, row in df.iterrows():
        raw = row.get(text_col)
        text = str(raw).strip() if pd.notna(raw) else ""
        if not text:
            continue
        records.append({
            "comment_id": safe_get(row, "comment_id", f"c{i}"),
            "thread_id": safe_get(row, "thread_id", DEFAULT_THREAD) or DEFAULT_THREAD,
            "parent_comment_id": safe_get(row, "parent_comment_id", "") or None,
            "author": safe_get(row, "author", ""),
            "timestamp": safe_get(row, "timestamp", ""),
            "comment_text": text, "original_index": len(records),
        })
    return records


def extract_public_consultation_responses(text: str, min_words: int = 4) -> list:
    text = re.sub(r'\s+', ' ', (text or "").strip())
    matches = list(re.finditer(r'\b(\d{6,7})\s+(?:Name\s+withheld|[A-Z][a-z]+)', text))
    responses = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[m.start():end].strip()
        block = re.sub(r'^\d{6,7}\s+(?:Name\s+withheld|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*', '', block)
        block = re.sub(r'Option\s+[ABC]\s*-\s*\w+,?\s*', '', block, flags=re.IGNORECASE)
        block = re.sub(r'\b(Yes|No|NULL)\s*$', '', block, flags=re.IGNORECASE).strip()
        block = re.sub(r'\s{2,}', ' ', block).strip(' ,.-')
        if block and block.upper() != "NULL" and len(block.split()) >= min_words:
            responses.append({"id": m.group(1), "text": block})
    return responses


def build_comment_records_from_consultation(text: str) -> list:
    return [{
        "comment_id": f"resp_{item['id']}", "thread_id": DEFAULT_THREAD,
        "parent_comment_id": None, "author": "", "timestamp": "",
        "comment_text": item["text"], "original_index": i,
    } for i, item in enumerate(extract_public_consultation_responses(text))]


def ensure_unique_ids(records: list) -> int:
    """Renames duplicate identifiers. Returns how many were renamed."""
    used, renamed = set(), 0
    for r in records:
        base = str(r["comment_id"])
        new_id, n = base, 1
        if base in used:
            renamed += 1
        while new_id in used:
            new_id = f"{base}_{n}"
            n += 1
        used.add(new_id)
        r["comment_id"] = new_id
    return renamed


def select_within_limit(records: list, limit: int):
    """Selects comments without splitting threads. Returns (selected, omitted_count)."""
    groups, index = [], {}
    for r in records:
        tid = r["thread_id"]
        if tid == DEFAULT_THREAD:
            groups.append([r])
            continue
        if tid not in index:
            index[tid] = len(groups)
            groups.append([])
        groups[index[tid]].append(r)
    selected, omitted, total = [], 0, 0
    for g in groups:
        if total + len(g) <= limit:
            selected += g
            total += len(g)
        else:
            omitted += len(g)
    return selected, omitted


def index_threads(records: list):
    by_id = {r["comment_id"]: r for r in records}
    thread_order = {}
    for r in records:
        thread_order.setdefault(r["thread_id"], []).append(r["comment_id"])
    return by_id, thread_order


def build_context(record: dict, by_id: dict, thread_order: dict,
                  consultation_prompt: str = "", is_consultation: bool = False):
    """Returns (context_text, context_type, has_context).
    Threads are used only when they are real (never for comments without a thread)."""
    parts, context_type = [], "NONE"
    focal_id = record["comment_id"]
    thread_id = record["thread_id"]
    parent_id = record.get("parent_comment_id")

    if thread_id != DEFAULT_THREAD:
        order = thread_order.get(thread_id, [focal_id])
        idx = order.index(focal_id) if focal_id in order else 0

        parent = by_id.get(parent_id) if parent_id else None
        if parent and parent["thread_id"] != thread_id:
            parent = None
        if parent:
            parts.append(f"PARENT COMMENT:\n{parent['comment_text']}")
            context_type = "PARENT_REPLY"

        for pid in order[max(0, idx - 2):idx]:
            if pid == parent_id:
                continue
            parts.append(f"PRECEDING COMMENT:\n{by_id[pid]['comment_text']}")
            if context_type == "NONE":
                context_type = "THREAD_WINDOW"

        following = next((by_id[c] for c in order if by_id[c].get("parent_comment_id") == focal_id), None)
        if following:
            parts.append(f"FOLLOWING REPLY:\n{following['comment_text']}")
            if context_type == "NONE":
                context_type = "THREAD_WINDOW"

        if order:
            root = by_id[order[0]]
            if root["comment_id"] not in (focal_id, parent_id) and not root.get("parent_comment_id"):
                parts.append(f"ORIGINAL POST:\n{root['comment_text']}")
                if context_type == "NONE":
                    context_type = "ORIGINAL_POST"

    if is_consultation and consultation_prompt and consultation_prompt.strip():
        parts.append(f"CONSULTATION QUESTION / POLICY CONTEXT:\n{consultation_prompt.strip()}")
        if context_type == "NONE":
            context_type = "CONSULTATION_PROMPT"

    return "\n\n".join(parts), context_type, bool(parts)


def prepare_records(records: list, consultation_prompt: str = "", is_consultation: bool = False) -> list:
    by_id, order = index_threads(records)
    for rec in records:
        ctx, ctype, _ = build_context(rec, by_id, order, consultation_prompt, is_consultation)
        rec["context_text"] = ctx
        rec["context_type"] = ctype
        rec["is_consultation"] = is_consultation
    return records


def pseudonymize(author: str) -> str:
    if not author:
        return ""
    digest = hashlib.sha256(author.encode("utf-8")).hexdigest()[:6].upper()
    return f"Participant {digest}"


# ─────────────────────────────────────────
# AGGREGATION (corpus level). Never diagnoses a challenge automatically.
# ─────────────────────────────────────────
def compute_dominant_distributions(results: list) -> dict:
    valid = [r for r in results if r and "_error" not in r]
    orientation_counts, activity_counts, subtype_counts = {}, {}, {}
    for r in valid:
        ori = _clean_enum(r.get("main_orientation", "")).upper()
        act = _clean_enum(r.get("main_activity", "")).upper()
        sub = _clean_enum(r.get("activity_subtype", "")).upper()
        if ori:
            orientation_counts[ori] = orientation_counts.get(ori, 0) + 1
        if act:
            activity_counts[act] = activity_counts.get(act, 0) + 1
        if act and sub:
            key = f"{act} / {sub}"
            subtype_counts[key] = subtype_counts.get(key, 0) + 1
    return {
        "n_analyzed": len(valid),
        "orientation_counts": orientation_counts,
        "activity_counts": activity_counts,
        "subtype_counts": subtype_counts,
    }


def orientation_coexistence(results: list) -> dict:
    """How many orientations appear in the set, and their shares."""
    counts = {o: 0 for o in ORIENTATION_ORDER}
    for r in results:
        o = _clean_enum(r.get("main_orientation", "")).upper()
        if o in counts:
            counts[o] += 1
    total = sum(counts.values())
    present = [o for o in ORIENTATION_ORDER if counts[o] > 0]
    return {
        "counts": counts,
        "present": present,
        "n_present": len(present),
        "total": total,
        "shares": {o: (counts[o] / total if total else 0.0) for o in ORIENTATION_ORDER},
    }


def orientations_by_thread(results: list) -> dict:
    """For real threads with at least two comments: which orientations appear in each."""
    table = {}
    for r in results:
        tid = r.get("_thread_id", DEFAULT_THREAD)
        if tid == DEFAULT_THREAD:
            continue
        entry = table.setdefault(tid, {"n": 0, "counts": {o: 0 for o in ORIENTATION_ORDER}})
        entry["n"] += 1
        o = _clean_enum(r.get("main_orientation", "")).upper()
        if o in entry["counts"]:
            entry["counts"][o] += 1
    return {tid: e for tid, e in table.items() if e["n"] >= 2}


def compute_challenge_review_candidates(results: list) -> dict:
    """Flags LINKED pairs with the same activity but different performance, for human review.
    Does not diagnose the challenge and does not assign scores."""
    valid = [r for r in results if r and "_error" not in r]
    by_id = {r.get("_comment_id"): r for r in valid if r.get("_comment_id")}
    candidates = {k: [] for k in CHALLENGE_DEFINITIONS}
    for r in valid:
        parent_id = r.get("_parent_comment_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        act_r = _clean_enum(r.get("main_activity", "")).upper()
        if act_r != _clean_enum(parent.get("main_activity", "")).upper():
            continue
        same_ori = (_clean_enum(r.get("main_orientation", "")).upper()
                    == _clean_enum(parent.get("main_orientation", "")).upper())
        same_sub = (_clean_enum(r.get("activity_subtype", "")).upper()
                    == _clean_enum(parent.get("activity_subtype", "")).upper())
        if same_ori and same_sub:
            continue
        key = next((k for k, v in CHALLENGE_DEFINITIONS.items() if v["activity"] == act_r), None)
        if key:
            candidates[key].append({"parent": parent, "reply": r})
    return candidates


# ─────────────────────────────────────────
# STRUCTURAL TESTS AND CONSISTENCY CHECK (from the terminal)
# ─────────────────────────────────────────
REQUIRED_ORIENTATIONS = set(ORIENTATION_ORDER)
REQUIRED_SUBTYPES = {
    "EVALUATION": {"SIMPLIFY", "STALL", "AVOID", "COMPLEXIFY"},
    "NEGOTIATION": {"ADVOCATE", "QUESTION", "REJECT", "CONTEST"},
    "ENACTMENT": {"ACCELERATE", "DELAY", "PREVENT", "REROUTE"},
}
REQUIRED_CHALLENGES = {"CONVOLUTED_EVALUATIONS", "CONFRONTATIONAL_NEGOTIATIONS", "COMPETING_ENACTMENTS"}
OBSOLETE_TERMS_FILE = "terminos_prohibidos.txt"  # optional: one term per line


def check_obsolete_terms():
    base = Path(__file__).resolve().parent
    list_file = base / OBSOLETE_TERMS_FILE
    if not list_file.exists():
        return True, f"{OBSOLETE_TERMS_FILE} not found (test skipped)"
    terms = [t.strip().lower() for t in list_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    hits = []
    for name in ("motor.py", "app.py"):
        path = base / name
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            hits += [f"{name}: '{t}'" for t in terms if t in text]
    return not hits, "No findings" if not hits else "; ".join(hits)


def run_framework_validation() -> list:
    """Returns a list of (name, passed, detail). No API calls."""
    tests = []

    def add(name, ok, detail=""):
        tests.append((name, bool(ok), detail))

    add("Exactly 4 orientations", set(ORIENTATIONS) == REQUIRED_ORIENTATIONS,
        f"Found: {sorted(ORIENTATIONS)}")
    for act, subs in REQUIRED_SUBTYPES.items():
        found = set(ACTIVITY_META.get(act, {}).get("subtypes", {}))
        add(f"{act}: 4 expected subtypes", found == subs, f"Found: {sorted(found)}")
    add("Orientation x activity matrix complete (12 combinations)",
        len(ORIENTATION_TO_SUBTYPE) == 12, f"Found: {len(ORIENTATION_TO_SUBTYPE)}")
    add("Policy roadmap: 7 steps in order",
        [t for _, t, _ in POLICY_ROADMAP_STEPS] == EXPECTED_POLICY_STEP_TITLES, "")
    add("Managerial roadmap: 6 steps in order",
        [t for _, t, _ in MANAGER_ROADMAP_STEPS] == EXPECTED_MANAGER_STEP_TITLES, "")
    add("Exactly 3 challenges", set(CHALLENGE_DEFINITIONS) == REQUIRED_CHALLENGES,
        f"Found: {sorted(CHALLENGE_DEFINITIONS)}")
    add("Challenge guidance complete (question and action)",
        all("diagnostic_question" in CHALLENGE_POLICY_GUIDANCE.get(k, {})
            and "action" in CHALLENGE_POLICY_GUIDANCE.get(k, {})
            and "diagnostic_question" in CHALLENGE_MANAGER_GUIDANCE.get(k, {})
            and "action" in CHALLENGE_MANAGER_GUIDANCE.get(k, {})
            for k in REQUIRED_CHALLENGES), "")
    add("Step 3 diagnostic questions differ between roadmaps",
        all(POLICY_STEP3_QUESTIONS[k] != MANAGER_STEP3_QUESTIONS[k] for k in REQUIRED_CHALLENGES), "")
    add("Orientation guidance complete in both roadmaps",
        all(o in POLICY_ORIENTATION_GUIDANCE and o in MANAGER_ORIENTATION_GUIDANCE
            for o in REQUIRED_ORIENTATIONS), "")
    add("Managerial guidance includes 'avoid'; policy guidance does not",
        all("avoid" in MANAGER_ORIENTATION_GUIDANCE[o] for o in REQUIRED_ORIENTATIONS)
        and all("avoid" not in POLICY_ORIENTATION_GUIDANCE[o] for o in REQUIRED_ORIENTATIONS), "")
    add("Demo examples: references valid under the matrix",
        all(ORIENTATION_TO_SUBTYPE.get((e["referencia"]["activity"], e["referencia"]["orientation"]))
            == e["referencia"]["subtype"] for e in DEMO_EXAMPLES), "")
    add("Demo examples cover all 4 orientations",
        {e["referencia"]["orientation"] for e in DEMO_EXAMPLES} == REQUIRED_ORIENTATIONS, "")
    add("Interface examples (joe, clara) exist",
        "joe" in DEMO_BY_ID and "clara" in DEMO_BY_ID, "")
    add("Demo conversation: one reference per orientation",
        {v["orientation"] for v in DEMO_CONVERSATION_REFERENCE.values()} == REQUIRED_ORIENTATIONS
        and len(DEMO_CONVERSATION_REFERENCE) == 4, "")
    ok, detail = check_obsolete_terms()
    add("No obsolete terms in the code", ok, detail)
    return tests


def run_consistency_suite() -> dict:
    """INTERNAL consistency check against the paper examples. The examples are also in the prompt,
    so this is not empirical validation or inter-coder reliability."""
    def run(ex):
        try:
            pred = analyze_comment(PF_EV, ex["comment"], ex["context"], ex["context_type"], False, True)
            return ex, pred, ""
        except (ValueError, RuntimeError) as e:
            return ex, None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=DOC_MAX_WORKERS) as pool:
        outcomes = list(pool.map(run, DEMO_EXAMPLES))

    rows = []
    for ex, pred, err in outcomes:
        ref = ex["referencia"]
        expected = (ref["activity"], ref["subtype"], ref["orientation"])
        if pred is None:
            rows.append({"example": ex["titulo"], "expected": expected, "predicted": None,
                         "match": False, "error": err})
            continue
        predicted = (pred["main_activity"], pred["activity_subtype"], pred["main_orientation"])
        rows.append({"example": ex["titulo"], "expected": expected, "predicted": predicted,
                     "match": predicted == expected, "error": ""})
    n = len(rows)
    return {"results": rows, "overall_agreement": (sum(r["match"] for r in rows) / n) if n else 0.0}


if __name__ == "__main__":
    failed = 0
    for name, ok, detail in run_framework_validation():
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        failed += 0 if ok else 1
    if "--consistencia" in sys.argv:
        report = run_consistency_suite()
        print(f"\nInternal consistency: {report['overall_agreement']:.0%} "
              "(the examples are also in the prompt; this is not validation).")
        for row in report["results"]:
            mark = "✔" if row["match"] else "✘"
            print(f"  {mark} {row['example']}: {row['predicted']} {row['error']}")
    sys.exit(1 if failed else 0)
