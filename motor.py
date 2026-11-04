from shared_resources import get_openai_client, openai_semaphore
import openai
import json
import re
import inspect
import concurrent.futures
import pandas as pd

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# CITATION CONSTANTS
# ─────────────────────────────────────────
PAPER_TITLE = "Futures in the Making: How Consumers Respond to Future-Oriented Interventions"
PAPER_URL   = "REPLACE_WITH_YOUR_DOI_OR_URL"

DOC_MAX_WORKERS = 5
DEFAULT_THREAD = "_default_thread_"

# ─────────────────────────────────────────
# SCOPE / INTERPRETIVE-USE NOTES
# ─────────────────────────────────────────
INTERPRETIVE_USE_NOTE = (
    "Interpretive-use note: future-making activities are interdependent, "
    "entangled, and recursive rather than sequential, and future-making "
    "orientations are diagnostic, socially patterned ways of performing "
    "future-making rather than fixed consumer or patient types or market "
    "segments. This application is an interactive demonstration tool, not "
    "a validated diagnostic instrument. Review outputs alongside the "
    "complete comment, any available context, and relevant behavioral "
    "evidence."
)

HOMEPAGE_DESCRIPTION = """
Use this application to support the mapping of consumer future-making in
response to a policy or market intervention, as theorized in **"Futures in
the Making: How Consumers Respond to Future-Oriented Interventions."**

Upload or paste consumer comments, consultation responses, forum posts, or
social-media conversations. The application classifies each **focal
comment**, using available parent comments, nearby comments, original
posts, or consultation prompts as interpretive context, to identify how
consumers evaluate, negotiate, and enact preferred futures.

**Scope statement:**

* The framework was developed through qualitative research on Australian
  Zero Emission Vehicle (ZEV) interventions.
* The authors expect the insights may apply to other future-oriented
  intervention contexts.
* Intervention type -- Fixed, Bounded, Flexible, or Open -- depends on the
  specific scope and prescriptiveness of the intervention being analyzed,
  not on its general domain.
* Future-making orientations -- Catalyzer, Ambivalent, Resistant, and
  Expander -- are diagnostic, socially patterned ways of performing
  future-making expressed through discourse and practice, **not** fixed
  consumer or patient types or market segments.
* Results require interpretation alongside context and relevant
  behavioral evidence. Institutional or policy documents may be used to
  define the prescribed future or supply consultation context; they are
  not themselves classified as consumer orientations.
* Future-making challenges (Convoluted Evaluations, Confrontational
  Negotiations, Competing Enactments) and Fragile Futures are qualitative,
  theoretical conditions that emerge from patterns across linked
  discourse, practices, actors, touchpoints, or time. They are never
  inferred from a single comment, and this application never assigns them
  an automatic score or probability.
* This application reproduces a Policymaking Roadmap (7 steps) and a
  Managerial Roadmap (6 steps) to support orientation-sensitive responses.
"""

# ─────────────────────────────────────────
# MODE LABELS
# ─────────────────────────────────────────
MODE_SINGLE = "single"
MODE_DOC = "document"
MODE_SINGLE_LABEL = "Analyze a Single Comment"
MODE_DOC_LABEL = "Map orientations across selected comments or uploaded file"

VALID_CONTEXT_TYPES = {
    "PARENT_REPLY", "THREAD_WINDOW", "ORIGINAL_POST", "CONSULTATION_PROMPT", "NONE"
}

CONTEXT_TYPE_CHOICES = [
    ("No context", "NONE",
     "Use when the focal comment must be interpreted on its own."),
    ("Parent comment", "PARENT_REPLY",
     "The specific comment to which the focal comment directly replies."),
    ("Thread window", "THREAD_WINDOW",
     "A small set of nearby comments from the same conversation. Use this "
     "when the exact parent comment is unavailable or when several nearby "
     "comments are needed to understand the exchange."),
    ("Original post", "ORIGINAL_POST",
     "The post, article caption, question, or topic that initiated the "
     "conversation."),
    ("Consultation prompt", "CONSULTATION_PROMPT",
     "The official question, proposal, or policy description to which a "
     "consultation response was submitted."),
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
# SYSTEM PROMPT -- classification logic
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
formulations: "not enough," "bigger picture," "does it have to be a
car?"

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
# ORIENTATION / ACTIVITY CONFIG (for UI + validation)
# ─────────────────────────────────────────
ORIENTATIONS = {
    "CATALYZER": {
        "color": "#27AE60", "bg": "#EAFAF1", "border": "#2ECC71",
        "goal": "Accelerate change toward the prescribed future",
        "narrative": "Urgency Narrative",
        "temporality": "The future is close -- change is happening now",
        "activities": "Simplify - Advocate - Accelerate",
    },
    "AMBIVALENT": {
        "color": "#D68910", "bg": "#FEFDE7", "border": "#F4D03F",
        "goal": "Slow down change (speed of change), delay decisions, balance risks and benefits",
        "narrative": "Pragmatic Narrative",
        "temporality": "The future is contingent -- change is uncertain",
        "activities": "Stall - Question - Delay",
    },
    "RESISTANT": {
        "color": "#C0392B", "bg": "#FDEDEC", "border": "#E74C3C",
        "goal": "Contest the prescribed future, protect the status quo",
        "narrative": "Control Narrative",
        "temporality": "The future is distant -- no change",
        "activities": "Avoid - Reject - Prevent",
    },
    "EXPANDER": {
        "color": "#7D3C98", "bg": "#F4ECF7", "border": "#9B59B6",
        "goal": "Expand the prescribed future, propose new pathways and alternative futures",
        "narrative": "Bigger Picture Narrative",
        "temporality": "The future is distant -- broader change",
        "activities": "Complexify - Contest - Reroute",
    }
}

ACTIVITY_META = {
    "EVALUATION":  {"color": "#2980B9", "bg": "#EBF5FB",
        "definition": "References to how consumers made sense of the prescribed future.",
        "subtypes": {"SIMPLIFY": "CATALYZER", "STALL": "AMBIVALENT", "AVOID": "RESISTANT", "COMPLEXIFY": "EXPANDER"}},
    "NEGOTIATION": {"color": "#E67E22", "bg": "#FEF9E7",
        "definition": "The activity through which consumers attempt to shape collective trajectories toward a preferred future.",
        "subtypes": {"ADVOCATE": "CATALYZER", "QUESTION": "AMBIVALENT", "REJECT": "RESISTANT", "CONTEST": "EXPANDER"}},
    "ENACTMENT":   {"color": "#8E44AD", "bg": "#F5EEF8",
        "definition": "References to how consumers gave form to futures through imagined, planned, or actual changes in everyday practices.",
        "subtypes": {"ACCELERATE": "CATALYZER", "DELAY": "AMBIVALENT", "PREVENT": "RESISTANT", "REROUTE": "EXPANDER"}},
}

# ─────────────────────────────────────────
# FUTURE-MAKING CHALLENGES -- corpus-level only. Never inferred from a
# single comment.
# ─────────────────────────────────────────
CHALLENGE_DEFINITIONS = {
    "CONVOLUTED_EVALUATIONS": {
        "label": "Convoluted Evaluations", "activity": "EVALUATION",
        "definition": (
            "Arise as consumers evaluate the prescribed future with more or "
            "less certainty and thoroughness. When consumers evaluate the "
            "prescribed future through divergent assumptions, evidence, and "
            "temporal horizons, evaluations become convoluted."
        )
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "label": "Confrontational Negotiations", "activity": "NEGOTIATION",
        "definition": (
            "Arise as consumers negotiate their preferred futures. When "
            "consumers simultaneously advocate for, question, reject, or "
            "contest the prescribed future in relation to others without "
            "conceding to alternatives, negotiations become confrontational."
        )
    },
    "COMPETING_ENACTMENTS": {
        "label": "Competing Enactments", "activity": "ENACTMENT",
        "definition": (
            "Arise as consumers enact different preferred futures through "
            "their current practices. When consumers accelerate, delay, "
            "prevent, or re-route the prescribed future through present "
            "practices, enactments become competitive."
        )
    },
}
FRAGILE_FUTURES_DEFINITION = (
    "Fragile Futures: multiple, volatile, and conflicting preferred futures "
    "that may interfere with the actualization of the prescribed one. "
    "Fragile Futures is a qualitative theoretical condition produced by "
    "differently oriented performances interfering with one another across "
    "evaluation, negotiation, and enactment -- it is NOT an automatic "
    "score, probability, or single-comment classification. Diagnosing it "
    "requires human interpretation of patterns across linked discourse, "
    "practices, actors, touchpoints, or time."
)

# ════════════════════════════════════════════════════════════════════
# ROADMAP CONTENT -- single source of truth for all roadmap wording used
# in the UI. Fixed reference content; never generated by the LLM.
# ════════════════════════════════════════════════════════════════════

EXPECTED_POLICY_STEP_TITLES = [
    "Determine the prescribed future",
    "Conduct social listening to map future-making orientations",
    "Diagnose and monitor key future-making challenges in evidence from social listening",
    "Implement support initiatives",
    "Facilitate enactment",
    "Measure multiple outcomes",
    "Revise intervention",
]

EXPECTED_MANAGER_STEP_TITLES = [
    "Determine the prescribed future",
    "Conduct social listening to consider future-making orientations",
    "Diagnose and monitor key future-making challenges in evidence from social listening",
    "Select orientation-sensitive strategy to support consumers",
    "Match messaging to key future-making challenges",
    "Support consumers through enactment",
]

POLICY_ROADMAP_STEPS = [
    ("1", EXPECTED_POLICY_STEP_TITLES[0],
     "Make explicit what future the intervention seeks to prescribe."),
    ("2", EXPECTED_POLICY_STEP_TITLES[1],
     "Identify how people adopting different orientations evaluate, "
     "negotiate, and enact (or not) the prescribed future. Consider "
     "consumer narratives, goals, emotions, and temporalities to "
     "identify orientations, rather than segments."),
    ("3", EXPECTED_POLICY_STEP_TITLES[2],
     "Identify which future-making challenges are the most pressing. "
     "Monitor where different performances of future-making interfere "
     "with one another."),
    ("4", EXPECTED_POLICY_STEP_TITLES[3],
     "Match support to the future-making orientations."),
    ("5", EXPECTED_POLICY_STEP_TITLES[4],
     "Provide infrastructure and build capabilities needed to navigate "
     "the change."),
    ("6", EXPECTED_POLICY_STEP_TITLES[5],
     "Is the intervention accurate and fair? Do consumers understand it? "
     "Who benefits? Who is excluded? Are alternative pathways emerging?"),
    ("7", EXPECTED_POLICY_STEP_TITLES[6],
     "Treat the prescribed future as revisable."),
]

MANAGER_ROADMAP_STEPS = [
    ("1", EXPECTED_MANAGER_STEP_TITLES[0],
     "Identify the future prescribed by the intervention."),
    ("2", EXPECTED_MANAGER_STEP_TITLES[1],
     "Identify how people adopting different orientations evaluate, "
     "negotiate, and enact (or not) the prescribed future. Consider "
     "consumer narratives, goals, emotions, and temporalities to "
     "identify orientations, rather than segments."),
    ("3", EXPECTED_MANAGER_STEP_TITLES[2],
     "Identify which future-making challenges are the most pressing. "
     "Monitor where different performances of future-making interfere "
     "with one another."),
    ("4", EXPECTED_MANAGER_STEP_TITLES[3],
     "Choose strategy based on the orientation and the key future-making "
     "challenges."),
    ("5", EXPECTED_MANAGER_STEP_TITLES[4],
     "Do not rely on a single persuasive frame. Universal claims (\"the "
     "change is inevitable,\" \"everyone benefits\") may mobilize "
     "consumers with catalyzer orientations while intensifying "
     "resistance and confrontation elsewhere."),
    ("6", EXPECTED_MANAGER_STEP_TITLES[5],
     "Place support at the touchpoints where consumers must adjust "
     "practices: onboarding, everyday workflows, escalation points, "
     "training, and appeals. Provide adjustable involvement, human "
     "assistance, and easy ways to pause, reverse, or modify adoption."),
]

# --- Step 2 content: orientation "lens" sentence is identical across
# both roadmaps; the monitoring bullets differ between the policy and
# managerial versions. ---
STEP2_ORIENTATION_LENS = {
    "CATALYZER": "This orientation sees the prescribed future as urgent, desirable, and already underway.",
    "AMBIVALENT": "This orientation sees the prescribed future as valuable, but conditions are not yet ready.",
    "RESISTANT": "This orientation sees the prescribed future as threatening their autonomy, identity, or rights.",
    "EXPANDER": "This orientation sees the prescribed future as framed too narrowly.",
}

POLICY_STEP2_SIGNALS = {
    "CATALYZER": [
        "Look for language emphasizing urgency and/or inevitability.",
        "Track voluntary early adoption.",
    ],
    "AMBIVALENT": [
        "Look for conditional language (\"I would, but\") and trials without conversion.",
        "Diagnose the specific unresolved condition.",
    ],
    "RESISTANT": [
        "Look for language about coercion, bans, loss of choice, and/or distrust.",
        "Track opt-outs, cancellations, organized opposition.",
    ],
    "EXPANDER": [
        "Look for claims that the intervention does not address the underlying problem or calls for broader system change.",
        "Track visions of broader change.",
    ],
}

MANAGER_STEP2_SIGNALS = {
    "CATALYZER": [
        "Monitor urgency and inevitability language, early pilot participation, and advocacy.",
        "Identify the resources enabling early adoption.",
    ],
    "AMBIVALENT": [
        "Monitor conditional language, such as \"I would, but…,\" \"not yet.\"",
        "Track hesitation signals.",
        "Identify trial without conversion or adoption.",
    ],
    "RESISTANT": [
        "Monitor coercion and surveillance language, opt-outs, and organized opposition.",
        "Distinguish ideological opposition from material disadvantage.",
    ],
    "EXPANDER": [
        "Watch for \"this does not solve the real problem,\" advocacy for collective alternatives, and participation in alternative infrastructures.",
    ],
}

# --- Step 3 content: diagnostic questions differ in wording between the
# policy and managerial roadmaps. ---
POLICY_STEP3_QUESTIONS = {
    "CONVOLUTED_EVALUATIONS": "Are incompatible evidence, assumptions, or temporal horizons preventing shared sensemaking?",
    "CONFRONTATIONAL_NEGOTIATIONS": "Is disagreement escalating around autonomy, fairness, legitimacy, or problem framing?",
    "COMPETING_ENACTMENTS": "Are accelerating, delaying, preventing, and re-routing practices creating incompatible pathways towards futures?",
}

MANAGER_STEP3_QUESTIONS = {
    "CONVOLUTED_EVALUATIONS": "Are consumers simplifying, stalling, avoiding, and/or complexifying the evaluation of the prescribed future?",
    "CONFRONTATIONAL_NEGOTIATIONS": "Are advocacy, questioning, rejection, and/or contestation escalating in conflict?",
    "COMPETING_ENACTMENTS": "Are consumers accelerating, delaying, preventing, and/or re-routing practice change through incompatible behaviours?",
}

# --- Step 4 content: the policy roadmap gives one Objective line per
# orientation; the managerial roadmap gives an Objective AND an Avoid
# line per orientation. These are NOT interchangeable and must not be
# merged. ---
POLICY_STEP4_OBJECTIVES = {
    "CATALYZER": "Enable responsible acceleration only where public value can be demonstrated.",
    "AMBIVALENT": "Convert uncertainty into confident decisions.",
    "RESISTANT": "Protect rights of consumers and ensure distribution of responsibility.",
    "EXPANDER": "Consider alternative futures; plan staged policies towards broader change.",
}

MANAGER_STEP4_OBJECTIVES = {
    "CATALYZER": {
        "objective": "Convert enthusiasm into credible and responsible experimentation.",
        "avoid": "Inevitability claims; treating early adopters as proof the transition is easy for everyone.",
    },
    "AMBIVALENT": {
        "objective": "Convert generalized uncertainty into specific, addressable conditions.",
        "avoid": "Pressure and artificial urgency; framing hesitation as ignorance or resistance.",
    },
    "RESISTANT": {
        "objective": "Protect rights of consumers and ensure distribution of responsibility.",
        "avoid": "\"There is no alternative\"; ridicule; hidden automation of decisions.",
    },
    "EXPANDER": {
        "objective": "Incorporate systemic critique and explore alternative futures.",
        "avoid": "Presenting the intervention as a complete solution; dismissing critiques.",
    },
}

# ─────────────────────────────────────────
# ORIENTATION-SENSITIVE GUIDANCE -- assembled from the Step 2/Step 4
# roadmap content above, supplemented only by short narrative sentences
# consistent with that content (never contradicting the roadmap steps).
# Orientations are treated as diagnostic patterns, not segments.
# ─────────────────────────────────────────
POLICY_ORIENTATION_GUIDANCE = {
    "CATALYZER": {
        "lens": STEP2_ORIENTATION_LENS["CATALYZER"],
        "implications": (
            "Consumers adopting a Catalyzer orientation may describe the "
            "prescribed future as desirable, inevitable, and already "
            "underway. Their experiences can reveal what currently "
            "facilitates enactment, but should not be treated as evidence "
            "that other consumers can or should follow the same pathway."
        ),
        "monitor": POLICY_STEP2_SIGNALS["CATALYZER"],
        "objective": POLICY_STEP4_OBJECTIVES["CATALYZER"],
    },
    "AMBIVALENT": {
        "lens": STEP2_ORIENTATION_LENS["AMBIVALENT"],
        "implications": (
            "Consumers adopting an Ambivalent orientation may consider the "
            "prescribed future potentially beneficial while regarding "
            "particular conditions as unresolved. Ambivalence can play a "
            "constructive role by surfacing weak points and demanding "
            "stronger infrastructure before large-scale adoption."
        ),
        "monitor": POLICY_STEP2_SIGNALS["AMBIVALENT"],
        "objective": POLICY_STEP4_OBJECTIVES["AMBIVALENT"],
    },
    "RESISTANT": {
        "lens": STEP2_ORIENTATION_LENS["RESISTANT"],
        "implications": (
            "Consumers adopting a Resistant orientation may frame the "
            "prescribed future as coercive, inequitable, or threatening to "
            "autonomy, identity, or established practices. Examining this "
            "orientation can provide valuable insight into consumers' "
            "reasons for pushback."
        ),
        "monitor": POLICY_STEP2_SIGNALS["RESISTANT"],
        "objective": POLICY_STEP4_OBJECTIVES["RESISTANT"],
    },
    "EXPANDER": {
        "lens": STEP2_ORIENTATION_LENS["EXPANDER"],
        "implications": (
            "Consumers adopting an Expander orientation may argue that the "
            "intervention addresses a narrowly framed problem while "
            "leaving broader problems unresolved, surfacing issues that "
            "may not have been considered when designing the intervention."
        ),
        "monitor": POLICY_STEP2_SIGNALS["EXPANDER"],
        "objective": POLICY_STEP4_OBJECTIVES["EXPANDER"],
    },
}

MANAGER_ORIENTATION_GUIDANCE = {
    "CATALYZER": {
        "lens": STEP2_ORIENTATION_LENS["CATALYZER"],
        "implications": (
            "Catalyzers simplify evaluation, advocate for the prescribed "
            "future, and accelerate enactment. Their enthusiasm can "
            "normalize new practices but may obscure the resources and "
            "competencies that supported early adoption."
        ),
        "monitor": MANAGER_STEP2_SIGNALS["CATALYZER"],
        "objective": MANAGER_STEP4_OBJECTIVES["CATALYZER"]["objective"],
        "avoid": MANAGER_STEP4_OBJECTIVES["CATALYZER"]["avoid"],
    },
    "AMBIVALENT": {
        "lens": STEP2_ORIENTATION_LENS["AMBIVALENT"],
        "implications": (
            "Ambivalent consumers see potential value but regard "
            "particular conditions as unresolved. Their hesitation can "
            "identify addressable barriers rather than generalized "
            "opposition."
        ),
        "monitor": MANAGER_STEP2_SIGNALS["AMBIVALENT"],
        "objective": MANAGER_STEP4_OBJECTIVES["AMBIVALENT"]["objective"],
        "avoid": MANAGER_STEP4_OBJECTIVES["AMBIVALENT"]["avoid"],
    },
    "RESISTANT": {
        "lens": STEP2_ORIENTATION_LENS["RESISTANT"],
        "implications": (
            "Resistant consumers perceive threats to autonomy, identity, "
            "rights, or established practices. They avoid evaluation, "
            "reject the prescribed future, and seek to prevent enactment."
        ),
        "monitor": MANAGER_STEP2_SIGNALS["RESISTANT"],
        "objective": MANAGER_STEP4_OBJECTIVES["RESISTANT"]["objective"],
        "avoid": MANAGER_STEP4_OBJECTIVES["RESISTANT"]["avoid"],
    },
    "EXPANDER": {
        "lens": STEP2_ORIENTATION_LENS["EXPANDER"],
        "implications": (
            "Expanders complexify evaluation, contest the prescribed "
            "future, and reroute enactment toward broader alternatives, "
            "revealing broader value propositions and alternative or "
            "complementary pathways."
        ),
        "monitor": MANAGER_STEP2_SIGNALS["EXPANDER"],
        "objective": MANAGER_STEP4_OBJECTIVES["EXPANDER"]["objective"],
        "avoid": MANAGER_STEP4_OBJECTIVES["EXPANDER"]["avoid"],
    },
}

# ─────────────────────────────────────────
# CHALLENGE-SENSITIVE GUIDANCE -- Step 3 diagnostic question plus a
# fuller action reference for each future-making challenge. Fixed;
# never LLM-generated.
# ─────────────────────────────────────────
CHALLENGE_POLICY_GUIDANCE = {
    "CONVOLUTED_EVALUATIONS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["CONVOLUTED_EVALUATIONS"],
        "action": (
            "Identify incompatible evidence, assumptions, or temporal "
            "horizons underlying divergent evaluations, and support "
            "shared sensemaking (e.g., a shared assessment covering "
            "accuracy, trade-offs, and distributional effects) rather "
            "than treating one evaluation as simply correct and others as "
            "misinformed."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["CONFRONTATIONAL_NEGOTIATIONS"],
        "action": (
            "Create legitimate and accessible opportunities for "
            "comparison, questioning, contestation, and negotiation among "
            "consumers. Reconsider the intervention's framing or support "
            "measures when conflict escalates, rather than relying on "
            "stronger one-way persuasion or presenting the prescribed "
            "future as inevitable."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["COMPETING_ENACTMENTS"],
        "action": (
            "Identify how consumers' current practices accelerate, delay, "
            "prevent, or re-route the prescribed future. Determine "
            "whether these pathways can coexist or interfere with one "
            "another, and coordinate support across viable pathways "
            "without treating every divergence from the prescribed "
            "practice as noncompliance."
        ),
    },
}

CHALLENGE_MANAGER_GUIDANCE = {
    "CONVOLUTED_EVALUATIONS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["CONVOLUTED_EVALUATIONS"],
        "action": (
            "Monitor how consumers assess the feasibility and desirability "
            "of the prescribed future at relevant touchpoints. Relevant "
            "indicators include contradictory claims, unresolved "
            "questions, repeated comparison, requests for evidence, "
            "abandoned onboarding, and gaps between organizational claims "
            "and consumer experience."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["CONFRONTATIONAL_NEGOTIATIONS"],
        "action": (
            "Monitor exchanges in which consumers defend preferred "
            "futures, respond to competing claims, or attempt to "
            "influence collective trajectories. Indicators include "
            "advocacy, persistent questioning, explicit rejection, "
            "systemic contestation, polarized discussion, and disputes "
            "over autonomy or legitimacy."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["COMPETING_ENACTMENTS"],
        "action": (
            "Monitor what consumers do in the present, including rapid "
            "adoption, trials without continuation, delayed use, "
            "refusals, continued reliance on existing practices, and "
            "movement toward alternative pathways. These present "
            "practices provide evidence of the preferred futures "
            "consumers are enacting."
        ),
    },
}

# Step 5 (managerial) messaging note.
MANAGER_MESSAGING_NOTE = (
    "Do not rely on a single persuasive frame. Universal claims (\"the "
    "change is inevitable,\" \"everyone benefits\") may mobilize consumers "
    "with catalyzer orientations while intensifying resistance and "
    "confrontation elsewhere."
)

# Supplementary cross-orientation interference check -- kept distinct
# from the Step 5 messaging note above.
CROSS_ORIENTATION_NOTE = (
    "Cross-orientation interference check: a response tailored to one "
    "orientation may intensify a challenge for another -- for example, an "
    "evidence campaign that reassures Ambivalent consumers may deepen "
    "Resistant distrust if it ignores autonomy, or reinforce Expander "
    "critique if it presents the intervention as a complete solution. "
    "Check for such spillover before finalizing a response."
)

# ─────────────────────────────────────────
# BENCHMARK EXAMPLES -- hidden from normal workflows; used only inside
# Advanced / Developer Tools for the Coding Consistency Check.
# ─────────────────────────────────────────
PF_EV = (
    "Transition all vehicles to Zero Emission Vehicles (EVs) to achieve Australia's "
    "net-zero emissions targets, as prescribed by Australia's National Electric "
    "Vehicle Strategy (2023)"
)

EXAMPLES = {
    "CONTROL | Evaluation -> Stall (Clara)": {
        "prescribed": PF_EV, "activity": "EVALUATION", "subtype": "STALL", "orientation": "AMBIVALENT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "Living in Outback Northwest Queensland there's no charging stations "
            "at the time. I did like the appeal of an electric vehicle mainly "
            "because you don't have to put fuel in it, which is great. But just "
            "at the time I went and bought a fairly decent car for five and a "
            "half grand. If we went on a driving holiday, we would take our big "
            "car. So if we got an EV it would just be our daily run around."
        )
    },
    "CONTROL | Evaluation -> Avoid (Esther)": {
        "prescribed": PF_EV, "activity": "EVALUATION", "subtype": "AVOID", "orientation": "RESISTANT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "Electric vehicles are not the solution. The current electricity "
            "infrastructure can't keep up with the demand now. I feel this is a "
            "lazy policy just appealing to city people and is just going to "
            "result in expensive car prices."
        )
    },
    "CONTROL | Evaluation -> Complexify (Peter)": {
        "prescribed": PF_EV, "activity": "EVALUATION", "subtype": "COMPLEXIFY", "orientation": "EXPANDER",
        "context": "", "context_type": "NONE", "is_consultation": False,
        "secondary_expected": ("EXPANDER", "ENACTMENT", "REROUTE"),
        "comment": (
            "The embodied carbon in a new vehicle is more than the emissions "
            "that are going to be produced by the current vehicle over the "
            "course of its lifetime until it falls apart. So that's the plan: "
            "to extract maximum value out of that current vehicle until it is "
            "no longer functional. I am at the moment on a waiting list for a "
            "new electric cargo bike because my current electric cargo bike is "
            "about seven years old."
        )
    },
    "CONTROL | Enactment -> Accelerate (Johnny)": {
        "prescribed": PF_EV, "activity": "ENACTMENT", "subtype": "ACCELERATE", "orientation": "CATALYZER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "Toyota is still very much trying to slow down the transition to EVs. "
            "We have ordered two Teslas that will be delivered hopefully this "
            "year. We are selling our Prado and it looks like we are going to "
            "sell our last Toyota car."
        )
    },
    "CONTROL | Enactment -> Prevent (StarT)": {
        "prescribed": PF_EV, "activity": "ENACTMENT", "subtype": "PREVENT", "orientation": "RESISTANT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": "I for one WILL NOT be forced into an elec vehicle and spend half my travel time charging the damn thing to go to hell."
    },
    "CONTROL | Enactment -> Reroute (Phillip)": {
        "prescribed": PF_EV, "activity": "ENACTMENT", "subtype": "REROUTE", "orientation": "EXPANDER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "I uprooted my life and moved from the Sunshine Coast to Melbourne "
            "with some of my strongest reasoning being the ability to use "
            "public transport, ride a bike around and use a car as little as "
            "possible."
        )
    },
    "NEGOTIATION | Advocate/Catalyzer (Joe)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "ADVOCATE", "orientation": "CATALYZER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "We are already so far behind! We need to sprint to catch up. We "
            "should be WORLD LEADERS in solar and battery manufacturing. Why are "
            "we not using our own minerals to make batteries for EVs on a global "
            "scale??"
        )
    },
    "NEGOTIATION | Advocate/Catalyzer (Forum User 1)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "ADVOCATE", "orientation": "CATALYZER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "EVs will be on an exponential adoption curve. Everyone will want "
            "one... Nobody will want an expensive 2nd hand ICE... Governments "
            "are going to start making fossil fuels very expensive. T-A-X-E-S "
            "will be levied on this foul, polluting rubbish we are all burning "
            "today... Or are you advocating that we go back to bicycles and "
            "horses, or maybe just buses?"
        )
    },
    "NEGOTIATION | Question/Ambivalent (Martin)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "QUESTION", "orientation": "AMBIVALENT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": "Have you thought about what they are gonna do with all the batteries once they expire because they aren't recyclable?"
    },
    "NEGOTIATION | Question/Ambivalent (Ned)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "QUESTION", "orientation": "AMBIVALENT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "To legislate in their favour is a further disadvantage to those "
            "already struggling. So where do we get the $50k to buy the cheapest "
            "new EV? It will not be possible for us to make the transition until "
            "a huge number of second hand EV's hit the market. And that won't "
            "happen until the governments, state and federal, change their "
            "entire fleets to EV's."
        )
    },
    "NEGOTIATION | Question/Ambivalent (Forum User 2)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "QUESTION", "orientation": "AMBIVALENT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "Better tell that to the Prius owners replacing their batteries. My "
            "car is now 13 years old... Batteries wear over time... so what "
            "magic bullet have you discovered that defies physics...? Once "
            "someone like me can get a used EV for <$10k, and have the battery "
            "replaced cheaply, then I'll agree with you... I'm not anti EV, I'm "
            "just realistic about costs and time frames."
        )
    },
    "NEGOTIATION | Reject/Resistant (Tom)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "No thanks, protest here we come. We get a say, this is our country "
            "not the governments'. I say freedom of choice, freedom to speak, "
            "some people don't even like electric cars."
        )
    },
    "NEGOTIATION | Reject/Resistant (Jocelyn)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": "The big green lie to cost taxpayers billions. Politicians forcing us to go this way need to be voted out."
    },
    "NEGOTIATION | Reject/Resistant (Raj)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": "We don't need politicians and their cronies telling us what sort of car we can have."
    },
    "NEGOTIATION | Reject/Resistant (Forum User 3)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "Nope, I'm not confused, thanks for the concern though... Not even "
            "close to the financial ruin you are trying to peddle... Technology "
            "adoption curves typically look like bell curves... not what you are "
            "suggesting... This is delusional."
        )
    },
    "NEGOTIATION | Contest/Expander (Bill)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "CONTEST", "orientation": "EXPANDER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "Cycling around Sydney is heavily unpleasant with the excessive "
            "emissions from commercial vehicles and buses. If you want modal "
            "shift and reduced emissions, it starts with reducing the impact of "
            "vehicles on pedestrians and cyclists. There's no reason we can't "
            "adopt tighter standards mandated elsewhere... Australia's "
            "regulation around vehicle emissions and efficiency is utterly "
            "laughable by global standards."
        )
    },
    "NEGOTIATION | Contest/Expander (Forum User 4)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "CONTEST", "orientation": "EXPANDER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "I fully get what you're saying, it's not rocket science, but "
            "that's not what I'm on about... I simply object to being told I'm "
            "an idiot... I'd like to see passenger cars filled with passengers, "
            "less cars on the road, less money spent on new roads!... Where "
            "people may simply drive less."
        )
    },
    "NEGOTIATION | Contest/Expander (Dan, declarative)": {
        "prescribed": PF_EV, "activity": "NEGOTIATION", "subtype": "CONTEST", "orientation": "EXPANDER",
        "context": "", "context_type": "NONE", "is_consultation": False, "secondary_expected": None,
        "comment": (
            "The future is less cars, in higher density pedestrian/bike and "
            "train orientated urban environments, where cars are a secondary "
            "transport really only for those who really need it."
        )
    },
}

# ─────────────────────────────────────────
# CONSISTENCY SAFEGUARD
# ─────────────────────────────────────────

def get_secondary_classifications(result: dict) -> list:
    sec = result.get("secondary_classifications")
    if not sec:
        return []
    cleaned = []
    for item in sec:
        if isinstance(item, dict):
            cleaned.append({
                "activity": _clean_enum(item.get("activity", "")).upper(),
                "activity_subtype": _clean_enum(item.get("activity_subtype", "")).upper(),
                "orientation": _clean_enum(item.get("orientation", "")).upper(),
                "rationale": item.get("rationale", "")
            })
    return cleaned


def _fix_pairing(orientation: str, activity: str, subtype: str):
    subtype_map = ACTIVITY_META.get(activity, {}).get("subtypes", {})
    if not subtype_map:
        return subtype, None
    expected_orientation = subtype_map.get(subtype)
    if expected_orientation and expected_orientation != orientation:
        corrected = next((st for st, ori in subtype_map.items() if ori == orientation), None)
        if corrected:
            return corrected, f"subtype adjusted from {subtype} to {corrected} to match orientation {orientation}"
    return subtype, None


def enforce_consistency(result: dict) -> dict:
    notes = []
    main_orientation = _clean_enum(result.get("main_orientation", "")).upper()
    main_activity = _clean_enum(result.get("main_activity", "")).upper()
    main_subtype = _clean_enum(result.get("activity_subtype", "")).upper()
    fixed_subtype, note = _fix_pairing(main_orientation, main_activity, main_subtype)
    result["activity_subtype"] = fixed_subtype
    if note:
        notes.append(f"Primary: {note}.")

    secondary = get_secondary_classifications(result)
    fixed_secondary = []
    for i, sec in enumerate(secondary):
        ori, act, sub = sec.get("orientation", ""), sec.get("activity", ""), sec.get("activity_subtype", "")
        if act and sub and ori:
            fixed_sub, note2 = _fix_pairing(ori, act, sub)
            if note2:
                notes.append(f"Secondary #{i+1}: {note2}.")
            sec["activity_subtype"] = fixed_sub
        fixed_secondary.append(sec)
    result["secondary_classifications"] = fixed_secondary

    if notes:
        result["_consistency_note"] = " ".join(notes)
    return result


def enforce_context_metadata(result: dict, context_available: bool, context_type: str) -> dict:
    result["context_used"] = bool(context_available)
    result["context_type"] = context_type if context_type in VALID_CONTEXT_TYPES else "NONE"
    if not context_available:
        result["context_note"] = result.get("context_note", "") if result.get("context_note") else ""
    return result


# ─────────────────────────────────────────
# CORE FUNCTION -- focal comment + context
# ─────────────────────────────────────────

def analyze_comment(prescribed_future: str, focal_text: str, context_text: str = "",
                     context_type: str = "NONE", is_consultation: bool = False,
                     api_key: str = None, allow_secondary_classifications: bool = True) -> dict:
    client = get_openai_client()

    if is_consultation:
        focal_label = "FOCAL RESPONSE TO CLASSIFY"
        context_label = "CONSULTATION/POLICY CONTEXT -- USE FOR INTERPRETATION BUT DO NOT CLASSIFY"
    else:
        focal_label = "FOCAL COMMENT TO CLASSIFY"
        context_label = "CONTEXT (parent comment / nearby comments / original post) -- USE FOR INTERPRETATION BUT DO NOT CLASSIFY"

    context_available = bool(context_text and context_text.strip())
    context_block = context_text.strip() if context_available else "No context is available for this comment."

    secondary_instruction = (
        "You MAY return up to two secondary classifications if substantively "
        "and separably supported, per Section G."
        if allow_secondary_classifications else
        "Do NOT return any secondary classifications for this request. "
        "secondary_classifications must be an empty list."
    )

    user_message = f"""
PRESCRIBED FUTURE:
{prescribed_future}

{focal_label}:
{focal_text}

{context_label}:
{context_block}

SECONDARY CLASSIFICATIONS FOR THIS REQUEST: {secondary_instruction}

Apply the Decision Procedure in Section E, in order. FIRST ask whether the
focal comment's overarching function is to Advocate, Question, Reject, or
Contest a collective trajectory -- do this BEFORE considering how much of
the comment reads as technical, factual, or evaluative in isolation.
Verify the activity_subtype belongs to the valid pairing row for its own
orientation (Section D) before responding. Do not produce any
future-making challenge, Fragile Futures assessment, or policy/managerial
recommendation -- these are outside your task.
"""
    with openai_semaphore:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    parsed = json.loads(response.choices[0].message.content)
    parsed = enforce_consistency(parsed)
    if not allow_secondary_classifications:
        parsed["secondary_classifications"] = []
    parsed = enforce_context_metadata(parsed, context_available, context_type)
    return parsed

def run_consistency_suite(api_key: str) -> dict:
    """Internal consistency check against illustrative benchmark examples.
    Does NOT constitute empirical validation, intercoder reliability, or
    evidence of generalizability."""
    results = []
    for name, ex in EXAMPLES.items():
        try:
            pred = analyze_comment(
                ex["prescribed"], ex["comment"], ex.get("context", ""),
                ex.get("context_type", "NONE"), ex.get("is_consultation", False),
                api_key, allow_secondary_classifications=True
            )
        except Exception as e:
            results.append({
                "example": name, "error": str(e),
                "expected": (ex["orientation"], ex["activity"], ex["subtype"]),
                "predicted": (None, None, None), "match": False,
                "activity_match": False, "subtype_match": False, "orientation_match": False,
                "secondary_expected": ex.get("secondary_expected"), "secondary_match": None
            })
            continue
        pred_orientation = _clean_enum(pred.get("main_orientation", "")).upper()
        pred_activity    = _clean_enum(pred.get("main_activity", "")).upper()
        pred_subtype     = _clean_enum(pred.get("activity_subtype", "")).upper()

        activity_match = (pred_activity == ex["activity"])
        subtype_match = (pred_subtype == ex["subtype"])
        orientation_match = (pred_orientation == ex["orientation"])
        match = activity_match and subtype_match and orientation_match

        secondary_match = None
        sec_expected = ex.get("secondary_expected")
        if sec_expected:
            secondary_list = get_secondary_classifications(pred)
            secondary_match = any(
                sec.get("orientation") == sec_expected[0] and sec.get("activity") == sec_expected[1]
                and sec.get("activity_subtype") == sec_expected[2] for sec in secondary_list
            )

        results.append({
            "example": name,
            "expected": (ex["orientation"], ex["activity"], ex["subtype"]),
            "predicted": (pred_orientation, pred_activity, pred_subtype),
            "match": match,
            "activity_match": activity_match, "subtype_match": subtype_match,
            "orientation_match": orientation_match,
            "secondary_expected": sec_expected, "secondary_match": secondary_match
        })

    if not results:
        return {"results": [], "overall_agreement": 0.0}

    n = len(results)
    return {
        "results": results,
        "overall_agreement": sum(r["match"] for r in results) / n,
        "overall_activity_agreement": sum(r["activity_match"] for r in results) / n,
        "overall_subtype_agreement": sum(r["subtype_match"] for r in results) / n,
        "overall_orientation_agreement": sum(r["orientation_match"] for r in results) / n,
    }


# ─────────────────────────────────────────
# FRAMEWORK VALIDATION -- structural tests. Pure code-level checks, no
# API calls.
# ─────────────────────────────────────────

REQUIRED_ORIENTATIONS = {"CATALYZER", "AMBIVALENT", "RESISTANT", "EXPANDER"}
REQUIRED_SUBTYPES = {
    "EVALUATION": {"SIMPLIFY", "STALL", "AVOID", "COMPLEXIFY"},
    "NEGOTIATION": {"ADVOCATE", "QUESTION", "REJECT", "CONTEST"},
    "ENACTMENT": {"ACCELERATE", "DELAY", "PREVENT", "REROUTE"},
}
REQUIRED_CHALLENGES = {"CONVOLUTED_EVALUATIONS", "CONFRONTATIONAL_NEGOTIATIONS", "COMPETING_ENACTMENTS"}

# Terms that must NEVER appear anywhere in this module's guidance /
# roadmap content, because they are not part of this framework.

EXPECTED_OUTPUT_SCHEMA_KEYS = [
    "prescribed_future_acknowledged", "main_activity", "activity_subtype",
    "activity_rationale", "main_orientation", "orientation_rationale",
    "secondary_classifications", "narrative_identified", "emotions_identified",
    "temporality_identified", "notable_conditions_of_adoption",
    "supporting_text", "context_note", "input_scope_warning",
]


def run_framework_validation() -> list:
    """Returns a list of (test_name, passed: bool, detail: str) tuples.
    Pure code-level structural checks -- no API calls."""
    tests = []

    tests.append((
        "Exactly 4 future-making orientations",
        set(ORIENTATIONS.keys()) == REQUIRED_ORIENTATIONS,
        f"Found: {sorted(ORIENTATIONS.keys())}"
    ))

    for act, expected_subs in REQUIRED_SUBTYPES.items():
        actual_subs = set(ACTIVITY_META.get(act, {}).get("subtypes", {}).keys())
        tests.append((
            f"{act} has exactly the 4 expected performances",
            actual_subs == expected_subs,
            f"Found: {sorted(actual_subs)}"
        ))

    tests.append((
        "Policy roadmap has exactly 7 recursive steps",
        len(POLICY_ROADMAP_STEPS) == 7,
        f"Found: {len(POLICY_ROADMAP_STEPS)}"
    ))
    tests.append((
        "Managerial roadmap has exactly 6 recursive steps",
        len(MANAGER_ROADMAP_STEPS) == 6,
        f"Found: {len(MANAGER_ROADMAP_STEPS)}"
    ))

    tests.append((
        "Policy roadmap step titles match the expected sequence, in order",
        [t for _, t, _ in POLICY_ROADMAP_STEPS] == EXPECTED_POLICY_STEP_TITLES,
        f"Found: {[t for _, t, _ in POLICY_ROADMAP_STEPS]}"
    ))
    tests.append((
        "Managerial roadmap step titles match the expected sequence, in order",
        [t for _, t, _ in MANAGER_ROADMAP_STEPS] == EXPECTED_MANAGER_STEP_TITLES,
        f"Found: {[t for _, t, _ in MANAGER_ROADMAP_STEPS]}"
    ))

    tests.append((
        "Exactly 3 future-making challenges, correctly named",
        set(CHALLENGE_DEFINITIONS.keys()) == REQUIRED_CHALLENGES,
        f"Found: {sorted(CHALLENGE_DEFINITIONS.keys())}"
    ))

    policy_challenge_coverage = all(
        k in CHALLENGE_POLICY_GUIDANCE
        and "diagnostic_question" in CHALLENGE_POLICY_GUIDANCE[k]
        and "action" in CHALLENGE_POLICY_GUIDANCE[k]
        for k in REQUIRED_CHALLENGES
    )
    manager_challenge_coverage = all(
        k in CHALLENGE_MANAGER_GUIDANCE
        and "diagnostic_question" in CHALLENGE_MANAGER_GUIDANCE[k]
        and "action" in CHALLENGE_MANAGER_GUIDANCE[k]
        for k in REQUIRED_CHALLENGES
    )
    tests.append((
        "Complete challenge-reference coverage (policy: question + action)",
        policy_challenge_coverage,
        f"Missing/incomplete: {[k for k in REQUIRED_CHALLENGES if k not in CHALLENGE_POLICY_GUIDANCE]}"
    ))
    tests.append((
        "Complete challenge-reference coverage (managerial: question + action)",
        manager_challenge_coverage,
        f"Missing/incomplete: {[k for k in REQUIRED_CHALLENGES if k not in CHALLENGE_MANAGER_GUIDANCE]}"
    ))

    questions_correctly_distinguished = all(
        POLICY_STEP3_QUESTIONS[k] != MANAGER_STEP3_QUESTIONS[k] for k in REQUIRED_CHALLENGES
    )
    tests.append((
        "Policy and managerial Step-3 diagnostic questions use distinct wording (not merged)",
        questions_correctly_distinguished,
        "Some questions were found identical across roadmaps" if not questions_correctly_distinguished else "Confirmed distinct"
    ))

    orientation_policy_coverage = all(o in POLICY_ORIENTATION_GUIDANCE for o in REQUIRED_ORIENTATIONS)
    orientation_manager_coverage = all(o in MANAGER_ORIENTATION_GUIDANCE for o in REQUIRED_ORIENTATIONS)
    manager_has_avoid = all("avoid" in MANAGER_ORIENTATION_GUIDANCE[o] for o in REQUIRED_ORIENTATIONS)
    policy_has_no_avoid_field = all("avoid" not in POLICY_ORIENTATION_GUIDANCE[o] for o in REQUIRED_ORIENTATIONS)
    tests.append((
        "Complete orientation-reference coverage (policy)",
        orientation_policy_coverage,
        f"Missing: {[o for o in REQUIRED_ORIENTATIONS if o not in POLICY_ORIENTATION_GUIDANCE]}"
    ))
    tests.append((
        "Complete orientation-reference coverage (managerial)",
        orientation_manager_coverage,
        f"Missing: {[o for o in REQUIRED_ORIENTATIONS if o not in MANAGER_ORIENTATION_GUIDANCE]}"
    ))
    tests.append((
        "Managerial Step 4 guidance includes 'avoid' content (managerial roadmap only)",
        manager_has_avoid,
        "Confirmed" if manager_has_avoid else "Missing 'avoid' field for one or more orientations"
    ))
    tests.append((
        "Policy Step 4 guidance does not import managerial-only 'avoid' content (policy roadmap has none)",
        policy_has_no_avoid_field,
        "Confirmed" if policy_has_no_avoid_field else "Policy guidance incorrectly contains an 'avoid' field"
    ))

    return tests


# ─────────────────────────────────────────
# COMMENT / THREAD DATA STRUCTURES (technical infrastructure)
# ─────────────────────────────────────────

def has_comment_text_column(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip() for c in df.columns}
    return bool(cols & {"comment_text", "text", "comment"})


def strip_trailing_orientation_label(text: str) -> str:
    cleaned = re.sub(
        r'\(\s*(Catalyzer|Ambivalent|Resistant|Expander)\s+Orientation\s*\)\.?\s*$',
        '', text, flags=re.IGNORECASE
    ).strip()
    return cleaned


USER_LABEL_PATTERN = re.compile(r'(?im)^\s*User\s*\d+\s*:\s*')


def detect_user_labeled_exchange(text: str) -> int:
    return len(USER_LABEL_PATTERN.findall(text or ""))


def build_comment_records_from_user_labels(text: str) -> list:
    matches = list(USER_LABEL_PATTERN.finditer(text or ""))
    if len(matches) < 2:
        return []
    thread_id = "_labeled_exchange_1"
    records = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[start:end].strip()
        cleaned = strip_trailing_orientation_label(raw)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned:
            continue
        records.append({
            "comment_id": f"user{i+1}", "thread_id": thread_id, "parent_comment_id": None,
            "author": f"User {i+1}", "timestamp": "", "comment_text": cleaned, "original_index": i,
        })
    return records


def build_comment_records_from_paragraphs(text: str, separator: str = None) -> list:
    text = (text or "").strip()
    if not text:
        return []
    if separator:
        raw_items = [x.strip() for x in text.split(separator) if x.strip()]
    else:
        raw_items = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    records = []
    for i, item in enumerate(raw_items):
        item = strip_trailing_orientation_label(item)
        if len(item.split()) < 2:
            continue
        records.append({
            "comment_id": f"c{i}", "thread_id": DEFAULT_THREAD, "parent_comment_id": None,
            "author": "", "timestamp": "", "comment_text": item, "original_index": i,
        })
    return records


def build_comment_records_from_csv(df: pd.DataFrame) -> list:
    cols = {c.lower().strip(): c for c in df.columns}
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
        text = str(row.get(text_col, "")).strip() if pd.notna(row.get(text_col)) else ""
        if not text:
            continue
        comment_id = safe_get(row, "comment_id", f"c{i}")
        thread_id = safe_get(row, "thread_id", DEFAULT_THREAD) or DEFAULT_THREAD
        parent_id = safe_get(row, "parent_comment_id", "") or None
        author = safe_get(row, "author", "")
        timestamp = safe_get(row, "timestamp", "")
        records.append({
            "comment_id": comment_id, "thread_id": thread_id, "parent_comment_id": parent_id,
            "author": author, "timestamp": timestamp, "comment_text": text, "original_index": i,
        })
    return records


def extract_public_consultation_responses(text: str, min_words: int = 4) -> list:
    text = re.sub(r'\s+', ' ', text.strip())
    matches = list(re.finditer(r'\b(\d{6,7})\s+(?:Name\s+withheld|[A-Z][a-z]+)', text))
    responses = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        resp_id = m.group(1)
        block = re.sub(r'^\d{6,7}\s+(?:Name\s+withheld|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*', '', block)
        block = re.sub(r'Option\s+[ABC]\s*-\s*\w+,?\s*', '', block, flags=re.IGNORECASE)
        block = re.sub(r'\b(Yes|No|NULL)\s*$', '', block, flags=re.IGNORECASE).strip()
        block = re.sub(r'\s{2,}', ' ', block).strip(' ,.-')
        if not block or block.upper() == "NULL":
            continue
        if len(block.split()) >= min_words:
            responses.append({"id": resp_id, "text": block})
    return responses


def build_comment_records_from_consultation(text: str) -> list:
    raw = extract_public_consultation_responses(text)
    records = []
    for i, item in enumerate(raw):
        records.append({
            "comment_id": f"resp_{item['id']}", "thread_id": DEFAULT_THREAD, "parent_comment_id": None,
            "author": "", "timestamp": "", "comment_text": item["text"], "original_index": i,
        })
    return records


def index_threads(records: list):
    by_id = {r["comment_id"]: r for r in records}
    thread_order = {}
    for r in records:
        thread_order.setdefault(r["thread_id"], []).append(r["comment_id"])
    return by_id, thread_order


def build_context(record: dict, by_id: dict, thread_order: dict,
                   consultation_prompt: str = None, is_consultation: bool = False):
    parts = []
    context_type = "NONE"
    focal_id = record["comment_id"]
    thread_id = record["thread_id"]
    order = thread_order.get(thread_id, [focal_id])
    idx_in_thread = order.index(focal_id) if focal_id in order else 0

    parent_id = record.get("parent_comment_id")
    parent_record = by_id.get(parent_id) if parent_id else None
    if parent_record and parent_record.get("thread_id") != thread_id:
        parent_record = None

    if parent_record:
        parts.append(f"PARENT COMMENT:\n{parent_record['comment_text']}")
        context_type = "PARENT_REPLY"

    preceding_ids = order[max(0, idx_in_thread - 2):idx_in_thread]
    preceding_ids = [pid for pid in preceding_ids if pid != parent_id]
    for pid in preceding_ids:
        parts.append(f"PRECEDING COMMENT:\n{by_id[pid]['comment_text']}")
        if context_type == "NONE":
            context_type = "THREAD_WINDOW"

    following_record = None
    for cid in order:
        cand = by_id.get(cid)
        if cand and cand.get("parent_comment_id") == focal_id:
            following_record = cand
            break
    if following_record:
        parts.append(f"FOLLOWING REPLY:\n{following_record['comment_text']}")
        if context_type == "NONE":
            context_type = "THREAD_WINDOW"

    if order:
        root_record = by_id.get(order[0])
        if (root_record and root_record["comment_id"] not in (focal_id, parent_id)
                and not root_record.get("parent_comment_id")):
            parts.append(f"ORIGINAL POST:\n{root_record['comment_text']}")
            if context_type == "NONE":
                context_type = "ORIGINAL_POST"

    if is_consultation and consultation_prompt and consultation_prompt.strip():
        parts.append(f"CONSULTATION QUESTION / POLICY CONTEXT:\n{consultation_prompt.strip()}")
        if context_type == "NONE":
            context_type = "CONSULTATION_PROMPT"

    context_text = "\n\n".join(parts)
    return context_text, context_type, bool(parts)


def compute_evenly_spaced_sample_indices(total: int, k: int) -> list:
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    seen = set()
    for i in range(k):
        idx = max(0, min(total - 1, int(round(i * step))))
        probe, forward = idx, True
        while probe in seen:
            probe = probe + 1 if forward else probe - 1
            if probe >= total:
                probe, forward = idx, False
                continue
            if probe < 0:
                break
        seen.add(max(0, min(total - 1, probe)))
    return sorted(seen)


def analyze_document(prepared_records: list, prescribed_future: str, api_key: str, progress_bar=None) -> list:
    total = len(prepared_records)
    results = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOC_MAX_WORKERS) as executor:
        future_to_pos = {
            executor.submit(
                analyze_comment, prescribed_future, rec["comment_text"], rec["context_text"],
                rec["context_type"], rec["is_consultation"], api_key,
                True  # allow_secondary_classifications -- permitted for corpus mapping
            ): pos
            for pos, rec in enumerate(prepared_records)
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_pos):
            pos = future_to_pos[future]
            rec = prepared_records[pos]
            try:
                r = future.result()
            except Exception as e:
                r = {"_error": str(e)}
            r["_comment_id"] = rec["comment_id"]
            r["_thread_id"] = rec["thread_id"]
            r["_parent_comment_id"] = rec["parent_comment_id"]
            r["_comment_text"] = rec["comment_text"]
            r["_original_index"] = rec["original_index"]
            results[pos] = r
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(completed / total, text=f"Analyzed {completed}/{total} comments...")
    return sorted([r for r in results if r is not None], key=lambda r: r.get("_original_index", 0))


# ─────────────────────────────────────────
# CORPUS-LEVEL AGGREGATION -- challenge review requires linked/multi-
# comment patterns and explicit human interpretation; never automatic.
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


def compute_challenge_review_candidates(results: list) -> dict:
    """Flags LINKED comment pairs performing the SAME activity with
    DIFFERENT orientation-specific performances, as candidates requiring
    human interpretive review for the corresponding future-making
    challenge. Never auto-diagnoses that the challenge occurred, never
    scored, never converted into a percentage or a Fragile Futures
    assessment."""
    valid = [r for r in results if r and "_error" not in r]
    by_id = {r.get("_comment_id"): r for r in valid if r.get("_comment_id")}
    candidates = {k: [] for k in CHALLENGE_DEFINITIONS}

    for r in valid:
        parent_id = r.get("_parent_comment_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        act_r = _clean_enum(r.get("main_activity", "")).upper()
        act_p = _clean_enum(parent.get("main_activity", "")).upper()
        if act_r != act_p:
            continue
        ori_r = _clean_enum(r.get("main_orientation", "")).upper()
        ori_p = _clean_enum(parent.get("main_orientation", "")).upper()
        sub_r = _clean_enum(r.get("activity_subtype", "")).upper()
        sub_p = _clean_enum(parent.get("activity_subtype", "")).upper()
        if ori_r == ori_p and sub_r == sub_p:
            continue
        challenge_key = next((k for k, v in CHALLENGE_DEFINITIONS.items() if v["activity"] == act_r), None)
        if challenge_key:
            candidates[challenge_key].append({"parent": parent, "reply": r})

    return candidates


def compute_same_thread_diversity(results: list) -> dict:
    """For comments sharing a thread_id without explicit parent-reply
    links, summarizes the diversity of performances present. Descriptive
    only -- does not claim an observed interaction or a diagnosed
    challenge."""
    valid = [r for r in results if r and "_error" not in r]
    by_thread = {}
    for r in valid:
        tid = r.get("_thread_id", DEFAULT_THREAD)
        by_thread.setdefault(tid, []).append(r)
    summary = {}
    for tid, comments in by_thread.items():
        if len(comments) < 2:
            continue
        acts = {}
        for c in comments:
            act = _clean_enum(c.get("main_activity", "")).upper()
            sub = _clean_enum(c.get("activity_subtype", "")).upper()
            ori = _clean_enum(c.get("main_orientation", "")).upper()
            key = (act, sub, ori)
            acts[key] = acts.get(key, 0) + 1
        summary[tid] = {"comments": comments, "performance_counts": acts}
    return summary
