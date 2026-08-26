import streamlit as st
import openai
import json
import re
import concurrent.futures
import pandas as pd

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Future-Making Analyzer",
    page_icon="FM",
    layout="wide"
)

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
    "Interpretive-use note: the paper defines future-making activities as "
    "interdependent, entangled, and recursive rather than sequential, and "
    "orientations as situated ways of performing future-making rather than "
    "fixed consumer types or stable market segments. This application is an "
    "interactive demonstration tool, not a validated diagnostic instrument. "
    "Review outputs alongside the complete comment, any available context, "
    "and relevant behavioral evidence."
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
* Applying the framework to AI-integrated healthcare is illustrative of
  expected transferability; it is **not** independent empirical validation.
* Intervention type -- Fixed, Bounded, Flexible, or Open -- depends on the
  specific scope and prescriptiveness of the intervention being analyzed,
  not on its general domain.
* Orientations -- Catalyzer, Ambivalent, Resistant, and Expander -- are
  situated ways of performing future-making, **not** fixed consumer types
  or market segments.
* Results require interpretation alongside context and relevant
  behavioral evidence. Institutional or policy documents may be used to
  define the prescribed future or supply consultation context; they are
  not themselves classified as consumer orientations.
"""

# ─────────────────────────────────────────
# MODE LABELS
# ─────────────────────────────────────────
MODE_SINGLE = "single"
MODE_DOC = "document"
MODE_SINGLE_LABEL = "Analyze a Single Comment"
MODE_DOC_LABEL = "Map Orientations Across Selected Comments"

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
# SYSTEM PROMPT -- corrected functional classification rule for Negotiation
# ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are an analytical assistant supporting the mapping of consumer
future-making, grounded exclusively in the practice-theoretical framework
developed in "Futures in the Making: How Consumers Respond to
Future-Oriented Interventions," based on qualitative research on
Australian Zero Emission Vehicle (ZEV) interventions.

Use ONLY the concepts, definitions, and categories described below. Do not
introduce interaction-analysis taxonomies, discourse-analysis constructs,
sentiment-analysis categories, or any other framework not defined here.

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

  FIXED (Narrow scope, Highly prescriptive)   -- e.g., ban on single-use plastic bags
  BOUNDED (Broad scope, Highly prescriptive)  -- e.g., ZEV policies and strategies
  FLEXIBLE (Narrow scope, Lowly prescriptive) -- e.g., Meat-free Mondays
  OPEN (Broad scope, Lowly prescriptive)      -- e.g., decentralized adoption of AI in healthcare

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
Consumers' cognitive assessment of the prescribed future -- its meaning,
likelihood, desirability, benefits, costs, risks, assumptions, and
trade-offs. Classify a comment as Evaluation when its primary function is
this cognitive assessment WITHOUT primarily attempting to shape a
collective trajectory through Advocate, Question, Reject, or Contest.

--- NEGOTIATION ---
"Negotiation refers to the activity through which consumers attempt to
shape collective trajectories toward a preferred future."

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
punctuation mark. It can be performed through declarative statements,
rhetorical questions, conditional statements, technical/economic/
practical/equity arguments, or descriptions of personal experience used
to support a collective claim.

MOST IMPORTANT INSTRUCTION IN THIS PROMPT -- FUNCTIONAL DOMINANCE OVER
VOLUME: When assessments, predictions, technical claims, cost figures, or
personal experience are MOBILIZED to advocate, question, reject, or
contest a collective trajectory, NEGOTIATION IS PRIMARY, even when most
of the comment's sentences read as factual, technical, evaluative, or
predictive in isolation. Do NOT count sentences or estimate the
proportion of evaluative-sounding content to decide the primary activity.
Instead, ask what the COMPLETE comment is functionally DOING as a whole.
Supporting evaluative content does not make Evaluation primary when that
content is being used in service of Advocate, Question, Reject, or
Contest.

Example of the error to avoid: a comment that spends most of its length
presenting technical predictions about EV adoption curves, then closes by
challenging a specific alternative ("Or are you advocating that we go
back to bicycles and horses?"), is NOT Evaluation with a small
Negotiation remainder. Its entire content is organized around defending
and normalizing the prescribed future against an implied alternative --
the technical predictions are RHETORICAL AMMUNITION for that defense, not
a self-standing assessment. This comment is NEGOTIATION / ADVOCATE /
CATALYZER in full.

SAFEGUARDS -- these signals ALONE do not establish Negotiation (they are
insufficient by themselves, but may co-occur with genuine functional
Negotiation):
  - A question mark alone.
  - Strong or negative language alone.
  - Negative sentiment alone.
  - Technical complexity alone.
  - Mentioning government or policy alone.
  - Imagining or describing a future alone.
  - A personal action or first-person statement alone.
The deciding issue is always whether the comment's OVERARCHING FUNCTION
is to shape a collective trajectory through Advocate, Question, Reject,
or Contest. If a comment ONLY assesses the prescribed future without
performing one of these functions, classify it as Evaluation.

Sub-types (organized by orientation):
  ADVOCATE  (Catalyzer)  -- recruiting others; presenting the transition
    as a collective endeavor; calling for stronger policy signals; calling
    for urgent collective action; normalizing or defending rapid movement
    toward the prescribed future; challenging actors who are slowing or
    opposing the transition.
  QUESTION  (Ambivalent) -- seeking reassurance or proof that the
    transition is technically feasible, affordable, fair, and adequately
    supported; direct or rhetorical questions; requests for evidence;
    challenges to optimistic claims; conditional willingness to join;
    specifying what other actors must do first; technical or equity
    concerns used to test whether collective enactment is credible;
    polite skepticism. A declarative account (no question mark required)
    performs Question when it establishes conditions other actors must
    satisfy before the consumer can join the prescribed future.
  REJECT    (Resistant)  -- refusing the prescribed future; rejecting
    mandates or government authority; framing the intervention as
    coercive; defending freedom, autonomy, identity, or choice; calling
    for protest or political opposition; seeking to prevent collective
    enactment; directly disputing another actor's claim; rejecting the
    legitimacy of those promoting the intervention.
  CONTEST   (Expander)   -- arguing the intervention defines the problem
    too narrowly; challenging others' engagement with the prescribed
    future; advancing a broader collective trajectory; connecting the
    prescribed future to wider systems and practices; calling for
    systemic, infrastructural, regulatory, or practice-level change;
    proposing preferred futures beyond the intervention's existing scope.

--- ENACTMENT ---
What consumers do in the present to materialize a preferred future
through actual, planned, delayed, refused, or rerouted practices. Do NOT
classify a comment as Enactment merely because it contains "I bought,"
"I drive," "my car," "I will," a description of current practice, or a
conditional intention. Determine how the behavioral statement FUNCTIONS
in the complete comment. If personal behavior is primarily used as
EVIDENCE to advocate, question, reject, or contest a collective
trajectory, NEGOTIATION remains the dominant activity, even though the
comment mentions personal practice. Use Enactment as dominant only when
the comment's main function is to describe how the consumer is materially
giving form to a preferred future -- not when the behavioral description
is instrumental to a negotiating move.

Sub-types: ACCELERATE (Catalyzer), DELAY (Ambivalent), PREVENT
(Resistant), REROUTE (Expander) -- as previously defined: materializing
the prescribed future through present decisions; continuing current
practice pending maturing conditions; entrenching current practice while
explicitly refusing adoption; directing practice toward a different,
broader preferred future.

====================================================================
C. FUTURE-MAKING ORIENTATIONS (Table 2 of the paper)
====================================================================

--- CATALYZER --- Urgency narrative: the future is now, and the
prescribed transition is the rightfully determined future. Goal:
accelerate change. Emotions: utopian optimism, enthusiasm, confidence,
pride. Temporality: the future is close -- change is happening now.

--- AMBIVALENT --- Pragmatic narrative: arguments around the
practicalities of the prescribed future. Goal: slow down change, delay
decisions, balance risks and benefits. Emotions: curiosity, caution,
anxiety, frustration, optimism. Temporality: the future is contingent --
change is uncertain.

--- RESISTANT --- Control narrative: interventions framed as coercive
and inequitable. Goal: contest the prescribed future, protect the status
quo. Emotions: pessimism, anger, anxiety, fear. Temporality: the future
is distant -- there will be no change.

--- EXPANDER --- Bigger picture narrative: "zooms out" from the
intervention to alternative systems, practices, and pathways. Goal:
expand the prescribed future, propose new pathways and alternative
futures. Emotions: dystopian optimism, hope. Temporality: the future is
distant -- change will be broader.

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
   is advancing or defending (which may be the same as, a staged version
   of, or a departure from the prescribed future).
3. Ask: is the comment's OVERARCHING FUNCTION to shape a collective
   trajectory through Advocate, Question, Reject, or Contest (Section B)?
   Consider the comment as a whole -- do not decide this by counting how
   many sentences are technical, factual, or evaluative in isolation.
4. If YES: classify NEGOTIATION as the dominant activity, and select
   Advocate, Question, Reject, or Contest according to which function
   dominates.
5. If NO: ask whether the comment's main function is to materialize a
   preferred future through present, planned, delayed, refused, or
   rerouted practices (Section B, Enactment) -- not merely to mention a
   personal action in service of a negotiating move. If YES, classify
   ENACTMENT as dominant.
6. Otherwise, classify the comment's cognitive assessment as EVALUATION.
7. Determine the ORIENTATION using the full configuration of narrative,
   goal, emotion, temporality, relationship to the prescribed future, and
   practice implications (Section C) -- never from sentiment or keywords
   alone.
8. Enforce the exact orientation-activity performance matrix (Section D).

This is an internal application decision procedure. It does not introduce
any new theoretical category beyond Evaluation, Negotiation, Enactment,
and the four orientations already defined in the paper.

====================================================================
F. GROUNDING EXAMPLES (paraphrased or quoted from the manuscript's dataset)
====================================================================

Example (CATALYZER, Evaluation/Simplify) -- Whirlpool forum:
"Once EVs are cheaper to buy than ICE cars the transition will happen
fast... EVs can stand on their own merits now."
-> Assesses the prescribed future as already unproblematic, with no
advocacy, question, rejection, or contestation directed at another actor
or claim: EVALUATION / SIMPLIFY / CATALYZER.

Example (CATALYZER, Negotiation/Advocate) -- public consultation (Joe):
"We are already so far behind! We need to sprint to catch up. We should
be WORLD LEADERS in solar and battery manufacturing. Why are we not using
our own minerals to make batteries for EVs on a global scale??"
-> Recruits others, frames the transition as a collective endeavor, and
calls for urgent collective action: NEGOTIATION / ADVOCATE / CATALYZER.

Example (CATALYZER, Negotiation/Advocate, MOSTLY TECHNICAL CONTENT --
functional dominance over volume) -- forum exchange, User 1:
"EVs will be on an exponential adoption curve. Everyone will want one...
Nobody will want an expensive 2nd hand ICE... Globally, governments are
going to start making fossil fuels very expensive. T-A-X-E-S will be
levied on this foul, polluting rubbish we are all burning today... Or are
you advocating that we go back to bicycles and horses, or maybe just
buses?"
-> This comment is dominated, by sentence count, by predictions and
assessments about adoption curves and taxation. However, its OVERARCHING
FUNCTION is to defend, promote, and normalize the prescribed future and
to challenge an implied alternative position ("Or are you advocating that
we go back to bicycles and horses...?"). The technical predictions are
mobilized as rhetorical support for this advocacy, not presented as a
self-standing assessment. Per the functional-dominance rule (Section B),
NEGOTIATION is primary, not Evaluation: NEGOTIATION / ADVOCATE /
CATALYZER.

Example (AMBIVALENT, Evaluation/Stall) -- interview (Clara):
"Living in Outback Northwest Queensland there's no charging stations at
the time. I did like the appeal of an electric vehicle... But just at the
time I went and bought a fairly decent car... So if we got an EV it would
just be our daily run around."
-> Careful, cautious consideration of material conditions, with no
attempt to persuade, question, reject, or contest another actor's claim:
EVALUATION / STALL / AMBIVALENT (may carry a secondary ENACTMENT / DELAY
/ AMBIVALENT classification when secondary classifications are permitted
for the request, since a separable practice-change detail is present).

Example (AMBIVALENT, Negotiation/Question) -- Facebook comment (Martin):
"Have you thought about what they are gonna do with all the batteries
once they expire because they aren't recyclable?"
-> Seeks reassurance/proof before committing: NEGOTIATION / QUESTION /
AMBIVALENT.

Example (AMBIVALENT, Negotiation/Question, conditional affordability
argument) -- Facebook comment (Ned):
"To legislate in their favour is a further disadvantage to those already
struggling... So where do we get the $50k to buy the cheapest new EV? It
will not be possible for us to make the transition until a huge number of
second hand EV's hit the market. And that won't happen until the
governments, state and federal, change their entire fleets to EV's..."
-> Establishes conditions that other actors (governments) must satisfy
before the consumer can join the prescribed future: NEGOTIATION /
QUESTION / AMBIVALENT.

Example (AMBIVALENT, Negotiation/Question, MOSTLY PERSONAL/TECHNICAL
CONTENT -- functional dominance over volume) -- forum exchange, User 2:
"Better tell that to the Prius owners replacing their batteries. My car
is now 13 years old... Batteries wear over time... so what magic bullet
have you discovered that defies physics...? Once someone like me can get
a used EV for <$10k, and have the battery replaced cheaply, then I'll
agree with you... I'm not anti EV, I'm just realistic about costs and
time frames."
-> Most sentences describe personal car ownership and battery physics.
However, this personal/technical content is mobilized to CHALLENGE
another comment's claim and to ESTABLISH CONDITIONS ("once... then I'll
agree with you") under which the consumer would join the transition. This
is a functional Question, not a self-standing Evaluation or Enactment:
NEGOTIATION / QUESTION / AMBIVALENT.

Example (RESISTANT, Negotiation/Reject) -- Facebook comments (Tom,
Jocelyn):
"No thanks, protest here we come. We get a say, this is our country not
the governments'. I say freedom of choice, freedom to speak..."
"The big green lie to cost taxpayers billions. Politicians forcing us to
go this way need to be voted out."
-> Refuses the prescribed future, frames it as coercive, defends
autonomy, calls for political opposition: NEGOTIATION / REJECT /
RESISTANT.

Example (RESISTANT, Negotiation/Reject) -- YouTube comment (Raj):
"We don't need politicians and their cronies telling us what sort of car
we can have."
-> Rejects the legitimacy of the authority behind the prescribed future,
no alternative future proposed: NEGOTIATION / REJECT / RESISTANT.

Example (RESISTANT, Negotiation/Reject, CONTAINS AN EVALUATION OF COSTS
AND ADOPTION CURVES -- functional dominance over volume) -- forum
exchange, User 3:
"Nope, I'm not confused, thanks for the concern though... Not even close
to the financial ruin you are trying to peddle... Technology adoption
curves typically look like bell curves... not what you are suggesting...
This is delusional."
-> Although this comment discusses costs and technology-adoption curves,
its central, overarching function is to REJECT and REBUT another actor's
account and resist the collective trajectory being promoted ("thanks for
the concern though," "This is delusional"). Per the functional-dominance
rule, NEGOTIATION is primary: NEGOTIATION / REJECT / RESISTANT.

Example (EXPANDER, Negotiation/Contest) -- forum comment (Bill):
"Cycling around Sydney is heavily unpleasant with the excessive emissions
from commercial vehicles and buses. If you want modal shift and reduced
emissions, it starts with reducing the impact of vehicles on pedestrians
and cyclists. There's no reason we can't adopt tighter standards mandated
by Europe and California... Australia's regulation around vehicle
emissions and efficiency is utterly laughable by global standards."
-> This is not a standalone evaluation of regulation. It attempts to
redirect collective change toward a broader preferred future (modal
shift, pedestrians, cyclists, tighter vehicle standards, systemic
regulation): NEGOTIATION / CONTEST / EXPANDER.

Example (EXPANDER, Negotiation/Contest, DECLARATIVE, no imperative) --
Facebook comment (Dan):
"The future is less cars, in higher density pedestrian/bike and train
orientated urban environments, where cars are a secondary transport
really only for those who really need it."
-> Advances a different, broader collective trajectory without assessing
the prescribed future's own merits: NEGOTIATION / CONTEST / EXPANDER.

Example (EXPANDER, Negotiation/Contest, responds to and contests another
position) -- forum exchange, User 4:
"I fully get what you're saying, it's not rocket science, but that's not
what I'm on about... I simply object to being told I'm an idiot... I'd
like to see passenger cars filled with passengers, less cars on the road,
less money spent on new roads!... Where people may simply drive less."
-> Responds to and contests the framing of another comment, and advances
a broader collective trajectory (fewer cars, more shared use, less
road spending): NEGOTIATION / CONTEST / EXPANDER.

Example (EXPANDER, Evaluation/Complexify) -- interview (Peter):
"The embodied carbon in a new vehicle... is more than the emissions that
are going to be produced by the current vehicle over the course of its
lifetime... So that's the plan: to extract maximum value out of that
current vehicle... I am at the moment on a waiting list for a new
electric cargo bike."
-> A standalone, systemic assessment of trade-offs, with no attempt to
advocate/question/reject/contest another actor's position: EVALUATION /
COMPLEXIFY / EXPANDER (may carry a secondary ENACTMENT / REROUTE /
EXPANDER classification when secondary classifications are permitted).

Example (RESISTANT, Evaluation/Avoid) -- Facebook comment (Esther):
"Electric vehicles are not the solution... The current electricity
infrastructure can't keep up with the demand... I feel this is a lazy
policy just appealing to city people..."
-> A standalone dismissal of the transition, with no advocacy/question/
rejection/contestation FUNCTION directed at another actor: EVALUATION /
AVOID / RESISTANT.

Example (RESISTANT, Enactment/Prevent) -- news media comment (StarT):
"I for one WILL NOT be forced into an elec vehicle and spend half my
travel time charging the damn thing to go to hell."
-> Entrenches current practice, explicitly refusing the prescribed
future: ENACTMENT / PREVENT / RESISTANT.

Example (EXPANDER, Enactment/Reroute) -- public consultation (Phillip):
"I uprooted my life and moved from the Sunshine Coast to Melbourne with
some of my strongest reasoning being the ability to use public transport,
ride a bike around and use a car as little as possible."
-> Directs present practice toward a different, broader preferred
future, with no functional advocacy/question/rejection/contestation move
directed at another actor: ENACTMENT / REROUTE / EXPANDER.

====================================================================
G. PRIMARY AND SECONDARY CLASSIFICATIONS
====================================================================

Return ONE dominant classification (main_activity, activity_subtype,
main_orientation) determined via the Decision Procedure (Section E).

The calling application will indicate, for each request, whether
secondary classifications are permitted. When permitted, return UP TO TWO
secondary classifications ONLY when the comment contains a SUBSTANTIVELY
DISTINCT performance of a second activity -- one that stands on its own
and is not simply the evidence mobilized to perform the dominant
Negotiation move. Evaluation-sounding content that only serves as
rhetorical support for an Advocate, Question, Reject, or Contest move is
NOT a substantively distinct secondary Evaluation -- it is part of the
dominant Negotiation performance and must not be listed separately. When
secondary classifications are NOT permitted for a request, always return
an empty list for secondary_classifications.

====================================================================
H. WHAT NOT TO DO
====================================================================

Do NOT determine or output any future-making challenge (Convoluted
Evaluations, Confrontational Negotiations, or Competing Enactments) or any
Fragile Futures assessment for a single comment -- these are emergent,
exchange-level outcomes determined by the calling application across
multiple comments, never by you for one comment.

Do NOT generate policy or managerial recommendations, instruments, or
evidence requirements -- these are provided by the calling application as
fixed reference content.

Do NOT introduce any interaction-type category, negotiation-evidence
category, relational-positioning category, or named-address rule. Base
your classification only on the functional definitions in Sections B-E.

====================================================================
OUTPUT FORMAT -- Return ONLY valid JSON
====================================================================

{
  "prescribed_future_acknowledged": "Brief restatement of the prescribed future",

  "main_activity": "EVALUATION, NEGOTIATION, or ENACTMENT",
  "activity_subtype": "SIMPLIFY, STALL, AVOID, COMPLEXIFY, ADVOCATE, QUESTION, REJECT, CONTEST, ACCELERATE, DELAY, PREVENT, REROUTE",
  "activity_rationale": "Apply the Decision Procedure (Section E) explicitly: state whether the comment's overarching function is to Advocate/Question/Reject/Contest, citing specific phrases, BEFORE considering volume of evaluative content. If the comment contains substantial technical/evaluative content mobilized in service of a negotiating move, say so explicitly and explain why Negotiation remains dominant.",

  "main_orientation": "CATALYZER, AMBIVALENT, RESISTANT, or EXPANDER",
  "orientation_rationale": "The configuration of narrative, goal, emotion, temporality, and practice implications (Section C) that supports this orientation",

  "secondary_classifications": [
    {"activity": "...", "activity_subtype": "...", "orientation": "...", "rationale": "..."}
  ],

  "narrative_identified": "Name of the dominant narrative from Section C",
  "emotions_identified": "Comma-separated list of emotions evidenced in the comment",
  "temporality_identified": "How the comment expresses temporality, per Section C",
  "notable_conditions_of_adoption": "Which notable condition of adoption (Section C) is evidenced, if any",

  "supporting_text": "The specific phrase(s) from the FOCAL comment that support the classification",
  "context_note": "Brief note on how the supplied context (if any) helped interpret the focal comment's relation to another position, claim, or pathway -- leave empty if no context was supplied or used",
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
        "temporality": "The future is distant -- there will be no change",
        "activities": "Avoid - Reject - Prevent",
    },
    "EXPANDER": {
        "color": "#7D3C98", "bg": "#F4ECF7", "border": "#9B59B6",
        "goal": "Expand the prescribed future (magnitude of change), propose new pathways and alternative futures",
        "narrative": "Bigger Picture Narrative",
        "temporality": "The future is distant -- change will be broader",
        "activities": "Complexify - Contest - Reroute",
    }
}

ACTIVITY_META = {
    "EVALUATION":  {"color": "#2980B9", "bg": "#EBF5FB",
        "definition": "Consumers' cognitive assessment of the prescribed future.",
        "subtypes": {"SIMPLIFY": "CATALYZER", "STALL": "AMBIVALENT", "AVOID": "RESISTANT", "COMPLEXIFY": "EXPANDER"}},
    "NEGOTIATION": {"color": "#E67E22", "bg": "#FEF9E7",
        "definition": "The activity through which consumers attempt to shape collective trajectories toward a preferred future.",
        "subtypes": {"ADVOCATE": "CATALYZER", "QUESTION": "AMBIVALENT", "REJECT": "RESISTANT", "CONTEST": "EXPANDER"}},
    "ENACTMENT":   {"color": "#8E44AD", "bg": "#F5EEF8",
        "definition": "What consumers do in the present to materialize a preferred future.",
        "subtypes": {"ACCELERATE": "CATALYZER", "DELAY": "AMBIVALENT", "PREVENT": "RESISTANT", "REROUTE": "EXPANDER"}},
}

CHALLENGE_DEFINITIONS = {
    "CONVOLUTED_EVALUATIONS": {
        "label": "Convoluted Evaluations", "activity": "EVALUATION",
        "definition": (
            "Arise from the interplay among Simplify, Stall, Avoid, and "
            "Complexify performances, supported by different and often "
            "clashing narratives and emotions."
        )
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "label": "Confrontational Negotiations", "activity": "NEGOTIATION",
        "definition": (
            "Arise when consumers Advocate, Question, Reject, and Contest "
            "preferred futures without conceding to alternative ones."
        )
    },
    "COMPETING_ENACTMENTS": {
        "label": "Competing Enactments", "activity": "ENACTMENT",
        "definition": (
            "Arise when consumers Accelerate, Delay, Prevent, and Reroute "
            "different preferred futures through their current practices."
        )
    },
}
FRAGILE_FUTURES_DEFINITION = (
    "Fragile Futures: multiple, volatile, and conflicting preferred futures "
    "that may interfere with the actualization of the prescribed one."
)

# Internal-only constants -- used solely for hidden benchmark examples in
# Advanced / Developer Tools, never shown as presets in the normal flow.
PF_EV = (
    "Transition all vehicles to Zero Emission Vehicles (EVs) to achieve Australia's "
    "net-zero emissions targets, as prescribed by Australia's National Electric "
    "Vehicle Strategy (2023)"
)

# ─────────────────────────────────────────
# STATIC ROADMAPS (verbatim step names from the manuscript) -- never
# generated by the LLM.
# ─────────────────────────────────────────
POLICY_ROADMAP_STEPS = [
    ("1", "Determine the prescribed future",
     "Define the future the intervention prescribes, the practice changes on which its "
     "public value depends, and which populations may lack the resources to enact it."),
    ("2", "Map future-making orientations",
     "Triangulate discourse with behavior; analyze by application/decision context."),
    ("3", "Diagnose key future-making challenges",
     "Identify which of Convoluted Evaluations, Confrontational Negotiations, or "
     "Competing Enactments is most pressing; the three activities are interdependent, "
     "not sequential."),
    ("4", "Implement support initiatives",
     "Match support measures to future-making orientations."),
    ("5", "Facilitate enactment",
     "Provide the infrastructure and capabilities required to reconfigure practices "
     "safely and equitably."),
    ("6", "Measure multiple outcomes",
     "Measure how the intervention is progressing across multiple outcomes."),
    ("7", "Revise the intervention",
     "Continuously revise the intervention, providing more support as needed."),
]

MANAGER_ROADMAP_STEPS = [
    ("1", "Determine the prescribed future",
     "Define the intervention through the future it asks consumers to enact."),
    ("2", "Consider future-making orientations",
     "Treat orientations as a diagnostic lens, not fixed consumer segments."),
    ("3", "Monitor key future-making challenges",
     "Build a future-making customer journey combining discursive, experiential, "
     "and behavioral evidence."),
    ("4", "Select an orientation-sensitive response",
     "Match responses to the orientations identified in the data."),
    ("5", "Match messaging to key future-making challenges",
     "Communicate achievements alongside uncertainty, trade-offs, and limitations."),
    ("6", "Support consumers through enactment",
     "Place support at touchpoints where practices change."),
]

POLICY_ORIENTATION_REFERENCE = {
    "CATALYZER": "Time-limited sandboxes, independent evaluation, mandatory failure reporting, predefined thresholds for expansion/withdrawal.",
    "AMBIVALENT": "Impact assessments, staged authorization, sunset clauses, public registers, guaranteed alternative pathways.",
    "RESISTANT": "Protect human-review and appeal rights, prohibit unacceptable uses, independent audits, moratoria where evidence is insufficient.",
    "EXPANDER": "Deliberative forums, broader impact assessment, fund complementary pathways, citizen assemblies, data trusts, alternative governance models.",
}
MANAGER_ORIENTATION_REFERENCE = {
    "CATALYZER": "Governed pilots, peer learning, documentation, limitation reporting. Avoid inevitability claims and treating early adopters as representative.",
    "AMBIVALENT": "Comparison tools, staged adoption, transparent evidence, training, human assistance. Avoid artificial urgency and framing hesitation as ignorance.",
    "RESISTANT": "Consultation, opt-outs, human review, audits, appeals, harm protections. Avoid 'there is no alternative' messaging, ridicule, hidden automation.",
    "EXPANDER": "Participatory design, futures workshops, broader-impact evaluation, partnerships, alternative governance/service models. Avoid presenting the offering as complete or dismissing critique.",
}
CROSS_ORIENTATION_NOTE = (
    "Cross-orientation interference check: check whether a response "
    "tailored to one orientation intensifies fragility elsewhere."
)

# ─────────────────────────────────────────
# BENCHMARK EXAMPLES -- hidden from normal workflows; used only inside
# Advanced / Developer Tools for the Coding Consistency Check.
# Includes manuscript-grounded Negotiation tests plus retained Evaluation
# controls, so improving Negotiation recall does not push every comment
# toward Negotiation.
# ─────────────────────────────────────────
EXAMPLES = {
    # --- Evaluation controls (retained) ---
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

    # --- Manuscript-grounded Negotiation tests (Section 12 of the request) ---
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
            "one... Nobody will want an expensive 2nd hand ICE... Globally, "
            "governments are going to start making fossil fuels very expensive. "
            "T-A-X-E-S will be levied on this foul, polluting rubbish we are all "
            "burning today... Or are you advocating that we go back to bicycles "
            "and horses, or maybe just buses?"
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
            "adopt tighter standards mandated by Europe and California... "
            "Australia's regulation around vehicle emissions and efficiency is "
            "utterly laughable by global standards."
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
# CONSISTENCY SAFEGUARD -- enforce the mandatory orientation x subtype
# pairing table for main and secondary classifications
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
# CORE FUNCTION -- focal comment + context, paper-grounded schema only
# ─────────────────────────────────────────

def analyze_comment(prescribed_future: str, focal_text: str, context_text: str = "",
                     context_type: str = "NONE", is_consultation: bool = False,
                     api_key: str = None, allow_secondary_classifications: bool = True) -> dict:
    client = openai.OpenAI(api_key=api_key)

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
Contest a collective trajectory (Section B) -- do this BEFORE considering
how much of the comment reads as technical, factual, or evaluative in
isolation. If the comment mobilizes assessments, predictions, or personal
experience in service of such a function, Negotiation is dominant, per
the functional-dominance rule. Only if no such function is present should
you consider Enactment, and only if neither applies should you classify
Evaluation. Verify the activity_subtype belongs to the valid pairing row
for its own orientation (Section D) before responding. Do not produce any
future-making challenge, Fragile Futures assessment, or policy/managerial
recommendation -- these are outside your task.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    parsed = json.loads(response.choices[0].message.content)
    parsed = enforce_consistency(parsed)
    if not allow_secondary_classifications:
        parsed["secondary_classifications"] = []
    parsed = enforce_context_metadata(parsed, context_available, context_type)
    return parsed


def run_consistency_suite(api_key: str) -> dict:
    """Tests whether the current prompt reproduces predetermined coding
    decisions for benchmark examples paraphrased/quoted from the
    manuscript. This is an internal consistency check -- it does NOT
    constitute empirical validation, intercoder reliability, or evidence
    of generalizability."""
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
    overall_activity_agreement = sum(r["activity_match"] for r in results) / n
    overall_subtype_agreement = sum(r["subtype_match"] for r in results) / n
    overall_orientation_agreement = sum(r["orientation_match"] for r in results) / n
    overall_agreement = sum(r["match"] for r in results) / n

    negotiation_examples = [r for r in results if r["expected"][1] == "NEGOTIATION"]
    negotiation_recall = (
        sum(r["activity_match"] for r in negotiation_examples) / len(negotiation_examples)
        if negotiation_examples else None
    )

    return {
        "results": results,
        "overall_agreement": overall_agreement,
        "overall_activity_agreement": overall_activity_agreement,
        "overall_subtype_agreement": overall_subtype_agreement,
        "overall_orientation_agreement": overall_orientation_agreement,
        "negotiation_recall": negotiation_recall,
        "n_negotiation_examples": len(negotiation_examples),
    }


# ─────────────────────────────────────────
# COMMENT / THREAD DATA STRUCTURES (technical infrastructure, no theory)
# ─────────────────────────────────────────

def has_comment_text_column(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip() for c in df.columns}
    return bool(cols & {"comment_text", "text", "comment"})


def strip_trailing_orientation_label(text: str) -> str:
    """Removes manuscript-style parenthetical orientation labels (e.g.
    '(Catalyzer Orientation)') from the end of a comment, so the expected
    answer is not leaked to the model. Purely a text-cleaning step --
    introduces no new theoretical construct."""
    cleaned = re.sub(
        r'\(\s*(Catalyzer|Ambivalent|Resistant|Expander)\s+Orientation\s*\)\.?\s*$',
        '', text, flags=re.IGNORECASE
    ).strip()
    return cleaned


USER_LABEL_PATTERN = re.compile(r'(?im)^\s*User\s*\d+\s*:\s*')


def detect_user_labeled_exchange(text: str) -> int:
    """Returns the count of 'User N:' style labels found in the text."""
    return len(USER_LABEL_PATTERN.findall(text or ""))


def build_comment_records_from_user_labels(text: str) -> list:
    """Parses manuscript-style exchanges where consecutive comments are
    labeled 'User 1:', 'User 2:', etc. Assigns a shared technical
    thread_id and preserves order; does NOT invent parent-reply
    relationships (parent_comment_id stays None so build_context uses the
    thread-window mechanism already in place for nearby comments)."""
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
# CORPUS-LEVEL AGGREGATION -- paper-grounded challenge review only
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
    valid = [r for r in results if r and "_error" not in r]
    by_id = {r.get("_comment_id"): r for r in valid if r.get("_comment_id")}
    candidates = {k: [] for k in CHALLENGE_DEFINITIONS}

    # Linked parent-reply pairs (structured metadata)
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
    links (e.g., a manuscript-style labeled exchange), summarizes the
    diversity of activities/orientations present within that thread. This
    is descriptive only -- it does not claim an observed interaction or a
    diagnosed challenge."""
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


def build_linked_pair_table(results: list) -> pd.DataFrame:
    valid = [r for r in results if r and "_error" not in r]
    by_id = {r.get("_comment_id"): r for r in valid if r.get("_comment_id")}
    rows = []
    for r in valid:
        parent_id = r.get("_parent_comment_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        rows.append({
            "parent_comment_id": parent.get("_comment_id", ""),
            "parent_orientation": _clean_enum(parent.get("main_orientation", "")).upper(),
            "parent_activity": _clean_enum(parent.get("main_activity", "")).upper(),
            "parent_subtype": _clean_enum(parent.get("activity_subtype", "")).upper(),
            "reply_comment_id": r.get("_comment_id", ""),
            "reply_orientation": _clean_enum(r.get("main_orientation", "")).upper(),
            "reply_activity": _clean_enum(r.get("main_activity", "")).upper(),
            "reply_subtype": _clean_enum(r.get("activity_subtype", "")).upper(),
        })
    return pd.DataFrame(rows)


def build_results_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        if not r:
            continue
        base = {
            "comment_index": r.get("_original_index", ""),
            "comment_id": r.get("_comment_id", ""),
            "thread_id": r.get("_thread_id", ""),
            "parent_comment_id": r.get("_parent_comment_id", ""),
            "comment_text": r.get("_comment_text", ""),
        }
        if "_error" in r:
            base.update({
                "main_orientation": "ERROR", "main_activity": "", "activity_subtype": "",
                "secondary_classifications": "", "activity_rationale": "",
                "orientation_rationale": "", "supporting_text": "",
                "context_used": "", "context_type": "", "input_scope_warning": "",
                "error": r.get("_error", "")
            })
            rows.append(base)
            continue
        secondary = get_secondary_classifications(r)
        sec_str = " || ".join(
            f"{s.get('orientation','')}/{s.get('activity','')}/{s.get('activity_subtype','')}: {s.get('rationale','')}"
            for s in secondary
        )
        base.update({
            "main_orientation": _clean_enum(r.get("main_orientation", "")).upper(),
            "main_activity": _clean_enum(r.get("main_activity", "")).upper(),
            "activity_subtype": _clean_enum(r.get("activity_subtype", "")).upper(),
            "secondary_classifications": sec_str,
            "activity_rationale": r.get("activity_rationale", ""),
            "orientation_rationale": r.get("orientation_rationale", ""),
            "supporting_text": r.get("supporting_text", ""),
            "context_used": r.get("context_used", ""),
            "context_type": r.get("context_type", ""),
            "input_scope_warning": r.get("input_scope_warning", "") or "",
            "error": ""
        })
        rows.append(base)
    return pd.DataFrame(rows)


def render_pct_bars(counts: dict, meta_dict: dict, total: int):
    if total == 0:
        st.caption("No data to display.")
        return
    for key, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct_val = round(cnt / total * 100, 1)
        meta = meta_dict.get(key, {})
        color = meta.get("color", "#888")
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                <span><strong>{key}</strong></span>
                <span style="color:#666;">{cnt} comments ({pct_val}%)</span>
            </div>
            <div style="background:#eee;border-radius:6px;height:14px;width:100%;overflow:hidden;">
                <div style="background:{color};width:{pct_val}%;height:14px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_document_summary(results: list, prescribed_future: str,
                           total_detected: int = None, sampling_description: str = ""):
    dist = compute_dominant_distributions(results)
    n = dist["n_analyzed"]
    n_errors = len([r for r in results if r and "_error" in r])

    if n == 0:
        st.error("No comments could be successfully analyzed.")
        return

    st.markdown(f"""
    <div style="background:#EBF5FB;border-left:5px solid #2980B9;border-radius:8px;padding:12px 18px;margin-bottom:16px;">
        <strong style="color:#2980B9;">Prescribed Future Analyzed:</strong><br>
        <em style="color:#333;">{prescribed_future}</em>
    </div>
    """, unsafe_allow_html=True)

    if sampling_description:
        st.caption(f"Sampling: {sampling_description}")
    st.caption(INTERPRETIVE_USE_NOTE)
    if n_errors:
        st.warning(f"{n_errors} comment(s) failed to analyze and were excluded.")

    n_with_context = sum(1 for r in results if r and "_error" not in r and r.get("context_used"))
    n_without_context = n - n_with_context

    st.markdown("### Comments Analyzed")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total comments detected", total_detected if total_detected is not None else n)
    c2.metric("Comments analyzed", n)
    c3.metric("With available context", n_with_context)
    c4.metric("Without context", n_without_context)
    st.caption(
        "These percentages and counts describe the analyzed comments only. "
        "They do not represent unique consumers, organizations, or "
        "population prevalence, and results depend on the source material, "
        "segmentation, and selected sample."
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Dominant-Orientation Distribution Among Analyzed Comments")
        render_pct_bars(dist["orientation_counts"], ORIENTATIONS, n)
    with col2:
        st.markdown("#### Dominant-Activity Distribution Among Analyzed Comments")
        render_pct_bars(dist["activity_counts"], ACTIVITY_META, n)
    with col3:
        st.markdown("#### Activity-Performance (Subtype) Distribution")
        for key, cnt in sorted(dist["subtype_counts"].items(), key=lambda x: -x[1]):
            pct_val = round(cnt / n * 100, 1)
            st.markdown(f"- **{key}**: {cnt} ({pct_val}%)")

    st.markdown("---")
    st.markdown("## Comments to Review for Emergent Future-Making Challenges")
    st.caption(
        "The paper defines these challenges as emergent outcomes arising "
        "when consumers perform the SAME activity differently across "
        "orientations. The lists below flag linked or same-thread comments "
        "as candidates for interpretive review. They do NOT constitute an "
        "automatic diagnosis that a challenge occurred, a percentage of "
        "affected comments, or a Fragile Futures score."
    )

    candidates = compute_challenge_review_candidates(results)
    has_links = any(len(v) > 0 for v in candidates.values())

    if has_links:
        for key, meta in CHALLENGE_DEFINITIONS.items():
            pairs = candidates[key]
            with st.expander(f"{meta['label']} -- {len(pairs)} linked comment pair(s) to review"):
                st.caption(meta["definition"])
                for p in pairs:
                    parent, reply = p["parent"], p["reply"]
                    st.markdown(
                        f"**Parent** [{parent.get('_comment_id','')}] -- "
                        f"{parent.get('main_orientation','')} / {parent.get('main_activity','')} / "
                        f"{parent.get('activity_subtype','')}"
                    )
                    st.caption(parent.get("_comment_text", "")[:300])
                    st.markdown(
                        f"**Reply** [{reply.get('_comment_id','')}] -- "
                        f"{reply.get('main_orientation','')} / {reply.get('main_activity','')} / "
                        f"{reply.get('activity_subtype','')}"
                    )
                    st.caption(reply.get("_comment_text", "")[:300])
                    st.markdown("---")
        st.markdown("#### Linked Parent-Reply Comments (descriptive table)")
        pair_df = build_linked_pair_table(results)
        if not pair_df.empty:
            st.dataframe(pair_df, use_container_width=True, height=250)

    thread_summary = compute_same_thread_diversity(results)
    if thread_summary:
        st.markdown("#### Same-Thread Exchanges (no explicit parent links -- for review)")
        st.caption(
            "These comments share a technical thread_id (e.g., a labeled "
            "exchange such as 'User 1', 'User 2', ...) but no explicit "
            "parent-reply relationship was specified. Diversity of "
            "performances within the same thread is shown below as "
            "candidate evidence for interpretive review -- it does not by "
            "itself demonstrate an observed interaction, clash, or Fragile "
            "Futures."
        )
        for tid, info in thread_summary.items():
            with st.expander(f"Thread '{tid}' -- {len(info['comments'])} comments"):
                for c in info["comments"]:
                    st.markdown(
                        f"[{c.get('_comment_id','')}] "
                        f"{c.get('main_orientation','')} / {c.get('main_activity','')} / "
                        f"{c.get('activity_subtype','')}"
                    )
                    st.caption(c.get("_comment_text", "")[:300])
                distinct_activities = {k[0] for k in info["performance_counts"]}
                for challenge_key, meta in CHALLENGE_DEFINITIONS.items():
                    if meta["activity"] in distinct_activities:
                        distinct_orientations_for_activity = {
                            k[2] for k in info["performance_counts"] if k[0] == meta["activity"]
                        }
                        if len(distinct_orientations_for_activity) >= 2:
                            st.info(
                                f"Candidate evidence for **{meta['label']}**: multiple "
                                f"orientations ({', '.join(sorted(distinct_orientations_for_activity))}) "
                                f"perform {meta['activity'].title()} differently within this thread. "
                                f"Requires interpretive review, not an automatic diagnosis."
                            )

    if not has_links and not thread_summary:
        st.info(
            "No parent-reply links or shared-thread exchanges were available "
            "among the analyzed comments. The activity-performance "
            "distribution above shows the diversity of performances present, "
            "but co-occurrence in a corpus without linked exchanges does not "
            "demonstrate interaction, clash, interference, or Fragile "
            "Futures. No fictitious exchanges have been reconstructed."
        )

    st.caption(FRAGILE_FUTURES_DEFINITION)

    st.markdown("---")
    st.markdown("### Comment-Level Detail")
    df = build_results_dataframe(results)
    display_cols = ["comment_index", "comment_id", "parent_comment_id", "main_orientation",
                     "main_activity", "activity_subtype", "context_type"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, height=350)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full results as CSV",
        data=csv_bytes, file_name="future_making_comment_analysis.csv", mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### Comment Rationale Explorer")
    valid_indexed = [(i, r) for i, r in enumerate(results) if r and "_error" not in r]
    if valid_indexed:
        option_labels = [f"[{r.get('_comment_id', i)}] {r.get('_comment_text', '')[:90]}..." for i, r in valid_indexed]
        chosen_pos = st.selectbox("Choose a comment to inspect:", options=range(len(option_labels)), format_func=lambda x: option_labels[x])
        chosen_idx, chosen_result = valid_indexed[chosen_pos]
        st.markdown("**Full focal comment text:**")
        st.info(chosen_result.get("_comment_text", ""))
        show_results(chosen_result, prescribed_future, show_interpretive_note=False, allow_secondary_display=True)
    else:
        st.caption("No valid comments available to explore.")

    st.markdown("---")
    render_static_roadmaps(mode="corpus")


# ─────────────────────────────────────────
# UI HELPER FUNCTIONS
# ─────────────────────────────────────────

def render_static_roadmaps(mode: str = "single"):
    st.markdown("## Roadmap Reference")
    if mode == "single":
        st.caption(
            "This application, for a single comment, primarily supports "
            "mapping future-making orientations and activities. It does not "
            "generate later-step actions from one comment."
        )
    else:
        st.caption(
            "This application supports mapping orientations and organizing "
            "comments for human review of possible future-making challenges. "
            "It does not claim to complete the later roadmap steps."
        )
    policy_tab, manager_tab = st.tabs(["Policymaking Roadmap", "Managerial Roadmap"])
    with policy_tab:
        for num, title, desc in POLICY_ROADMAP_STEPS:
            st.markdown(f"**Step {num}: {title}**")
            st.caption(desc)
        with st.expander("Orientation-sensitive reference material (not generated by the analysis)"):
            for ori, text in POLICY_ORIENTATION_REFERENCE.items():
                st.markdown(f"**{ori}:** {text}")
    with manager_tab:
        for num, title, desc in MANAGER_ROADMAP_STEPS:
            st.markdown(f"**Step {num}: {title}**")
            st.caption(desc)
        with st.expander("Orientation-sensitive reference material (not generated by the analysis)"):
            for ori, text in MANAGER_ORIENTATION_REFERENCE.items():
                st.markdown(f"**{ori}:** {text}")
        st.caption(CROSS_ORIENTATION_NOTE)


def show_results(result: dict, prescribed_future: str, show_interpretive_note: bool = True,
                  allow_secondary_display: bool = False):
    orientation = _clean_enum(result.get("main_orientation", "")).upper().strip()
    main_act    = _clean_enum(result.get("main_activity", "")).upper().strip()
    act_sub     = _clean_enum(result.get("activity_subtype", "N/A")).upper().strip()
    secondary = get_secondary_classifications(result) if allow_secondary_display else []

    if show_interpretive_note:
        st.caption(INTERPRETIVE_USE_NOTE)

    st.markdown(f"""
    <div style="background:#EBF5FB;border-left:5px solid #2980B9;border-radius:8px;padding:12px 18px;margin-bottom:16px;">
        <strong style="color:#2980B9;">Prescribed Future Analyzed:</strong><br>
        <em style="color:#333;">{prescribed_future}</em>
    </div>
    """, unsafe_allow_html=True)

    warning_text = result.get("input_scope_warning", "") or ""
    if warning_text:
        st.warning(f"Input-scope note: {warning_text}")
    if result.get("_consistency_note"):
        st.caption(f"Note: {result['_consistency_note']}")

    ctx_type = result.get("context_type", "NONE")
    ctx_used = result.get("context_used", False)
    st.caption(
        f"Technical note -- context used for interpretation: {ctx_type}"
        + (" (no context was available for this comment)" if not ctx_used else "")
    )
    if result.get("context_note"):
        st.caption(f"Context note: {result.get('context_note')}")

    col1, col2 = st.columns(2)
    with col1:
        cfg = ORIENTATIONS.get(orientation, {})
        st.markdown(f"""
        <div style="background:{cfg.get('bg','#f5f5f5')};border-left:6px solid {cfg.get('border','#999')};border-radius:10px;padding:16px 18px;min-height:210px;">
            <h3 style="color:{cfg.get('color','#555')};margin:0;font-size:22px;">{orientation}</h3>
            <p style="color:#777;margin:6px 0 2px;font-size:11px;">{cfg.get('narrative','')}</p>
            <p style="color:#777;margin:2px 0;font-size:11px;">{cfg.get('temporality','')}</p>
            <p style="color:#777;margin:2px 0;font-size:11px;">{cfg.get('goal','')}</p>
            <p style="color:#999;margin:6px 0 0;font-size:10px;">Dominant orientation</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        ameta = ACTIVITY_META.get(main_act, {})
        st.markdown(f"""
        <div style="background:{ameta.get('bg','#f5f5f5')};border-left:6px solid {ameta.get('color','#555')};border-radius:10px;padding:16px 18px;min-height:210px;">
            <h3 style="color:{ameta.get('color','#555')};margin:0;font-size:20px;">{main_act}</h3>
            <p style="color:#555;margin:4px 0 3px;font-size:12px;"><strong>Dominant Future-Making Activity</strong></p>
            <span style="background:#f0f0f0;border:1.5px solid #bbb;color:#444;border-radius:12px;padding:3px 10px;font-weight:bold;font-size:12px;">-> {act_sub}</span>
            <p style="color:#777;margin:8px 0 0;font-size:11px;font-style:italic;">{ameta.get('definition','')}</p>
        </div>
        """, unsafe_allow_html=True)

    if allow_secondary_display and secondary:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Secondary Classification(s)")
        st.caption(
            "Future-making activities are interdependent and recursive; "
            "orientations may be combined. These are additional, "
            "substantively DISTINCT classifications -- not evidence that "
            "was mobilized in service of the dominant classification."
        )
        for sec in secondary:
            sec_ori, sec_act, sec_sub = sec.get("orientation", ""), sec.get("activity", ""), sec.get("activity_subtype", "")
            sec_cfg, sec_ameta = ORIENTATIONS.get(sec_ori, {}), ACTIVITY_META.get(sec_act, {})
            st.markdown(f"""
            <div style="border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin-bottom:8px;background:#fafafa;">
                <span style="background:{sec_cfg.get('bg','#eee')};border:1.5px solid {sec_cfg.get('border','#999')};color:{sec_cfg.get('color','#555')};border-radius:14px;padding:3px 10px;font-weight:bold;font-size:12px;">{sec_ori or 'N/A'}</span>
                &nbsp;-> &nbsp;
                <span style="background:{sec_ameta.get('bg','#eee')};border:1.5px solid {sec_ameta.get('color','#999')};color:{sec_ameta.get('color','#555')};border-radius:14px;padding:3px 10px;font-weight:bold;font-size:12px;">{sec_act or 'N/A'} / {sec_sub or 'N/A'}</span>
                <p style="font-size:12px;color:#666;margin:6px 0 0;">{sec.get('rationale','')}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab_ori, tab_act = st.tabs(["Orientation Rationale", "Activity Rationale"])
    with tab_ori:
        st.write(result.get("orientation_rationale", "--"))
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**Narrative**"); c1.caption(result.get("narrative_identified", "--"))
        c2.markdown("**Emotions**"); c2.caption(result.get("emotions_identified", "--"))
        c3.markdown("**Temporality**"); c3.caption(result.get("temporality_identified", "--"))
        c4.markdown("**Notable Conditions**"); c4.caption(result.get("notable_conditions_of_adoption", "--"))
    with tab_act:
        st.write(result.get("activity_rationale", "--"))
        st.markdown("**Supporting text (from the focal comment):**")
        st.info(result.get("supporting_text", "--"))
        for act_name, meta in ACTIVITY_META.items():
            is_main = (act_name == main_act)
            border = f"3px solid {meta['color']}" if is_main else "1px solid #ddd"
            st.markdown(f"""
            <div style="border:{border};border-radius:8px;padding:10px 14px;margin-bottom:8px;background:{'#fff' if is_main else '#fafafa'};">
                <strong style="color:{meta['color']};">{act_name}</strong>
                {'<span style="background:#27AE60;color:white;border-radius:8px;padding:1px 8px;font-size:11px;margin-left:8px;">DOMINANT</span>' if is_main else ''}<br>
                <span style="font-size:11px;color:#555;">{meta['definition']}</span>
            </div>
            """, unsafe_allow_html=True)

    if not allow_secondary_display:
        st.markdown("---")
        render_static_roadmaps(mode="single")
        st.markdown("---")
        st.caption(f'"{PAPER_TITLE}" | Read the paper: {PAPER_URL}')


def render_mode_selector():
    st.markdown("""
    <style>
    div.st-key-mode_single_btn_active button { background-color:#2980B9 !important;border:2px solid #2980B9 !important;color:white !important;font-weight:bold !important; }
    div.st-key-mode_single_btn_inactive button { background-color:#EBF5FB !important;border:2px solid #AED6F1 !important;color:#888 !important;font-weight:normal !important; }
    div.st-key-mode_doc_btn_active button { background-color:#8E44AD !important;border:2px solid #8E44AD !important;color:white !important;font-weight:bold !important; }
    div.st-key-mode_doc_btn_inactive button { background-color:#F5EEF8 !important;border:2px solid #D7BDE2 !important;color:#888 !important;font-weight:normal !important; }
    </style>
    """, unsafe_allow_html=True)

    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = MODE_SINGLE
    if "pending_mode" not in st.session_state:
        st.session_state["pending_mode"] = None

    active_mode = st.session_state["app_mode"]
    col1, col2 = st.columns(2)
    with col1:
        is_active = (active_mode == MODE_SINGLE)
        label = f"[Selected] {MODE_SINGLE_LABEL}" if is_active else MODE_SINGLE_LABEL
        if st.button(label, key="mode_single_btn_active" if is_active else "mode_single_btn_inactive", use_container_width=True) and not is_active:
            st.session_state["pending_mode"] = MODE_SINGLE
            st.rerun()
    with col2:
        is_active2 = (active_mode == MODE_DOC)
        label2 = f"[Selected] {MODE_DOC_LABEL}" if is_active2 else MODE_DOC_LABEL
        if st.button(label2, key="mode_doc_btn_active" if is_active2 else "mode_doc_btn_inactive", use_container_width=True) and not is_active2:
            st.session_state["pending_mode"] = MODE_DOC
            st.rerun()

    pending_mode = st.session_state["pending_mode"]
    if pending_mode and pending_mode != active_mode:
        pending_label = MODE_SINGLE_LABEL if pending_mode == MODE_SINGLE else MODE_DOC_LABEL
        st.warning(f"Switch to '{pending_label}'? Any unsaved input in the current mode may be lost.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Proceed", type="primary", key="confirm_switch_btn", use_container_width=True):
                st.session_state["app_mode"] = pending_mode
                st.session_state["pending_mode"] = None
                st.rerun()
        with c2:
            if st.button("Cancel", key="cancel_switch_btn", use_container_width=True):
                st.session_state["pending_mode"] = None
                st.rerun()
    else:
        active_label = MODE_SINGLE_LABEL if active_mode == MODE_SINGLE else MODE_DOC_LABEL
        active_color = "#2980B9" if active_mode == MODE_SINGLE else "#8E44AD"
        active_bg = "#EBF5FB" if active_mode == MODE_SINGLE else "#F5EEF8"
        st.markdown(f"""
        <div style="background:{active_bg};border-left:4px solid {active_color};padding:8px 14px;border-radius:6px;margin:10px 0 16px 0;">
            <strong style="color:{active_color};">Current mode:</strong> {active_label}
        </div>
        """, unsafe_allow_html=True)
    return st.session_state["app_mode"]


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

def main():
    st.title("Consumer Future-Making Analyzer")
    st.markdown(HOMEPAGE_DESCRIPTION)
    st.divider()

    api_key = None
    try:
        api_key = st.secrets["openai_api_key"]
    except Exception:
        with st.expander("API Settings -- click to configure", expanded=True):
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown("---")
    st.markdown("## Step 1 -- Choose the Analysis Mode")
    mode = render_mode_selector()

    # ═══════════════════════════════════════
    # MODE 1: SINGLE COMMENT
    # ═══════════════════════════════════════
    if mode == MODE_SINGLE:
        st.markdown("---")
        st.markdown("## Step 2 -- Define the Prescribed Future and Enter the Focal Comment")

        prescribed_future = st.text_area(
            "Prescribed future",
            placeholder="Describe the specific policy or market intervention and the future it prescribes.",
            height=85
        )

        st.markdown("**Focal comment**")
        input_method = st.radio("Input method:", ["Type or paste text", "Upload a .txt file"], horizontal=True)
        focal_text = ""
        if input_method == "Type or paste text":
            focal_text = st.text_area(
                "Focal comment", placeholder="Paste or type the consumer comment, response, or post to classify.",
                height=180, label_visibility="collapsed"
            )
        else:
            uploaded_file = st.file_uploader("Upload .txt file:", type=["txt"])
            if uploaded_file:
                focal_text = uploaded_file.read().decode("utf-8")
                st.success(f"Uploaded: {len(focal_text):,} characters")

        st.markdown("---")
        st.markdown("## Step 3 -- Enter Interpretive Context")
        st.write(
            "Context is optional but can help interpret what the focal comment "
            "is responding to. The application classifies only the focal "
            "comment; contextual text is used only to interpret it."
        )
        labels = [c[0] for c in CONTEXT_TYPE_CHOICES]
        chosen_label = st.selectbox("Context type", labels, index=0)
        chosen_context_type = next(c[1] for c in CONTEXT_TYPE_CHOICES if c[0] == chosen_label)
        chosen_help = next(c[2] for c in CONTEXT_TYPE_CHOICES if c[0] == chosen_label)
        st.caption(chosen_help)

        context_text = ""
        if chosen_context_type != "NONE":
            context_text = st.text_area(
                "Context text",
                placeholder="Paste the relevant context here (e.g., the parent comment, nearby comments, the original post, or the consultation prompt).",
                height=100
            )

        is_consultation = (chosen_context_type == "CONSULTATION_PROMPT")
        effective_context_type = chosen_context_type if context_text.strip() else "NONE"

        st.markdown("---")
        st.markdown("## Step 4 -- Analyze the Comment")
        ready = bool(api_key and focal_text.strip() and prescribed_future.strip())
        if not prescribed_future.strip():
            st.warning("Please define the prescribed future in Step 2.")
        if not focal_text.strip():
            st.warning("Please enter a focal comment in Step 2.")
        if not api_key:
            st.warning("Please configure your OpenAI API key above.")

        if st.button("Analyze Comment", type="primary", use_container_width=True, disabled=not ready):
            with st.spinner("Analyzing with the framework's definitions..."):
                try:
                    result = analyze_comment(
                        prescribed_future.strip(), focal_text.strip(), context_text.strip(),
                        effective_context_type, is_consultation, api_key,
                        allow_secondary_classifications=False
                    )
                    st.divider()
                    st.markdown("### Analysis Results")
                    show_results(result, prescribed_future.strip(), allow_secondary_display=False)
                except openai.AuthenticationError:
                    st.error("Invalid API key.")
                except openai.RateLimitError:
                    st.error("Rate limit reached. Please wait a moment.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ═══════════════════════════════════════
    # MODE 2: DOCUMENT / CORPUS ANALYSIS
    # ═══════════════════════════════════════
    else:
        if "map_step2_confirmed" not in st.session_state:
            st.session_state["map_step2_confirmed"] = False

        st.markdown("---")
        st.markdown("## Step 2 -- Define the Prescribed Future and Provide Comments")
        st.write(
            "Upload or paste consumer comments, consultation responses, forum "
            "posts, or social-media conversations. Each focal comment is "
            "classified using available conversational or consultation context."
        )

        prescribed_future_doc = st.text_area(
            "Prescribed future",
            placeholder="Describe the specific policy or market intervention and the future it prescribes.",
            height=85
        )

        data_structure = st.radio(
            "Data structure:",
            ["Unstructured text (paste or upload .txt/.md/.pdf)",
             "Structured comments file (.csv with comment_text, and optionally thread_id / comment_id / parent_comment_id / author / timestamp)"],
            horizontal=False
        )

        raw_text = ""
        csv_df = None
        if data_structure.startswith("Structured"):
            uploaded_csv = st.file_uploader("Upload structured .csv file:", type=["csv"])
            if uploaded_csv:
                try:
                    csv_df = pd.read_csv(uploaded_csv)
                    st.success(f"Loaded {len(csv_df)} rows from '{uploaded_csv.name}'.")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
        else:
            doc_input_method = st.radio("Input method:", ["Upload file (.txt, .md, .pdf)", "Paste text"], horizontal=True)
            if doc_input_method == "Upload file (.txt, .md, .pdf)":
                uploaded_doc = st.file_uploader("Upload document:", type=["txt", "md", "pdf"])
                if uploaded_doc:
                    if uploaded_doc.name.lower().endswith(".pdf"):
                        try:
                            from pypdf import PdfReader
                            with st.spinner("Extracting text from PDF..."):
                                reader = PdfReader(uploaded_doc)
                                raw_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                        except ImportError:
                            st.error("PDF support requires 'pypdf'. Add it to requirements.txt, or paste text instead.")
                    else:
                        raw_text = uploaded_doc.read().decode("utf-8", errors="ignore")
                    if raw_text:
                        st.success(f"Extracted {len(raw_text):,} characters from '{uploaded_doc.name}'")
            else:
                raw_text = st.text_area(
                    "Comments",
                    placeholder="Paste comments here, one per paragraph, separated by a blank line. "
                                "Manuscript-style exchanges labeled 'User 1:', 'User 2:', etc. are "
                                "automatically detected as a shared conversation.",
                    height=250
                )

        if data_structure.startswith("Structured"):
            valid_step2 = (csv_df is not None) and has_comment_text_column(csv_df)
        else:
            valid_step2 = bool(raw_text.strip()) and (
                len(build_comment_records_from_paragraphs(raw_text)) > 0
                or detect_user_labeled_exchange(raw_text) >= 2
            )

        next_disabled = not (prescribed_future_doc.strip() and valid_step2)
        if next_disabled:
            if not prescribed_future_doc.strip():
                st.caption("Enter a prescribed future to continue.")
            elif not valid_step2:
                st.caption("Provide comments (a valid comment-text column, or nonempty pasted/uploaded text) to continue.")

        if st.button("Next", type="primary", use_container_width=True, disabled=next_disabled):
            st.session_state["map_step2_confirmed"] = True
            st.session_state["map_prescribed_future"] = prescribed_future_doc.strip()
            st.session_state["map_data_structure"] = data_structure
            st.session_state["map_raw_text"] = raw_text
            st.session_state["map_csv_df"] = csv_df
            for key in ("doc_results", "doc_prescribed_future", "doc_total_detected", "doc_sampling_description"):
                st.session_state.pop(key, None)
            st.rerun()

        if st.session_state.get("map_step2_confirmed"):
            st.markdown("---")
            if st.button("Edit Step 2 input (prescribed future or comments)"):
                st.session_state["map_step2_confirmed"] = False
                for key in ("doc_results", "doc_prescribed_future", "doc_total_detected", "doc_sampling_description"):
                    st.session_state.pop(key, None)
                st.rerun()

            st.markdown("## Step 3 -- Configure Comment Boundaries and Context")
            st.caption(
                "When thread or parent metadata is available, the application "
                "automatically uses the parent comment, nearby thread comments, "
                "the original post, or the consultation prompt as interpretive "
                "context for each focal comment."
            )

            stored_prescribed_future = st.session_state.get("map_prescribed_future", "")
            stored_data_structure = st.session_state.get("map_data_structure", "")
            stored_raw_text = st.session_state.get("map_raw_text", "")
            stored_csv_df = st.session_state.get("map_csv_df")

            records = []
            is_consultation_mode = False
            consultation_prompt_text = ""

            if stored_data_structure.startswith("Structured") and stored_csv_df is not None:
                records = build_comment_records_from_csv(stored_csv_df)
                n_with_parent = sum(1 for r in records if r.get("parent_comment_id"))
                n_threads = len(set(r["thread_id"] for r in records))
                st.info(f"Parsed {len(records)} comments across {n_threads} thread(s); {n_with_parent} comments have a recorded parent_comment_id.")
            elif stored_raw_text.strip():
                id_hits = len(re.findall(r'\b\d{6,7}\s+(?:Name\s+withheld|[A-Z][a-z]+)', stored_raw_text))
                looks_like_consultation = id_hits >= 5
                user_label_hits = detect_user_labeled_exchange(stored_raw_text)
                looks_like_labeled_exchange = user_label_hits >= 2

                boundary_options = ["One comment per paragraph (default)", "Custom separator"]
                if looks_like_labeled_exchange:
                    boundary_options.insert(
                        0, f"Manuscript-style exchange ({user_label_hits} 'User N:' labels detected)"
                    )
                if looks_like_consultation:
                    boundary_options.insert(0, f"Public consultation responses (auto-detected {id_hits} respondent IDs)")

                boundary_choice = st.selectbox("Comment boundary detection:", boundary_options)

                if boundary_choice.startswith("Public consultation"):
                    is_consultation_mode = True
                    records = build_comment_records_from_consultation(stored_raw_text)
                    consultation_prompt_text = st.text_area(
                        "Consultation question / policy proposal (used as context for every response):",
                        height=90,
                        placeholder="Describe the official question or policy proposal that respondents were answering."
                    )
                    st.info(f"Extracted {len(records)} individual consultation responses, each treated as one focal response.")
                elif boundary_choice.startswith("Manuscript-style exchange"):
                    records = build_comment_records_from_user_labels(stored_raw_text)
                    st.info(
                        f"Detected {len(records)} labeled comments sharing one conversation. "
                        f"They will be treated as a thread window for interpretive context "
                        f"(no parent-reply relationship is assumed unless explicitly specified). "
                        f"Any trailing manuscript orientation labels (e.g., '(Catalyzer "
                        f"Orientation)') have been stripped from the text sent for analysis."
                    )
                elif boundary_choice == "Custom separator":
                    separator = st.text_input("Comment separator (exact string used to split comments):", value="---")
                    records = build_comment_records_from_paragraphs(stored_raw_text, separator=separator)
                    st.caption(
                        "This input does not contain explicit thread/parent metadata. "
                        "Nearby comments will be used as approximate context; true "
                        "reply relationships could not be reconstructed from "
                        "unstructured text."
                    )
                else:
                    records = build_comment_records_from_paragraphs(stored_raw_text)
                    st.caption(
                        "This input does not contain explicit thread/parent metadata. "
                        "Each blank-line-separated paragraph is treated as one "
                        "comment, and nearby comments are used as approximate "
                        "context; true reply relationships could not be "
                        "reconstructed from unstructured text."
                    )

                if records:
                    st.info(f"{len(records)} analyzable comments detected.")
                    with st.expander(f"Preview first comments (of {len(records)} total)"):
                        for rec in records[:10]:
                            st.caption(f"[{rec['comment_id']}] {rec['comment_text'][:200]}{'...' if len(rec['comment_text']) > 200 else ''}")
                else:
                    st.warning("No analyzable comments found with the current boundary method.")

            total_detected = len(records)

            if total_detected > 0:
                st.markdown("---")
                st.markdown("## Step 4 -- Run the Analysis")

                max_possible = max(1, min(total_detected, 300))
                default_val = min(30, max_possible)
                max_comments = st.slider(
                    "Number of comments to analyze (evenly sampled across the full set if fewer than all are selected)",
                    min_value=1, max_value=max_possible, value=default_val
                )
                est_seconds = round(max_comments / DOC_MAX_WORKERS * 2.5)
                est_cost = round(max_comments * 0.00075, 3)
                st.caption(f"Estimated time: ~{est_seconds}s | API calls: {max_comments} (parallelized, {DOC_MAX_WORKERS} at a time) | Estimated cost: ~${est_cost}")

                if max_comments >= total_detected:
                    sampling_preview = "Full set of analyzable comments."
                else:
                    sampling_preview = f"Evenly distributed sample of {max_comments} from {total_detected} analyzable comments."
                st.caption(f"Sampling method: {sampling_preview}")

                run_doc_analysis = st.button("Analyze Comments", type="primary", use_container_width=True, disabled=not api_key)
                if not api_key:
                    st.warning("Please configure your OpenAI API key above.")

                if run_doc_analysis:
                    by_id, thread_order = index_threads(records)

                    if max_comments >= total_detected:
                        sample_indices = list(range(total_detected))
                        sampling_description = "Full set of analyzable comments."
                    else:
                        sample_indices = compute_evenly_spaced_sample_indices(total_detected, max_comments)
                        sampling_description = f"Evenly distributed sample of {len(sample_indices)} from {total_detected} analyzable comments."

                    prepared_records = []
                    for idx in sample_indices:
                        rec = records[idx]
                        context_text, context_type, context_available = build_context(
                            rec, by_id, thread_order, consultation_prompt_text, is_consultation_mode
                        )
                        prepared_records.append({
                            **rec,
                            "context_text": context_text, "context_type": context_type,
                            "context_available": context_available, "is_consultation": is_consultation_mode,
                        })

                    progress_bar = st.progress(0, text="Starting analysis...")
                    doc_results = analyze_document(prepared_records, stored_prescribed_future, api_key, progress_bar)
                    progress_bar.empty()

                    st.session_state["doc_results"] = doc_results
                    st.session_state["doc_prescribed_future"] = stored_prescribed_future
                    st.session_state["doc_total_detected"] = total_detected
                    st.session_state["doc_sampling_description"] = sampling_description

        if "doc_results" in st.session_state:
            st.markdown("---")
            st.markdown("## Step 5 -- Review Results")
            show_document_summary(
                st.session_state["doc_results"],
                st.session_state.get("doc_prescribed_future", ""),
                total_detected=st.session_state.get("doc_total_detected"),
                sampling_description=st.session_state.get("doc_sampling_description", "")
            )
            if st.button("Clear results"):
                del st.session_state["doc_results"]
                st.rerun()

    # ─────────────────────────────────────────
    # ADVANCED / DEVELOPER TOOLS
    # ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("Advanced / Developer Tools"):
        st.markdown("#### Coding Consistency Check")
        st.caption(
            "Agreement with built-in benchmark examples (quoted or "
            "paraphrased from the manuscript's dataset) tests whether the "
            "current prompt reproduces predetermined coding decisions for "
            "activity, activity performance/subtype, orientation, and "
            "Negotiation recall. This is an internal consistency check for "
            "an interactive demonstration tool -- it does NOT constitute "
            "empirical validation, intercoder reliability, or evidence of "
            "generalizability."
        )
        if st.button("Run Coding Consistency Check"):
            if not api_key:
                st.warning("Configure your API key above first.")
            else:
                with st.spinner("Running consistency check across benchmark examples..."):
                    report = run_consistency_suite(api_key)
                if report["results"]:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Overall exact-match agreement", f"{report['overall_agreement']*100:.1f}%")
                    m2.metric("Activity agreement", f"{report['overall_activity_agreement']*100:.1f}%")
                    m3.metric("Subtype agreement", f"{report['overall_subtype_agreement']*100:.1f}%")
                    m4.metric("Orientation agreement", f"{report['overall_orientation_agreement']*100:.1f}%")
                    if report.get("negotiation_recall") is not None:
                        st.metric(
                            f"Negotiation recall ({report['n_negotiation_examples']} manuscript examples)",
                            f"{report['negotiation_recall']*100:.1f}%"
                        )
                    for r in report["results"]:
                        status = "PASS" if r["match"] else "FAIL"
                        with st.expander(f"[{status}] {r['example']}"):
                            st.write("Expected (dominant):", r["expected"])
                            st.write("Predicted (dominant):", r["predicted"])
                            st.write(
                                f"Activity match: {r['activity_match']} | "
                                f"Subtype match: {r['subtype_match']} | "
                                f"Orientation match: {r['orientation_match']}"
                            )
                            if r.get("secondary_expected"):
                                st.write(f"Secondary check [{'PASS' if r.get('secondary_match') else 'FAIL'}]: expected {r['secondary_expected']}")
                            if r.get("error"):
                                st.error(r["error"])
                else:
                    st.info("No labeled benchmark examples found.")


if __name__ == "__main__":
    main()
