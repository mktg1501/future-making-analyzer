"""
Consumer Future-Making Analyzer - motor de análisis.

Contiene la clasificación, la construcción del contexto, la lectura de datos
y la agregación. La interfaz (app.py) solo importa desde este módulo.

Uso desde la terminal:
    python motor.py                 -> pruebas estructurales (sin API)
    python motor.py --consistencia  -> chequeo de consistencia (usa la API)
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
# CONFIGURACIÓN
# ─────────────────────────────────────────
PAPER_TITLE = "Futures in the Making: How Consumers Respond to Future-Oriented Interventions"
PAPER_URL = "REPLACE_WITH_YOUR_DOI_OR_URL"  # Reemplazar por el DOI o enlace real

MODEL_NAME = "gpt-4o-mini"
DOC_MAX_WORKERS = 5
MAX_RETRIES = 3
MAX_COMMENTS = int(os.environ.get("MAX_COMENTARIOS", "300"))
EST_COST_PER_COMMENT = 0.00075
MAX_SECONDARY = 2
DEFAULT_THREAD = "_default_thread_"   # comentarios sin hilo conocido
LABELED_THREAD = "_labeled_exchange_1"  # conversación con etiquetas "User N:"
ORIENTATION_ORDER = ["CATALYZER", "AMBIVALENT", "RESISTANT", "EXPANDER"]

EXAMPLE_DISCLAIMER = (
    "Los ejemplos del paper también aparecen en las instrucciones del modelo. "
    "Sirven para ilustrar cómo funciona la herramienta, no para medir su precisión."
)

# ─────────────────────────────────────────
# NOTAS DE INTERPRETACIÓN
# ─────────────────────────────────────────
INTERPRETIVE_USE_NOTE = (
    "Nota de uso: las actividades de futuro-making son interdependientes, entrelazadas "
    "y recursivas, no secuenciales. Las orientaciones son formas diagnósticas y socialmente "
    "patronadas de practicar el futuro-making, no tipos fijos de consumidor ni segmentos de "
    "mercado. Esta aplicación es una herramienta interactiva de demostración, no un "
    "instrumento diagnóstico validado. Revisa los resultados junto con el comentario completo, "
    "el contexto disponible y la evidencia conductual pertinente."
)

HOMEPAGE_DESCRIPTION = """
Usa esta aplicación para apoyar el mapeo del futuro-making de consumidores frente a una
intervención de política o de mercado, según el marco de **"Futures in the Making: How
Consumers Respond to Future-Oriented Interventions."**

Puedes escribir o pegar comentarios, respuestas a consultas, publicaciones de foros o
conversaciones en redes sociales. La aplicación clasifica cada **comentario focal** usando,
como contexto interpretativo, el comentario padre, comentarios cercanos, la publicación
original o la pregunta de la consulta.

**Alcance:**

* El marco se desarrolló con investigación cualitativa sobre intervenciones de vehículos de
  cero emisiones (ZEV) en Australia. Los autores esperan que se aplique a otros contextos.
* El tipo de intervención (Fija, Acotada, Flexible o Abierta) depende de su alcance y de su
  grado de prescripción, no de su dominio general.
* Las orientaciones (Catalyzer, Ambivalent, Resistant y Expander) son formas diagnósticas,
  **no** tipos fijos de consumidor ni segmentos.
* Los documentos institucionales se usan para definir el futuro prescrito o el contexto de
  una consulta. No se clasifican como orientaciones de consumidores.
* Los desafíos del futuro-making (Evaluaciones Enredadas, Negociaciones Confrontacionales y
  Actuaciones en Competencia) y los Futuros Frágiles son condiciones cualitativas que surgen
  de patrones entre discursos, prácticas y actores vinculados. Nunca se infieren de un solo
  comentario ni reciben una puntuación o probabilidad automática.
"""

# ─────────────────────────────────────────
# TIPOS DE CONTEXTO Y UTILIDADES
# ─────────────────────────────────────────
VALID_CONTEXT_TYPES = {"PARENT_REPLY", "THREAD_WINDOW", "ORIGINAL_POST", "CONSULTATION_PROMPT", "NONE"}

CONTEXT_TYPE_CHOICES = [
    ("Sin contexto", "NONE",
     "Usar cuando el comentario focal debe interpretarse por sí solo."),
    ("Comentario padre", "PARENT_REPLY",
     "El comentario específico al que responde directamente el comentario focal."),
    ("Ventana de hilo", "THREAD_WINDOW",
     "Un conjunto pequeño de comentarios cercanos de la misma conversación."),
    ("Publicación original", "ORIGINAL_POST",
     "La publicación, pie de foto, pregunta o tema que inició la conversación."),
    ("Pregunta de consulta", "CONSULTATION_PROMPT",
     "La pregunta oficial, propuesta o descripción de política a la que responde la consulta."),
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
# PROMPT DEL SISTEMA (clasificación)
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
# ORIENTACIONES Y ACTIVIDADES
# ─────────────────────────────────────────
ORIENTATIONS = {
    "CATALYZER": {
        "color": "#27AE60", "bg": "#EAFAF1",
        "goal": "Accelerar el cambio hacia el futuro prescrito",
        "narrative": "Urgency Narrative",
        "temporality": "El futuro está cerca: el cambio ocurre ahora",
        "activities": "Simplify - Advocate - Accelerate",
    },
    "AMBIVALENT": {
        "color": "#D68910", "bg": "#FEFDE7",
        "goal": "Frenar el ritmo del cambio, posponer decisiones, equilibrar riesgos y beneficios",
        "narrative": "Pragmatic Narrative",
        "temporality": "El futuro es contingente: el cambio es incierto",
        "activities": "Stall - Question - Delay",
    },
    "RESISTANT": {
        "color": "#C0392B", "bg": "#FDEDEC",
        "goal": "Disputar el futuro prescrito y proteger el statu quo",
        "narrative": "Control Narrative",
        "temporality": "El futuro es distante: no hay cambio",
        "activities": "Avoid - Reject - Prevent",
    },
    "EXPANDER": {
        "color": "#7D3C98", "bg": "#F4ECF7",
        "goal": "Ampliar el futuro prescrito y proponer trayectorias alternativas",
        "narrative": "Bigger Picture Narrative",
        "temporality": "El futuro es distante: el cambio debe ser más amplio",
        "activities": "Complexify - Contest - Reroute",
    },
}

ACTIVITY_META = {
    "EVALUATION": {
        "definition": "Referencias a cómo los consumidores dan sentido al futuro prescrito.",
        "subtypes": {"SIMPLIFY": "CATALYZER", "STALL": "AMBIVALENT", "AVOID": "RESISTANT", "COMPLEXIFY": "EXPANDER"},
    },
    "NEGOTIATION": {
        "definition": "La actividad mediante la cual los consumidores intentan dar forma a trayectorias colectivas hacia un futuro preferido.",
        "subtypes": {"ADVOCATE": "CATALYZER", "QUESTION": "AMBIVALENT", "REJECT": "RESISTANT", "CONTEST": "EXPANDER"},
    },
    "ENACTMENT": {
        "definition": "Referencias a cómo los consumidores dan forma a futuros mediante cambios imaginados, planificados o reales en sus prácticas.",
        "subtypes": {"ACCELERATE": "CATALYZER", "DELAY": "AMBIVALENT", "PREVENT": "RESISTANT", "REROUTE": "EXPANDER"},
    },
}

SUBTYPE_TO_ACTIVITY = {st: act for act, m in ACTIVITY_META.items() for st in m["subtypes"]}
SUBTYPE_TO_ORIENTATION = {st: ori for m in ACTIVITY_META.values() for st, ori in m["subtypes"].items()}
ORIENTATION_TO_SUBTYPE = {(act, ori): st for act, m in ACTIVITY_META.items() for st, ori in m["subtypes"].items()}

# ─────────────────────────────────────────
# DESAFÍOS (solo a nivel de corpus; nunca de un comentario)
# ─────────────────────────────────────────
CHALLENGE_DEFINITIONS = {
    "CONVOLUTED_EVALUATIONS": {
        "label": "Evaluaciones enredadas", "activity": "EVALUATION",
        "definition": (
            "Surgen cuando los consumidores evalúan el futuro prescrito con más o menos "
            "certeza y minuciosidad. Cuando lo evalúan con supuestos, evidencia y horizontes "
            "temporales divergentes, las evaluaciones se vuelven enredadas."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "label": "Negociaciones confrontacionales", "activity": "NEGOTIATION",
        "definition": (
            "Surgen cuando los consumidores negocian sus futuros preferidos. Cuando abogan, "
            "cuestionan, rechazan o disputan el futuro prescrito en relación con otros, "
            "las negociaciones se vuelven confrontacionales."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "label": "Actuaciones en competencia", "activity": "ENACTMENT",
        "definition": (
            "Surgen cuando los consumidores actúan sobre futuros preferidos distintos mediante "
            "sus prácticas actuales. Cuando aceleran, retrasan, previenen o reencaminan el "
            "futuro prescrito, las actuaciones se vuelven competitivas."
        ),
    },
}

FRAGILE_FUTURES_DEFINITION = (
    "Futuros frágiles: futuros preferidos múltiples, volátiles y en conflicto que pueden "
    "interferir con la materialización del futuro prescrito. Es una condición teórica "
    "cualitativa que surge de desempeños orientados que interfieren entre sí. NO es un "
    "puntaje, una probabilidad ni una clasificación de un solo comentario."
)

# ─────────────────────────────────────────
# CONTENIDO DE HOJAS DE RUTA (referencia fija; no lo genera el LLM)
# ─────────────────────────────────────────
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
    ("1", EXPECTED_POLICY_STEP_TITLES[0], "Make explicit what future the intervention seeks to prescribe."),
    ("2", EXPECTED_POLICY_STEP_TITLES[1],
     "Identify how people adopting different orientations evaluate, negotiate, and enact (or not) the "
     "prescribed future. Consider narratives, goals, emotions, and temporalities to identify orientations, "
     "rather than segments."),
    ("3", EXPECTED_POLICY_STEP_TITLES[2],
     "Identify which future-making challenges are the most pressing. Monitor where different performances "
     "of future-making interfere with one another."),
    ("4", EXPECTED_POLICY_STEP_TITLES[3], "Match support to the future-making orientations."),
    ("5", EXPECTED_POLICY_STEP_TITLES[4],
     "Provide infrastructure and build capabilities needed to navigate the change."),
    ("6", EXPECTED_POLICY_STEP_TITLES[5],
     "Is the intervention accurate and fair? Do consumers understand it? Who benefits? Who is excluded? "
     "Are alternative pathways emerging?"),
    ("7", EXPECTED_POLICY_STEP_TITLES[6], "Treat the prescribed future as revisable."),
]

MANAGER_ROADMAP_STEPS = [
    ("1", EXPECTED_MANAGER_STEP_TITLES[0], "Identify the future prescribed by the intervention."),
    ("2", EXPECTED_MANAGER_STEP_TITLES[1],
     "Identify how people adopting different orientations evaluate, negotiate, and enact (or not) the "
     "prescribed future. Consider narratives, goals, emotions, and temporalities to identify orientations, "
     "rather than segments."),
    ("3", EXPECTED_MANAGER_STEP_TITLES[2],
     "Identify which future-making challenges are the most pressing. Monitor where different performances "
     "of future-making interfere with one another."),
    ("4", EXPECTED_MANAGER_STEP_TITLES[3],
     "Choose strategy based on the orientation and the key future-making challenges."),
    ("5", EXPECTED_MANAGER_STEP_TITLES[4],
     "Do not rely on a single persuasive frame. Universal claims (\"the change is inevitable,\" "
     "\"everyone benefits\") may mobilize consumers with catalyzer orientations while intensifying "
     "resistance and confrontation elsewhere."),
    ("6", EXPECTED_MANAGER_STEP_TITLES[5],
     "Place support at the touchpoints where consumers must adjust practices: onboarding, everyday "
     "workflows, escalation points, training, and appeals. Provide adjustable involvement, human "
     "assistance, and easy ways to pause, reverse, or modify adoption."),
]

STEP2_ORIENTATION_LENS = {
    "CATALYZER": "Esta orientación ve el futuro prescrito como urgente, deseable y ya en marcha.",
    "AMBIVALENT": "Esta orientación ve el futuro prescrito como valioso, pero las condiciones aún no están listas.",
    "RESISTANT": "Esta orientación ve el futuro prescrito como una amenaza para su autonomía, identidad o derechos.",
    "EXPANDER": "Esta orientación ve el futuro prescrito como un encuadre demasiado estrecho.",
}

POLICY_STEP2_SIGNALS = {
    "CATALYZER": [
        "Busca lenguaje que enfatice urgencia o inevitabilidad.",
        "Registra la adopción voluntaria temprana.",
    ],
    "AMBIVALENT": [
        "Busca lenguaje condicional (\"lo haría, pero\") y pruebas sin conversión.",
        "Diagnostica la condición específica sin resolver.",
    ],
    "RESISTANT": [
        "Busca lenguaje sobre coerción, prohibiciones, pérdida de libertad o desconfianza.",
        "Registra cancelaciones, opt-outs y oposición organizada.",
    ],
    "EXPANDER": [
        "Busca afirmaciones de que la intervención no resuelve el problema de fondo o pide un cambio sistémico.",
        "Registra visiones de un cambio más amplio.",
    ],
}

MANAGER_STEP2_SIGNALS = {
    "CATALYZER": [
        "Monitorea el lenguaje de urgencia e inevitabilidad, la participación en pilotos y la defensa pública.",
        "Identifica los recursos que facilitan la adopción temprana.",
    ],
    "AMBIVALENT": [
        "Monitorea el lenguaje condicional, como \"lo haría, pero…\" o \"todavía no\".",
        "Registra señales de duda.",
        "Identifica pruebas sin adopción posterior.",
    ],
    "RESISTANT": [
        "Monitorea el lenguaje de coerción y vigilancia, los opt-outs y la oposición organizada.",
        "Distingue la oposición ideológica de la desventaja material.",
    ],
    "EXPANDER": [
        "Observa \"esto no resuelve el problema real\", la defensa de alternativas colectivas y la participación en infraestructuras alternativas.",
    ],
}

POLICY_STEP3_QUESTIONS = {
    "CONVOLUTED_EVALUATIONS": "¿Hay evidencia, supuestos u horizontes temporales incompatibles que impidan construir sentido compartido?",
    "CONFRONTATIONAL_NEGOTIATIONS": "¿El desacuerdo escala en torno a la autonomía, la equidad, la legitimidad o el encuadre del problema?",
    "COMPETING_ENACTMENTS": "¿Las prácticas que aceleran, retrasan, previenen o reencaminan el cambio crean rutas incompatibles?",
}

MANAGER_STEP3_QUESTIONS = {
    "CONVOLUTED_EVALUATIONS": "¿Los consumidores simplifican, frenan, evitan o complejizan la evaluación del futuro prescrito?",
    "CONFRONTATIONAL_NEGOTIATIONS": "¿La defensa, las preguntas, el rechazo o la impugnación escalan en conflicto?",
    "COMPETING_ENACTMENTS": "¿Los consumidores aceleran, retrasan, previenen o reencaminan el cambio mediante conductas incompatibles?",
}

POLICY_STEP4_OBJECTIVES = {
    "CATALYZER": "Habilitar una aceleración responsable solo cuando pueda demostrarse valor público.",
    "AMBIVALENT": "Convertir la incertidumbre en decisiones seguras.",
    "RESISTANT": "Proteger los derechos de los consumidores y distribuir responsabilidades de forma justa.",
    "EXPANDER": "Considerar futuros alternativos y planificar políticas escalonadas hacia un cambio más amplio.",
}

MANAGER_STEP4_OBJECTIVES = {
    "CATALYZER": {
        "objective": "Convertir el entusiasmo en experimentación creíble y responsable.",
        "avoid": "Afirmaciones de inevitabilidad; tratar a los adoptantes tempranos como prueba de que la transición es fácil para todos.",
    },
    "AMBIVALENT": {
        "objective": "Convertir la incertidumbre general en condiciones específicas y atendibles.",
        "avoid": "Presión y urgencia artificial; tratar la duda como ignorancia o resistencia.",
    },
    "RESISTANT": {
        "objective": "Proteger los derechos de los consumidores y distribuir responsabilidades de forma justa.",
        "avoid": "\"No hay alternativa\"; la burla; la automatización oculta de decisiones.",
    },
    "EXPANDER": {
        "objective": "Incorporar la crítica sistémica y explorar futuros alternativos.",
        "avoid": "Presentar la intervención como solución completa; descartar las críticas.",
    },
}

POLICY_IMPLICATIONS = {
    "CATALYZER": (
        "Estos consumidores pueden describir el futuro prescrito como deseable, inevitable y ya en marcha. "
        "Su experiencia muestra qué facilita hoy la adopción, pero no debe tomarse como prueba de que otros "
        "consumidores puedan o deban seguir el mismo camino."
    ),
    "AMBIVALENT": (
        "Estos consumidores pueden considerar el futuro prescrito potencialmente beneficioso, pero ven "
        "condiciones particulares sin resolver. La ambivalencia puede ser constructiva: revela puntos débiles "
        "y exige infraestructura más sólida antes de una adopción masiva."
    ),
    "RESISTANT": (
        "Estos consumidores pueden presentar el futuro prescrito como coercitivo, inequitativo o amenazante "
        "para su autonomía, identidad o prácticas establecidas. Examinar esta orientación ayuda a entender "
        "las razones del rechazo."
    ),
    "EXPANDER": (
        "Estos consumidores pueden argumentar que la intervención atiende un problema estrechamente "
        "encuadrado y deja sin resolver problemas más amplios, lo que revela temas que quizá no se "
        "consideraron al diseñarla."
    ),
}

MANAGER_IMPLICATIONS = {
    "CATALYZER": (
        "Los Catalyzer simplifican la evaluación, defienden el futuro prescrito y aceleran la adopción. Su "
        "entusiasmo puede normalizar nuevas prácticas, pero puede ocultar los recursos y competencias que "
        "hicieron posible la adopción temprana."
    ),
    "AMBIVALENT": (
        "Los consumidores Ambivalent ven un valor potencial, pero consideran que ciertas condiciones no están "
        "resueltas. Su duda puede señalar barreras atendibles, no oposición generalizada."
    ),
    "RESISTANT": (
        "Los consumidores Resistant perciben amenazas a su autonomía, identidad, derechos o prácticas "
        "establecidas. Evitan la evaluación, rechazan el futuro prescrito y buscan impedir su materialización."
    ),
    "EXPANDER": (
        "Los Expander complejizan la evaluación, impugnan el futuro prescrito y reencaminan la adopción hacia "
        "alternativas más amplias. Revelan propuestas de valor más amplias y trayectorias alternativas o "
        "complementarias."
    ),
}

POLICY_ORIENTATION_GUIDANCE = {
    o: {
        "lens": STEP2_ORIENTATION_LENS[o],
        "implications": POLICY_IMPLICATIONS[o],
        "monitor": POLICY_STEP2_SIGNALS[o],
        "objective": POLICY_STEP4_OBJECTIVES[o],
    }
    for o in ORIENTATION_ORDER
}

MANAGER_ORIENTATION_GUIDANCE = {
    o: {
        "lens": STEP2_ORIENTATION_LENS[o],
        "implications": MANAGER_IMPLICATIONS[o],
        "monitor": MANAGER_STEP2_SIGNALS[o],
        "objective": MANAGER_STEP4_OBJECTIVES[o]["objective"],
        "avoid": MANAGER_STEP4_OBJECTIVES[o]["avoid"],
    }
    for o in ORIENTATION_ORDER
}

CHALLENGE_POLICY_GUIDANCE = {
    "CONVOLUTED_EVALUATIONS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["CONVOLUTED_EVALUATIONS"],
        "action": (
            "Identifica la evidencia, los supuestos o los horizontes temporales incompatibles detrás de "
            "evaluaciones divergentes, y apoya la construcción de sentido compartido (por ejemplo, una "
            "evaluación común de precisión, compensaciones y efectos distributivos), en lugar de tratar una "
            "evaluación como correcta y las demás como desinformadas."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["CONFRONTATIONAL_NEGOTIATIONS"],
        "action": (
            "Crea oportunidades legítimas y accesibles para comparar, cuestionar, impugnar y negociar. "
            "Reconsidera el encuadre o las medidas de apoyo cuando el conflicto escale, en lugar de depender "
            "de una persuasión unilateral más fuerte o de presentar el futuro prescrito como inevitable."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "diagnostic_question": POLICY_STEP3_QUESTIONS["COMPETING_ENACTMENTS"],
        "action": (
            "Identifica cómo las prácticas actuales aceleran, retrasan, previenen o reencaminan el futuro "
            "prescrito. Determina si estas rutas pueden coexistir o interferir entre sí, y coordina el apoyo "
            "entre rutas viables sin tratar toda divergencia como incumplimiento."
        ),
    },
}

CHALLENGE_MANAGER_GUIDANCE = {
    "CONVOLUTED_EVALUATIONS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["CONVOLUTED_EVALUATIONS"],
        "action": (
            "Monitorea cómo los consumidores evalúan la viabilidad y deseabilidad del futuro prescrito en los "
            "puntos de contacto relevantes: afirmaciones contradictorias, preguntas sin resolver, comparaciones "
            "repetidas, solicitudes de evidencia, abandono durante el alta y brechas entre lo que promete la "
            "organización y la experiencia del consumidor."
        ),
    },
    "CONFRONTATIONAL_NEGOTIATIONS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["CONFRONTATIONAL_NEGOTIATIONS"],
        "action": (
            "Monitorea los intercambios en los que los consumidores defienden futuros preferidos, responden a "
            "afirmaciones competidoras o intentan influir en trayectorias colectivas. Indicadores: defensa, "
            "preguntas persistentes, rechazo explícito, impugnación sistémica, discusión polarizada y disputas "
            "sobre autonomía o legitimidad."
        ),
    },
    "COMPETING_ENACTMENTS": {
        "diagnostic_question": MANAGER_STEP3_QUESTIONS["COMPETING_ENACTMENTS"],
        "action": (
            "Monitorea lo que los consumidores hacen en el presente: adopción rápida, pruebas sin continuidad, "
            "uso retrasado, negativas, dependencia continua de prácticas existentes y movimiento hacia rutas "
            "alternativas. Estas prácticas son evidencia de los futuros preferidos que los consumidores materializan."
        ),
    },
}

MANAGER_MESSAGING_NOTE = (
    "No dependas de un solo marco persuasivo. Las afirmaciones universales (\"el cambio es inevitable\", "
    "\"todos ganan\") pueden movilizar a consumidores Catalyzer mientras intensifican la resistencia y la "
    "confrontación en otros grupos."
)

CROSS_ORIENTATION_NOTE = (
    "Verificación de interferencia entre orientaciones: una respuesta dirigida a una orientación puede "
    "intensificar un desafío en otra. Por ejemplo, una campaña de evidencia que tranquiliza a los "
    "consumidores Ambivalent puede profundizar la desconfianza Resistant si ignora la autonomía, o reforzar "
    "la crítica Expander si presenta la intervención como solución completa. Revisa estos efectos antes de "
    "finalizar una respuesta."
)

# ─────────────────────────────────────────
# EJEMPLOS DEL PAPER (demostración y chequeo interno)
# ─────────────────────────────────────────
PF_EV = (
    "Transition all vehicles to Zero Emission Vehicles (ZEV), including electric vehicles, to achieve "
    "Australia's net-zero emissions targets, as prescribed by Australia's National Electric Vehicle Strategy (2023)"
)

DEMO_EXAMPLES = [
    {
        "id": "joe", "titulo": "Joe (consulta pública): Negociación · Advocate · Catalyzer",
        "comment": (
            "We are already so far behind! We need to sprint to catch up. We should be WORLD LEADERS in "
            "solar and battery manufacturing. Why are we not using our own minerals to make batteries for "
            "EVs on a global scale??"
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "NEGOTIATION", "subtype": "ADVOCATE", "orientation": "CATALYZER"},
        "nota_paper": "Llama a acelerar y a reclutar a otros. La negociación busca movilizar una trayectoria colectiva.",
    },
    {
        "id": "clara", "titulo": "Clara (entrevista): Evaluación · Stall · Ambivalent",
        "comment": (
            "Living in Outback Northwest Queensland there's no charging stations at the time. I did like the "
            "appeal of an electric vehicle mainly because you don't have to put fuel in it, which is great. "
            "But just at the time I went and bought a fairly decent car for five and a half grand. If we went "
            "on a driving holiday, we would take our big car. So if we got an EV it would just be our daily run around."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "EVALUATION", "subtype": "STALL", "orientation": "AMBIVALENT"},
        "nota_paper": "Evalúa el EV según la factibilidad (carga, viajes largos) sin rediseñar sus rutinas: evaluación que frena.",
    },
    {
        "id": "martin", "titulo": "Martin (redes sociales): Negociación · Question · Ambivalent",
        "comment": "Have you thought about what they are gonna do with all the batteries once they expire because they aren't recyclable?",
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "NEGOTIATION", "subtype": "QUESTION", "orientation": "AMBIVALENT"},
        "nota_paper": "Pregunta si el sistema está listo. Es una pregunta que plantea una condición para avanzar.",
    },
    {
        "id": "raj", "titulo": "Raj (video): Negociación · Reject · Resistant",
        "comment": "We don't need politicians and their cronies telling us what sort of car we can have.",
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "NEGOTIATION", "subtype": "REJECT", "orientation": "RESISTANT"},
        "nota_paper": "Rechaza la autoridad que impulsa la intervención y defiende la libertad de elección.",
    },
    {
        "id": "startt", "titulo": "StarT (noticias): Actuación · Prevent · Resistant",
        "comment": (
            "I for one WILL NOT be forced into an elec vehicle and spend half my travel time charging the damn "
            "thing to go to hell."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "ENACTMENT", "subtype": "PREVENT", "orientation": "RESISTANT"},
        "nota_paper": "Declara que no adoptará el EV, es decir, que mantiene su práctica actual y se niega a cambiar.",
    },
    {
        "id": "johnny", "titulo": "Johnny (Facebook): Actuación · Accelerate · Catalyzer",
        "comment": (
            "Toyota is still very much trying to slow down the transition to EVs. We have ordered two Teslas "
            "that will be delivered hopefully this year. We are selling our Prado and it looks like we are "
            "going to sell our last Toyota car."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "ENACTMENT", "subtype": "ACCELERATE", "orientation": "CATALYZER"},
        "nota_paper": "Compra EV y vende sus vehículos a combustión: materializa el futuro prescrito.",
    },
    {
        "id": "peter", "titulo": "Peter (entrevista): Evaluación · Complexify · Expander",
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
            "Evalúa las emisiones incorporadas y decide conservar su auto. La bicicleta de carga es una práctica "
            "secundaria de reroute: al activar clasificaciones secundarias, la herramienta puede detectarla."
        ),
    },
    {
        "id": "phillip", "titulo": "Phillip (consulta pública): Actuación · Reroute · Expander",
        "comment": (
            "I uprooted my life and moved from the Sunshine Coast to Melbourne with some of my strongest "
            "reasoning being the ability to use public transport, ride a bike around and use a car as little as possible."
        ),
        "context": "", "context_type": "NONE",
        "referencia": {"activity": "ENACTMENT", "subtype": "REROUTE", "orientation": "EXPANDER"},
        "nota_paper": "Cambió de lugar de residencia para depender menos del auto: reencamina su práctica hacia un futuro más amplio.",
    },
]

DEMO_BY_ID = {ex["id"]: ex for ex in DEMO_EXAMPLES}

# Conversación del paper (foro). Los vínculos son de adyacencia: cada usuario responde al anterior.
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
# SALIDA Y CONSISTENCIA DEL MODELO
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
    """Corrige una combinación actividad/subtipo/orientación según la matriz de la sección D.
    Devuelve (actividad, subtipo, orientación, notas)."""
    notes = []
    activity = activity if activity in ACTIVITY_META else ""
    subtype = subtype if subtype in SUBTYPE_TO_ACTIVITY else ""
    orientation = orientation if orientation in ORIENTATIONS else ""

    if subtype and activity and SUBTYPE_TO_ACTIVITY[subtype] != activity:
        notes.append(f"actividad corregida de {activity} a {SUBTYPE_TO_ACTIVITY[subtype]}")
        activity = SUBTYPE_TO_ACTIVITY[subtype]
    if subtype and not activity:
        activity = SUBTYPE_TO_ACTIVITY[subtype]
    if subtype and not orientation:
        orientation = SUBTYPE_TO_ORIENTATION[subtype]
        notes.append(f"orientación derivada del subtipo {subtype}")
    if activity and orientation:
        expected = ORIENTATION_TO_SUBTYPE[(activity, orientation)]
        if subtype != expected:
            notes.append(f"subtipo ajustado de {subtype or 'vacío'} a {expected}")
            subtype = expected
    return activity, subtype, orientation, notes


MAX_NOTE_CHARS = 300


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
            notes.append(f"secundaria {i + 1} descartada por estar incompleta")
            continue
        if (a, s, o) == main or any((a, s, o) == (x["activity"], x["activity_subtype"], x["orientation"]) for x in secondary):
            continue
        notes += [f"secundaria {i + 1}: {x}" for x in n]
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
# LLAMADA AL MODELO CON REINTENTOS
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
            logger.warning("Intento %d fallido (%s); se reintenta.", attempt + 1, type(e).__name__)
            time.sleep(2 ** attempt)
    raise RuntimeError(
        "El servicio de análisis no respondió tras varios intentos. Intenta de nuevo en unos minutos."
    ) from last_error


def _parse_json(content: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError("La respuesta del modelo no fue un JSON válido. Intenta de nuevo.") from e
    if not isinstance(data, dict):
        raise ValueError("La respuesta del modelo no tiene la estructura esperada.")
    return data


def analyze_comment(prescribed_future: str, focal_text: str, context_text: str = "",
                    context_type: str = "NONE", is_consultation: bool = False,
                    allow_secondary: bool = True) -> dict:
    focal_text = (focal_text or "").strip()
    if not focal_text:
        raise ValueError("El comentario está vacío.")

    if is_consultation:
        focal_label = "FOCAL RESPONSE TO CLASSIFY"
        context_label = "CONSULTATION/POLICY CONTEXT -- USE FOR INTERPRETATION BUT DO NOT CLASSIFY"
    else:
        focal_label = "FOCAL COMMENT TO CLASSIFY"
        context_label = "CONTEXT (parent comment / nearby comments / original post) -- USE FOR INTERPRETATION BUT DO NOT CLASSIFY"

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
        raise ValueError("El modelo no devolvió una clasificación completa. Intenta de nuevo.")
    if not allow_secondary:
        parsed["secondary_classifications"] = []
    return enforce_context_metadata(parsed, context_available, context_type)


def _safe_analyze(prescribed_future: str, rec: dict) -> dict:
    """Nunca lanza excepciones: devuelve {'_error': mensaje} si algo falla."""
    try:
        return analyze_comment(
            prescribed_future, rec["comment_text"], rec.get("context_text", ""),
            rec.get("context_type", "NONE"), rec.get("is_consultation", False), True,
        )
    except (ValueError, RuntimeError) as e:
        return {"_error": str(e)}
    except Exception:
        logger.exception("Error inesperado al analizar el comentario %s", rec.get("comment_id"))
        return {"_error": "Error inesperado al analizar este comentario (ver registro del servidor)."}


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
# LECTURA DE DATOS Y ESTRUCTURA DE HILOS
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
            # Adyacencia: cada usuario responde al anterior (supuesto explícito, no verificado).
            "parent_comment_id": f"user{kept[pos - 1][0]}" if pos > 0 else None,
            "author": f"User {num}", "timestamp": "",
            "comment_text": cleaned, "original_index": pos,
        })
    return records


def build_comment_records_from_paragraphs(text: str):
    """Devuelve (registros, cantidad_descartada)."""
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
    """Renombra identificadores repetidos. Devuelve cuántos se renombraron."""
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
    """Selecciona comentarios sin partir hilos. Devuelve (seleccionados, omitidos)."""
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
    """Devuelve (texto_de_contexto, tipo_de_contexto, hay_contexto).
    Los hilos solo se usan cuando son reales (no para comentarios sin hilo)."""
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
    return f"Participante {digest}"


# ─────────────────────────────────────────
# AGREGACIÓN (corpus). Nunca diagnostica un desafío automáticamente.
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
    """Cuántas orientaciones aparecen en el conjunto y con qué proporción."""
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
    """Para hilos reales con al menos dos comentarios: orientaciones presentes en cada uno."""
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
    """Marca PARES ENLAZADOS con la misma actividad y distinto desempeño, para revisión humana.
    No diagnostica el desafío ni asigna puntajes."""
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
# PRUEBAS ESTRUCTURALES Y CHEQUEO DE CONSISTENCIA (desde la terminal)
# ─────────────────────────────────────────
REQUIRED_ORIENTATIONS = set(ORIENTATION_ORDER)
REQUIRED_SUBTYPES = {
    "EVALUATION": {"SIMPLIFY", "STALL", "AVOID", "COMPLEXIFY"},
    "NEGOTIATION": {"ADVOCATE", "QUESTION", "REJECT", "CONTEST"},
    "ENACTMENT": {"ACCELERATE", "DELAY", "PREVENT", "REROUTE"},
}
REQUIRED_CHALLENGES = {"CONVOLUTED_EVALUATIONS", "CONFRONTATIONAL_NEGOTIATIONS", "COMPETING_ENACTMENTS"}
OBSOLETE_TERMS_FILE = "terminos_prohibidos.txt"  # opcional: una expresión por línea


def check_obsolete_terms():
    base = Path(__file__).resolve().parent
    list_file = base / OBSOLETE_TERMS_FILE
    if not list_file.exists():
        return True, f"Archivo {OBSOLETE_TERMS_FILE} no encontrado (prueba omitida)"
    terms = [t.strip().lower() for t in list_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    hits = []
    for name in ("motor.py", "app.py"):
        path = base / name
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            hits += [f"{name}: '{t}'" for t in terms if t in text]
    return not hits, "Sin hallazgos" if not hits else "; ".join(hits)


def run_framework_validation() -> list:
    """Devuelve una lista de (nombre, aprobada, detalle). Sin llamadas a la API."""
    tests = []

    def add(name, ok, detail=""):
        tests.append((name, bool(ok), detail))

    add("Exactamente 4 orientaciones", set(ORIENTATIONS) == REQUIRED_ORIENTATIONS,
        f"Encontradas: {sorted(ORIENTATIONS)}")
    for act, subs in REQUIRED_SUBTYPES.items():
        found = set(ACTIVITY_META.get(act, {}).get("subtypes", {}))
        add(f"{act}: 4 subtipos esperados", found == subs, f"Encontrados: {sorted(found)}")
    add("Matriz orientación × actividad completa (12 combinaciones)",
        len(ORIENTATION_TO_SUBTYPE) == 12, f"Encontradas: {len(ORIENTATION_TO_SUBTYPE)}")
    add("Hoja de ruta de política: 7 pasos en orden",
        [t for _, t, _ in POLICY_ROADMAP_STEPS] == EXPECTED_POLICY_STEP_TITLES, "")
    add("Hoja de ruta gerencial: 6 pasos en orden",
        [t for _, t, _ in MANAGER_ROADMAP_STEPS] == EXPECTED_MANAGER_STEP_TITLES, "")
    add("Exactamente 3 desafíos", set(CHALLENGE_DEFINITIONS) == REQUIRED_CHALLENGES,
        f"Encontrados: {sorted(CHALLENGE_DEFINITIONS)}")
    add("Guía de desafíos completa (pregunta y acción)",
        all("diagnostic_question" in CHALLENGE_POLICY_GUIDANCE.get(k, {})
            and "action" in CHALLENGE_POLICY_GUIDANCE.get(k, {})
            and "diagnostic_question" in CHALLENGE_MANAGER_GUIDANCE.get(k, {})
            and "action" in CHALLENGE_MANAGER_GUIDANCE.get(k, {})
            for k in REQUIRED_CHALLENGES), "")
    add("Preguntas del Paso 3 distintas entre hojas de ruta",
        all(POLICY_STEP3_QUESTIONS[k] != MANAGER_STEP3_QUESTIONS[k] for k in REQUIRED_CHALLENGES), "")
    add("Guía de orientaciones completa en ambas hojas",
        all(o in POLICY_ORIENTATION_GUIDANCE and o in MANAGER_ORIENTATION_GUIDANCE
            for o in REQUIRED_ORIENTATIONS), "")
    add("Gerencial incluye 'avoid'; política no lo incluye",
        all("avoid" in MANAGER_ORIENTATION_GUIDANCE[o] for o in REQUIRED_ORIENTATIONS)
        and all("avoid" not in POLICY_ORIENTATION_GUIDANCE[o] for o in REQUIRED_ORIENTATIONS), "")
    add("Ejemplos de demostración: referencias válidas según la matriz",
        all(ORIENTATION_TO_SUBTYPE.get((e["referencia"]["activity"], e["referencia"]["orientation"]))
            == e["referencia"]["subtype"] for e in DEMO_EXAMPLES), "")
    add("Ejemplos de demostración cubren las 4 orientaciones",
        {e["referencia"]["orientation"] for e in DEMO_EXAMPLES} == REQUIRED_ORIENTATIONS, "")
    add("Conversación de demostración: una referencia por orientación",
        {v["orientation"] for v in DEMO_CONVERSATION_REFERENCE.values()} == REQUIRED_ORIENTATIONS
        and len(DEMO_CONVERSATION_REFERENCE) == 4, "")
    ok, detail = check_obsolete_terms()
    add("Sin términos obsoletos en el código", ok, detail)
    return tests


def run_consistency_suite() -> dict:
    """Chequeo de consistencia INTERNA contra los ejemplos del paper.
    Los ejemplos también están en el prompt: no es validación empírica ni de confiabilidad entre codificadores."""
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
        print(f"[{'OK' if ok else 'FALLO'}] {name}" + (f" — {detail}" if detail else ""))
        failed += 0 if ok else 1
    if "--consistencia" in sys.argv:
        report = run_consistency_suite()
        print(f"\nConsistencia interna: {report['overall_agreement']:.0%} "
              "(los ejemplos también están en el prompt; no es validación).")
        for row in report["results"]:
            mark = "✔" if row["match"] else "✘"
            print(f"  {mark} {row['example']}: {row['predicted']} {row['error']}")
    sys.exit(1 if failed else 0)
