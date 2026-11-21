"""Vocabulary of the synthetic scenario generator: personas, noise, node archetypes, condition styles, tags.

Everything here is data (ids, weights, text shown to the generating model). The sampler (``spec.py``) draws from it with a seed; the
generating model only ever writes text for an already drawn specification.
"""

from __future__ import annotations

# Personas
# id -> (role, instruction shown to the generator). role: "seen" may appear in training data, "heldout" never does.
PERSONAS: dict[str, tuple[str, str]] = {
    "neutral": ("seen", "ordinary clear everyday English, no strong register"),
    "formal": ("seen", "polite and formal, full sentences, no contractions"),
    "casual": ("seen", "relaxed everyday chat, contractions, short sentences, an occasional filler like 'hey' or 'so'"),
    "frustrated": (
        "seen",
        "annoyed or impatient, may be curt or sarcastic; mild swearing such as 'damn' or 'crap' is fine",
    ),
    "anxious": (
        "seen",
        "worried and a bit over-explaining, asks for reassurance, may hedge ('I think', 'sorry if this is silly')",
    ),
    "confused": (
        "seen",
        "unsure what to call things, describes the situation vaguely or with the wrong term, asks 'is that even possible'",
    ),
    "hinting": (
        "seen",
        "never states the request directly; describes the situation or the problem and leaves the intent to be inferred",
    ),
    "lay": (
        "seen",
        "ordinary person with no domain knowledge, describes things in plain words instead of the professional vocabulary",
    ),
    "jargon": ("seen", "expert or insider, uses abbreviations and trade vocabulary of the domain correctly"),
    "multi_topic": (
        "seen",
        "mentions a second, side topic or complaint, but the main request is still clearly the target",
    ),
    "slang_heavy": (
        "heldout",
        "heavy colloquial slang and idioms, very informal, swearing allowed in a casual way (e.g. 'freaking', 'hell', 'crap'); "
        "no slurs, no hate, nothing sexual or threatening",
    ),
    "rambling_polite": (
        "heldout",
        "very polite and long-winded, adds personal backstory and tangents before and after the point, "
        "but the request is still recoverable",
    ),
    "non_native": (
        "heldout",
        "plausible second-language English: article and preposition errors, odd word order, simple vocabulary; "
        "meaning must stay understandable",
    ),
    "fragment_texter": (
        "heldout",
        "terse text-message fragments, no full sentences, drops subjects and verbs, as little punctuation as possible",
    ),
}
SEEN_PERSONAS = tuple(p for p, (r, _) in PERSONAS.items() if r == "seen")
HELDOUT_PERSONAS = tuple(p for p, (r, _) in PERSONAS.items() if r == "heldout")

# sampling weights for an evaluation set (heldout styles are over-represented on purpose, so they can be measured)
EVAL_PERSONA_WEIGHTS = {
    "neutral": 0.19,
    "casual": 0.13,
    "formal": 0.07,
    "frustrated": 0.07,
    "anxious": 0.04,
    "confused": 0.04,
    "hinting": 0.04,
    "lay": 0.03,
    "jargon": 0.02,
    "multi_topic": 0.02,
    "slang_heavy": 0.0875,
    "rambling_polite": 0.0875,
    "non_native": 0.0875,
    "fragment_texter": 0.0875,
}
# training sets only use seen personas (heldout styles are kept for evaluation)
TRAIN_PERSONA_WEIGHTS = {
    "neutral": 0.27,
    "casual": 0.19,
    "formal": 0.10,
    "frustrated": 0.10,
    "anxious": 0.06,
    "confused": 0.06,
    "hinting": 0.07,
    "lay": 0.06,
    "jargon": 0.04,
    "multi_topic": 0.05,
}

# Noise (applied by code, see noise.py)
NOISE_SEEN = ("none", "keyboard_typos", "no_punct_lower")
NOISE_HELDOUT = ("text_speak", "autocorrect")
NOISE_DESCRIPTION = {
    "none": "clean text",
    "keyboard_typos": "a few keyboard-adjacent substitutions, dropped, doubled or swapped letters",
    "no_punct_lower": "all lower case, commas and final punctuation removed, apostrophes sometimes dropped",
    "text_speak": "text-message abbreviations (u, 2day, tmrw, pls, thx, b4, w/)",
    "autocorrect": "wrong-word swaps typical of autocorrect and homophones (their/there, its/it's, then/than)",
}
# persona -> {noise: weight}; a persona never gets a noise that contradicts it
NOISE_BY_PERSONA: dict[str, dict[str, float]] = {
    "neutral": {"none": 0.55, "keyboard_typos": 0.20, "no_punct_lower": 0.20, "autocorrect": 0.05},
    "formal": {"none": 0.80, "keyboard_typos": 0.20},
    "casual": {"none": 0.40, "keyboard_typos": 0.20, "no_punct_lower": 0.25, "text_speak": 0.10, "autocorrect": 0.05},
    "frustrated": {
        "none": 0.40,
        "keyboard_typos": 0.20,
        "no_punct_lower": 0.25,
        "text_speak": 0.05,
        "autocorrect": 0.10,
    },
    "anxious": {"none": 0.50, "keyboard_typos": 0.25, "no_punct_lower": 0.20, "autocorrect": 0.05},
    "confused": {"none": 0.50, "keyboard_typos": 0.25, "no_punct_lower": 0.15, "autocorrect": 0.10},
    "hinting": {"none": 0.60, "keyboard_typos": 0.20, "no_punct_lower": 0.20},
    "lay": {"none": 0.55, "keyboard_typos": 0.20, "no_punct_lower": 0.15, "autocorrect": 0.10},
    "jargon": {"none": 0.80, "keyboard_typos": 0.20},
    "multi_topic": {"none": 0.60, "keyboard_typos": 0.20, "no_punct_lower": 0.20},
    "slang_heavy": {
        "none": 0.25,
        "keyboard_typos": 0.15,
        "no_punct_lower": 0.25,
        "text_speak": 0.25,
        "autocorrect": 0.10,
    },
    "rambling_polite": {"none": 0.80, "keyboard_typos": 0.20},
    "non_native": {"none": 0.75, "keyboard_typos": 0.25},
    "fragment_texter": {"none": 0.20, "keyboard_typos": 0.15, "no_punct_lower": 0.35, "text_speak": 0.30},
}

# Node archetypes
# id -> dict(description, k choices with weights, context_required, weight)
ARCHETYPES: dict[str, dict] = {
    "intent_router": {
        "description": "the workflow node decides what the user wants to do next (the most common kind of routing node)",
        "k": {3: 0.12, 4: 0.16, 5: 0.18, 6: 0.16, 7: 0.12, 8: 0.10, 9: 0.08, 10: 0.08},
        "context_required": False,
        "weight": 0.50,
    },
    "triage": {
        "description": "the node classifies a report or request by severity, priority or category",
        "k": {3: 0.4, 4: 0.35, 5: 0.25},
        "context_required": False,
        "weight": 0.10,
    },
    "confirmation": {
        "description": "the assistant has just summarised something (an order, booking, change) and asked the user to confirm; "
        "the node decides whether the user confirms, wants a change, cancels or asks a question",
        "k": {2: 0.25, 3: 0.45, 4: 0.30},
        "context_required": True,
        "weight": 0.10,
    },
    "post_answer_followup": {
        "description": "the assistant has just answered a question; the node decides how the user reacts "
        "(satisfied, wants more detail, says the answer is wrong, switches topic, ...)",
        "k": {3: 0.35, 4: 0.40, 5: 0.25},
        "context_required": True,
        "weight": 0.10,
    },
    "slot_answer_type": {
        "description": "the assistant has just asked the user for a piece of information; the node decides what kind of "
        "information the reply supplies (a date, a reference number, a choice, 'I don't know', a refusal, ...)",
        "k": {3: 0.25, 4: 0.30, 5: 0.25, 6: 0.20},
        "context_required": True,
        "weight": 0.12,
    },
    "handoff": {
        "description": "the node decides whether the user needs a human agent or can continue with the automated assistant",
        "k": {2: 0.6, 3: 0.4},
        "context_required": False,
        "weight": 0.08,
    },
}

# Condition description styles
CONDITION_STYLES: dict[str, tuple[float, str]] = {
    "user_wants": (
        0.35,
        "full descriptions starting with the actor, e.g. 'The customer wants to ...', 'Member asks about ...' (8-16 words)",
    ),
    "imperative_terse": (
        0.25,
        "short imperative action phrases of 2-6 words, e.g. 'Reschedule a visit', 'Report a defect'",
    ),
    "question": (0.10, "yes/no style questions, e.g. 'Is the user asking about the status of an existing job?'"),
    "detailed": (0.20, "long descriptions of 15-30 words that include examples or boundaries in the wording"),
    "internal_jargon": (
        0.10,
        "terse internal team wording with domain abbreviations that a colleague would write in a routing config, still self-explanatory",
    ),
}

# Example plan vocabulary
CONTEXT_KINDS = ("none", "first_turn", "later_turn")
CONTEXT_WEIGHTS = {"none": 0.50, "first_turn": 0.20, "later_turn": 0.30}
ROUTE_TAGS = {
    "indirect": "the user describes the situation or a symptom instead of naming the request",
    "negation": "the message contains a negation that the label depends on (e.g. 'I don't want to cancel, just pause')",
    "distractor": "the message also mentions another, secondary topic; the target is still the main request",
    "context_dependent": "on its own the message is ambiguous or elliptical; the preceding turns decide which condition it is",
}
NM_KINDS = {
    "sibling_removed": "the user's real request is a condition that is NOT among the candidates, while a close sibling condition IS "
    "(so a model that picks the most similar candidate would be wrong)",
    "in_domain_uncovered": "a legitimate request in this business context that none of the candidates covers",
    "oos": "unrelated to the business (chit-chat, general knowledge, another life topic)",
}
NM_WEIGHTS = {"sibling_removed": 0.45, "in_domain_uncovered": 0.35, "oos": 0.20}
