"""The instruction given to the model that writes scenarios for a batch of specifications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pathwiser.paths import REPO_ROOT
from pathwiser.synth import styles as S

EXAMPLE = [
    {
        "id": "format_only_example",
        "node": {
            "name": "festival_office_router",
            "description": "Decide what the caller wants from the balloon festival office.",
        },
        "conditions": {
            "a": "Ask about the festival schedule",
            "b": "Buy or change a pilot's registration",
            "c": "Report a lost item",
            "d": "Volunteer for the weekend crew",
        },
        "examples": [
            {
                "gold": "c",
                "ctx": [],
                "msg": "i think i left my red backpack near the launch field on saturday",
                "why": "describes a lost bag",
            },
            {
                "gold": "d",
                "ctx": [["a", "Festival office, how can I help?"]],
                "msg": "Hello, my brother and I have the whole weekend free and we would love to help out.",
                "why": "offers time to help",
            },
            {
                "gold": None,
                "nm": "sibling_removed",
                "intent": "b",
                "cands": ["a", "c", "d"],
                "ctx": [],
                "msg": "I need to move my pilot slot from Friday to Sunday",
                "why": "wants to change a registration (not offered)",
            },
            {
                "gold": None,
                "nm": "oos",
                "ctx": [],
                "msg": "what's a good recipe for lentil soup?",
                "why": "unrelated to the festival",
            },
        ],
    }
]


def _legend(title: str, items: dict[str, str]) -> str:
    return f"{title}\n" + "\n".join(f"- {k}: {v}" for k, v in items.items())


def build_prompt(
    specs: list[dict],
    specs_path: str,
    out_path: str,
    workdir: str | None = None,
    python: str | None = None,
) -> str:
    workdir = workdir or REPO_ROOT.as_posix()
    python = python or Path(sys.executable).as_posix()
    used_personas = sorted({e["persona"] for s in specs for e in s["examples"]})
    used_archetypes = sorted({s["archetype"] for s in specs})
    used_styles = sorted({s["condition_style"] for s in specs})
    used_tags = sorted({t for s in specs for e in s["examples"] for t in e["tags"]})
    used_nm = sorted({e["nm"] for s in specs for e in s["examples"] if e["kind"] == "no_match"})
    legends = "\n\n".join(
        [
            _legend(
                "Node archetypes used in this batch:", {a: S.ARCHETYPES[a]["description"] for a in used_archetypes}
            ),
            _legend(
                "Condition description styles used in this batch:", {c: S.CONDITION_STYLES[c][1] for c in used_styles}
            ),
            _legend(
                "Personas used in this batch (how the person WRITES; apply it to every user turn of the example):",
                {p: S.PERSONAS[p][1] for p in used_personas},
            ),
            _legend("Difficulty tags used in this batch:", {t: S.ROUTE_TAGS[t] for t in used_tags}),
            _legend("no_match kinds used in this batch:", {k: S.NM_KINDS[k] for k in used_nm}),
        ]
    )
    return f"""You are writing labelled routing examples for a machine-learning dataset. A *scenario* is a workflow decision node in a real
business context together with k candidate conditions (natural-language descriptions of branches). Each *example* is a conversation snippet:
optional earlier turns plus the user's current message. A model must choose the one condition that matches what the user wants, or decide that
none does. You write the text; the specification of every scenario (who, what, how many, which persona) is already fixed below and you must
follow it exactly.

WHAT TO PRODUCE
Write ONE valid JSON file (a list with one object per scenario, in the order of the specifications) to:
  {out_path}
Object schema (see the format example at the end; the example is about an unrelated invented domain, do not reuse its wording):
  {{"id": <spec id>, "node": {{"name": <snake_case>, "description": <one sentence>}},
    "conditions": {{"a": <text>, "b": <text>, ...}},          # exactly k letters, in order, a, b, c, ...
    "examples": [ <one object per planned example, in the planned order> ]}}
Example object:
  route:                     {{"gold": "<target letter>", "ctx": [...], "msg": "<current user message>", "why": "<=12 words"}}
  no_match, sibling_removed: {{"gold": null, "nm": "sibling_removed", "intent": "<letter>", "cands": [<all letters except intent>], "ctx": [...], "msg": "...", "why": "..."}}
  no_match, other kinds:     {{"gold": null, "nm": "<kind>", "ctx": [...], "msg": "...", "why": "..."}}
`ctx` is a list of turns [["a" or "u", "text"], ...] (a = assistant, u = user), oldest first. The current message `msg` is NOT part of ctx.

RULES
1. Labels are correct by construction. For a route example the message must be an expression of the TARGET condition and a careful reader looking at
   the node, the context and the k conditions must be able to pick it uniquely. Check every example against ALL conditions, especially the siblings.
2. Conditions: write them first. Each must be a distinct intent of the node (no catch-all such as "other" or "anything else"). Make some conditions
   close siblings that a keyword or embedding model could confuse (for instance "cancel a booking" vs "change a booking"). If the specifications plan
   a sibling_removed example, its `intent` condition and its `sibling` condition MUST be close in meaning but lead to a different action, and the
   message must make the `intent` reading clearly the right one (so choosing the sibling would be wrong). Follow the requested description style.
3. Context kinds: none -> ctx is []. first_turn -> ctx contains assistant turn(s) only (a greeting, or the question/answer/summary the archetype
   needs). later_turn -> ctx starts with a user turn, alternates, has 1-4 turns in total and ENDS with an assistant turn. Context must set up the
   node's archetype (for confirmation: the assistant just summarised something and asked to confirm, and so on). User turns inside ctx are written in
   the same persona as the message.
4. A message tagged context_dependent must be unclear on its own (e.g. "yes please", "the second one", "tomorrow works") and become unique only
   through the context. Other messages must stand on their own.
5. no_match: sibling_removed -> the message expresses the removed `intent` condition (it is not among the candidates, its sibling is).
   in_domain_uncovered -> a legitimate request for this business that NO candidate covers, not even loosely. oos -> unrelated to the business.
6. Persona: write the way that persona writes. Do NOT add typos, text-speak abbreviations or remove punctuation yourself; character-level noise is
   applied later by code (the `noise` field in the specification is only for your information). Exception: personas whose definition contains
   grammar errors (non_native) or fragments (fragment_texter) are part of the persona.
7. Realism: real people do not write polished sentences with the vocabulary of the condition. Vary wording across the examples of a scenario, do not
   copy the condition's words (unless the persona is jargon), and do not mention letters or labels. Messages are usually 3-35 words; rambling_polite
   may reach about 60. Every example of a scenario must differ in wording and in the situation it describes.
8. Content: invent everything (names, numbers, places); no real company or product names, no real personal data. Mild swearing is allowed ONLY for the
   personas frustrated and slang_heavy; never slurs, hate, sexual content, threats or self-harm.
9. The user speaks as someone who deals with this business context (see domain_description). The workflow node and conditions are what the business
   configured, written for developers.
10. `why` is a short note (at most 12 words) saying how the message maps to the target (or why nothing fits). It is for reviewers.

{legends}

SELF-CHECK (required)
After writing the file, run this command in the directory {workdir} and fix your file until it prints OK:
  {python} -X utf8 -m pathwiser.synth.check --specs {specs_path} --scenarios {out_path}
Then re-read your examples once more with the eyes of a careful annotator who does not know the targets: would they pick the target? Rewrite any
example where a second condition is also defensible. Do not read, list or search any other file or directory (the repository contains data that must
not influence you); only run the command above.

FINAL MESSAGE
Reply with: "done <n> scenarios" and, if any, a list of examples you are not fully sure about (id and one line each). Nothing else.

FORMAT EXAMPLE (unrelated domain, wording must not be reused)
{json.dumps(EXAMPLE, ensure_ascii=False, indent=1)}

SPECIFICATIONS ({len(specs)} scenarios)
{json.dumps(specs, ensure_ascii=False, separators=(",", ":"))}
"""


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    """``pathwiser sets prompt --specs specs.json --out scenarios.json [--first N]``: print the generation prompt."""
    parser = argparse.ArgumentParser(prog=prog, description="Print the prompt that asks a model to write scenarios.")
    parser.add_argument("--specs", required=True, help="specifications file (output of `pathwiser sets specs`)")
    parser.add_argument("--out", required=True, help="path the model should write its scenarios to")
    parser.add_argument("--first", type=int, default=None, help="only the first N specifications")
    args = parser.parse_args(argv)
    specs = json.loads(Path(args.specs).read_text(encoding="utf-8"))
    if isinstance(specs, dict):
        specs = specs["specs"]
    print(build_prompt(specs[: args.first], args.specs, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
