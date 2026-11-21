"""Command-line entry point: ``pathwiser <group> <command> [options]``."""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[tuple[str, str], tuple[str, str]] = {
    ("benchmark", "build"): ("pathwiser.benchmark.__main__", "build the benchmark from data/raw"),
    ("benchmark", "records"): ("pathwiser.benchmark.records", "export model-ready records from the benchmark"),
    ("sets", "specs"): ("pathwiser.synth.spec", "sample scenario specifications"),
    ("sets", "prompt"): ("pathwiser.synth.prompts", "print the generation prompt for a specifications file"),
    ("sets", "check"): ("pathwiser.synth.check", "check generated scenarios against their specifications"),
    ("sets", "build"): ("pathwiser.synth.build", "build an evaluation set from authored scenarios"),
    ("sets", "verify"): ("pathwiser.synth.verify", "turn blind annotations into a verification record"),
    ("scorers", "fit"): ("pathwiser.scorers.fit", "fit or load one scorer and store its scores"),
    ("baselines", "run"): ("pathwiser.reports.baselines", "run baselines under the evaluation protocol"),
}
REPORT_COMMANDS = ("external", "dev", "final", "encoders", "diagnose", "compare")


def usage() -> str:
    lines = ["usage: pathwiser <group> <command> [options]", "", "commands:"]
    lines += [f"  {g} {c:<9} {desc}" for (g, c), (_, desc) in COMMANDS.items()]
    lines += [f"  report {c:<8} reports from stored scores" for c in REPORT_COMMANDS]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] in ("-h", "--help"):
        print(usage())
        return 0 if args and args[0] in ("-h", "--help") else 2
    group, command, rest = args[0], args[1], args[2:]
    prog = f"pathwiser {group} {command}"
    if group == "report" and command in REPORT_COMMANDS:
        module = importlib.import_module("pathwiser.reports.cli")
        return module.main([command, *rest], prog=prog)
    if (group, command) not in COMMANDS:
        print(f"unknown command: {group} {command}\n\n{usage()}", file=sys.stderr)
        return 2
    return importlib.import_module(COMMANDS[(group, command)][0]).main(rest, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
