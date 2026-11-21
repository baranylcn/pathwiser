import json
import sys
import unittest
from pathlib import Path

from pathwiser.paths import DATA_DIR, REPO_ROOT
from pathwiser.synth.prompts import build_prompt

SPECS = DATA_DIR / "curated" / "realistic_v2" / "specs.json"


class TestPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = json.loads(SPECS.read_text(encoding="utf-8"))[:3]

    def test_prompt_names_output_and_check_command(self):
        text = build_prompt(self.specs, "specs.json", "out.json", workdir="/work", python="py")
        self.assertIn("out.json", text)
        self.assertIn("py -X utf8 -m pathwiser.synth.check --specs specs.json --scenarios out.json", text)
        self.assertIn("/work", text)

    def test_prompt_lists_every_spec_id(self):
        text = build_prompt(self.specs, "specs.json", "out.json")
        for spec in self.specs:
            self.assertIn(spec["id"], text)

    def test_defaults_come_from_the_running_environment(self):
        text = build_prompt(self.specs, "specs.json", "out.json")
        self.assertIn(REPO_ROOT.as_posix(), text)
        self.assertIn(Path(sys.executable).as_posix(), text)


if __name__ == "__main__":
    unittest.main()
