"""Tests for the blind-verification tooling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.synth import verify as V


def records(n: int = 10) -> list[dict]:
    out = []
    for i in range(n):
        k = 3 + i % 2
        out.append(
            {
                "example_id": f"e{i}",
                "input": {
                    "node": {"name": "demo_node", "description": "Decide something."},
                    "context": [{"role": "user", "content": "hi"}] if i % 2 else [],
                    "message": f"message {i}",
                    "conditions": [f"cond {i}-{j}" for j in range(k)],
                },
                "target": {"label": -1 if i % 4 == 3 else i % 3},
            }
        )
    return out


def answer_all(out_dir: Path, recs: list[dict], key: dict, wrong: set[str] = frozenset()) -> None:
    """Annotators that pick the intended label, except for the example ids in ``wrong`` (they answer NONE or another letter)."""
    by_id = {r["example_id"]: r for r in recs}
    for f in sorted((out_dir / "blind").glob("*.json")):
        items = json.loads(f.read_text(encoding="utf-8"))
        ans = []
        for it in items:
            k = key[it["id"]]
            r = by_id[k["example_id"]]
            label = r["target"]["label"]
            letter = "NONE" if label < 0 else next(L for L, o in k["letters"].items() if o == label)
            if k["example_id"] in wrong:
                letter = "NONE" if letter != "NONE" else "A"
            ans.append({"id": it["id"], "pick": letter, "alt": None, "note": ""})
        (out_dir / "answers" / f.name).write_text(json.dumps(ans), encoding="utf-8")


class TestVerify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.recs = records()
        self.files = V.make_blind(self.recs, self.dir, seed=3)
        self.key = json.loads((self.dir / "key" / "key.json").read_text(encoding="utf-8"))

    def test_every_example_is_in_exactly_one_file_of_each_pass(self):
        self.assertEqual(sorted(self.files), ["A1", "A2", "B1", "B2"])
        for p in "AB":
            ids = [self.key[i]["example_id"] for i in self.key if i.startswith(p)]
            self.assertEqual(sorted(ids), sorted(r["example_id"] for r in self.recs))

    def test_blind_items_do_not_leak_labels_or_ids_and_pass_b_is_reversed(self):
        for f in (self.dir / "blind").glob("*.json"):
            for item in json.loads(f.read_text(encoding="utf-8")):
                self.assertEqual(sorted(item), ["conditions", "context", "id", "message", "node"])
                self.assertTrue(item["id"][0] in "AB" and not item["id"].startswith("e"))
        a_first = json.loads((self.dir / "blind" / "A1.json").read_text(encoding="utf-8"))[0]
        k = self.key[a_first["id"]]
        self.assertEqual(list(k["letters"].values()), sorted(k["letters"].values()))
        b_first = json.loads((self.dir / "blind" / "B1.json").read_text(encoding="utf-8"))[0]
        kb = self.key[b_first["id"]]
        self.assertEqual(list(kb["letters"].values()), sorted(kb["letters"].values(), reverse=True))

    def test_perfect_annotators_give_full_agreement(self):
        answer_all(self.dir, self.recs, self.key)
        ans = V.collect(self.dir, ["A1", "A2", "B1", "B2"])
        ver = V.build_verification(self.recs, ans, {"A": "model-a", "B": "model-b"}, "test")
        self.assertEqual(V.disagreements(self.recs, ver), [])
        s = V.agreement_summary(self.recs, ver)
        self.assertEqual((s["examples"], s["all_annotators_agree"], s["share"]), (10, 10, 1.0))
        self.assertEqual(ver["examples"]["e0"]["passes"]["A"]["annotator"], "model-a")

    def test_disagreements_are_listed_with_context(self):
        answer_all(self.dir, self.recs, self.key, wrong={"e2", "e3"})
        ver = V.build_verification(self.recs, V.collect(self.dir, ["A1", "A2", "B1", "B2"]), {"A": "a", "B": "b"}, "t")
        bad = V.disagreements(self.recs, ver)
        self.assertEqual(sorted(d["example_id"] for d in bad), ["e2", "e3"])
        self.assertTrue(all(d["reason"] == "disagreement" and d["picks"] for d in bad))
        self.assertEqual(V.agreement_summary(self.recs, ver)["all_annotators_agree"], 8)

    def test_edited_example_is_reported_as_changed(self):
        answer_all(self.dir, self.recs, self.key)
        ver = V.build_verification(self.recs, V.collect(self.dir, ["A1", "A2", "B1", "B2"]), {"A": "a", "B": "b"}, "t")
        edited = json.loads(json.dumps(self.recs))
        edited[4]["input"]["message"] = "a different message"
        out = V.disagreements(edited, ver)
        self.assertEqual([(d["example_id"], d["reason"]) for d in out], [("e4", "changed after annotation")])

    def test_missing_and_invalid_answers_are_errors(self):
        answer_all(self.dir, self.recs, self.key)
        path = self.dir / "answers" / "A1.json"
        ans = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(ans[:-1]), encoding="utf-8")
        with self.assertRaises(ValueError):
            V.collect(self.dir, ["A1"])
        ans[0]["pick"] = "Z"
        path.write_text(json.dumps(ans), encoding="utf-8")
        with self.assertRaises(ValueError):
            V.collect(self.dir, ["A1"])

    def test_answers_for_unknown_examples_are_rejected(self):
        with self.assertRaises(ValueError):
            V.build_verification(self.recs, {"nope": {"A": {"pick": "NONE", "alt": None, "note": ""}}}, {"A": "a"}, "t")

    def test_cli_writes_the_record_and_the_disagreements(self):
        answer_all(self.dir, self.recs, self.key, wrong={"e1"})
        recs_path = self.dir / "records.jsonl"
        recs_path.write_text("".join(json.dumps(r) + "\n" for r in self.recs), encoding="utf-8")
        out, bad = self.dir / "ver.json", self.dir / "bad.json"
        V.main(
            [
                "--work",
                str(self.dir),
                "--records",
                str(recs_path),
                "--out",
                str(out),
                "--annotators",
                "A=m1",
                "B=m2",
                "--disagreements",
                str(bad),
            ]
        )
        ver = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(ver["examples"]["e0"]["passes"]["B"]["annotator"], "m2")
        self.assertEqual([d["example_id"] for d in json.loads(bad.read_text(encoding="utf-8"))], ["e1"])

    def test_merge_replaces_edited_examples_and_keeps_history(self):
        answer_all(self.dir, self.recs, self.key)
        old = V.build_verification(
            self.recs, V.collect(self.dir, ["A1", "A2", "B1", "B2"]), {"A": "a", "B": "b"}, "first"
        )
        new = {
            "examples": {
                "e1": {
                    "verified_sha": "new",
                    "passes": {"C": {"annotator": "c", "pick": "NONE", "alt": None, "note": ""}},
                }
            },
            "note": "second",
        }
        merged = V.merge_verification(old, new, "round 2")
        self.assertEqual(merged["examples"]["e1"]["verified_sha"], "new")
        self.assertEqual(set(merged["examples"]), set(old["examples"]))
        self.assertEqual([r["label"] for r in merged["rounds"]], ["round 1", "round 2"])
        self.assertEqual(merged["rounds"][0]["examples_annotated"], 10)
        self.assertEqual(merged["rounds"][1]["examples_annotated"], 1)
        self.assertIn("second", merged["note"])


if __name__ == "__main__":
    unittest.main()
