"""Tests for the deterministic noise functions."""

from __future__ import annotations

import unittest

from pathwiser.synth import noise as N

MSG = "Hi, I need to reschedule my appointment for tomorrow because the delivery will be late, please let me know about order 48213."


class TestNoise(unittest.TestCase):
    def test_all_noise_is_deterministic(self):
        for kind in N.NOISE_TYPES:
            self.assertEqual(N.apply_noise(MSG, kind, "ex1"), N.apply_noise(MSG, kind, "ex1"))

    def test_none_is_identity_and_unknown_raises(self):
        self.assertEqual(N.apply_noise(MSG, "none", "k"), (MSG, False))
        with self.assertRaises(ValueError):
            N.apply_noise(MSG, "gibberish", "k")

    def test_keyboard_typos_change_letters_but_never_numbers_or_first_letters(self):
        out, applied = N.keyboard_typos(MSG, "a")
        self.assertTrue(applied)
        self.assertNotEqual(out, MSG)
        self.assertIn("48213", out)
        # changed words keep their first letter
        for a, b in zip(MSG.split(" "), out.split(" ")):
            if a != b:
                self.assertEqual(a[:1], b[:1])
        seeds = {N.keyboard_typos(MSG, f"s{i}")[0] for i in range(20)}
        self.assertGreater(len(seeds), 5)  # different seeds, different damage

    def test_keyboard_typos_leave_short_or_numeric_text_alone(self):
        self.assertEqual(N.keyboard_typos("ok 123", "x"), ("ok 123", False))
        self.assertEqual(N.keyboard_typos("A1B2C3D4 7788", "x"), ("A1B2C3D4 7788", False))

    def test_no_punct_lower_keeps_decimals(self):
        out, applied = N.no_punct_lower("Hello, it's 3.5 kg. Is it ok?", "z")
        self.assertTrue(applied)
        self.assertNotRegex(out, r"[A-Z,?]")
        self.assertIn("3.5", out)
        self.assertFalse(out.endswith("."))

    def test_text_speak_uses_known_forms(self):
        out, applied = N.text_speak("Please call me tomorrow because you are late", "t")
        self.assertTrue(applied)
        self.assertTrue(any(w in out.split() for w in ("pls", "tmrw", "bc", "u", "r")))
        self.assertEqual(N.text_speak("Cancel order 4417", "t"), ("Cancel order 4417", False))  # nothing to abbreviate

    def test_autocorrect_swaps_known_confusions_only(self):
        out, applied = N.autocorrect("Their phone is dead and then it broke", "q")
        self.assertTrue(applied)
        self.assertNotEqual(out, "Their phone is dead and then it broke")
        self.assertEqual(N.autocorrect("Refund order 4417", "q"), ("Refund order 4417", False))

    def test_noise_stays_ascii_and_nonempty(self):
        for kind in N.NOISE_TYPES[1:]:
            out, _ = N.apply_noise(MSG, kind, "u")
            self.assertTrue(out.strip())
            self.assertTrue(out.isascii())


if __name__ == "__main__":
    unittest.main()
