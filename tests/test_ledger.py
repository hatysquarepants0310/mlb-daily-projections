from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from mlbproj import ledger
from mlbproj.bets import momio_americano, momio_decimal, p_over, p_win


class MomioTests(unittest.TestCase):
    def test_even(self):
        self.assertAlmostEqual(momio_decimal(0.5), 2.0)
        self.assertEqual(momio_americano(0.5), -100)

    def test_favorite(self):
        self.assertTrue(momio_decimal(0.65) < 2)
        self.assertTrue(momio_americano(0.65) < 0)


class PythagTests(unittest.TestCase):
    def test_better_offense_favored(self):
        self.assertGreater(p_win(5.2, 3.8, True), 0.5)
        self.assertGreater(p_over(6.5, 9.0), 0.5)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        ledger.LEDGER = Path(self.tmp.name) / "ledger.sqlite"
        ledger.init()

    def tearDown(self):
        self.tmp.cleanup()

    def test_lock_is_insert_only(self):
        p = {
            "pick_id": "m1:ml:Yankees",
            "game_pk": 1,
            "card_date": "2026-10-05",
            "selection": "Yankees",
            "implied_p": 0.5,
            "momio_decimal": 2.0,
            "model_p": 0.58,
        }
        a = ledger.lock_pick(p)
        p2 = dict(p)
        p2["model_p"] = 0.99
        b = ledger.lock_pick(p2)
        self.assertEqual(a["model_p"], 0.58)
        self.assertEqual(b["model_p"], 0.58)
        self.assertEqual(a["row_hash"], b["row_hash"])

    def test_update_and_delete_aborted(self):
        p = {
            "pick_id": "m2:ml:Rays",
            "game_pk": 2,
            "card_date": "2026-10-05",
            "selection": "Rays",
            "implied_p": 0.4,
            "momio_decimal": 2.5,
            "model_p": 0.55,
        }
        ledger.lock_pick(p)
        conn = sqlite3.connect(ledger.LEDGER)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("UPDATE picks SET payload='x' WHERE pick_id=?", ("m2:ml:Rays",))
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM picks WHERE pick_id=?", ("m2:ml:Rays",))
        conn.close()

    def test_settle_once(self):
        p = {
            "pick_id": "m3:ml:X",
            "game_pk": 3,
            "card_date": "2026-10-05",
            "selection": "X",
            "implied_p": 0.5,
            "momio_decimal": 2.0,
            "model_p": 0.6,
        }
        ledger.lock_pick(p)
        a = ledger.settle_pick("m3:ml:X", True, {"runs": 1})
        b = ledger.settle_pick("m3:ml:X", False, {"runs": 99})
        self.assertTrue(a["won"])
        self.assertTrue(b["won"])
        self.assertEqual(a["actual"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
