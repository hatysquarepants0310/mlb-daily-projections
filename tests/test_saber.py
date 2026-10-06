from __future__ import annotations

import math
import unittest

from mlbproj.saber import binomial_at_least_one, clamp, log5, shrink, woba_from_events


class SaberTests(unittest.TestCase):
    def test_log5_identity(self):
        lg = 0.250
        self.assertAlmostEqual(log5(lg, lg, lg), lg, places=6)

    def test_log5_good_vs_bad(self):
        # good hitter vs bad pitcher > good hitter vs league
        a = log5(0.300, 0.280, 0.250)
        b = log5(0.300, 0.250, 0.250)
        self.assertGreater(a, b)

    def test_shrink_to_league_small_n(self):
        self.assertAlmostEqual(shrink(1.0, 0, 0.250, 200), 0.250)
        s = shrink(0.400, 20, 0.250, 200)
        self.assertTrue(0.250 < s < 0.280)

    def test_woba_judge_like(self):
        # 600 AB, 50 BB, 8 HBP, 80 1B, 30 2B, 1 3B, 50 HR
        w = woba_from_events(50, 8, 80, 30, 1, 50, 600, 5, 4)
        self.assertTrue(0.38 < w < 0.50)

    def test_p_hit_increases_with_ab(self):
        p3 = binomial_at_least_one(0.280, 3)
        p5 = binomial_at_least_one(0.280, 5)
        self.assertGreater(p5, p3)
        self.assertTrue(0.6 < p4 if (p4 := binomial_at_least_one(0.280, 4)) else False)

    def test_clamp(self):
        self.assertEqual(clamp(-1), 0.001)
        self.assertEqual(clamp(2), 0.999)


if __name__ == "__main__":
    unittest.main()
