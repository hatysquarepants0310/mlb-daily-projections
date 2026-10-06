from __future__ import annotations

import unittest

from mlbproj.engine import inputs_status, weather_observed


class InputsStatusTests(unittest.TestCase):
    def test_listo_when_all_five_present(self):
        s = inputs_status(
            sp_away=True,
            sp_home=True,
            lineup_away=True,
            lineup_home=True,
            weather=True,
        )
        self.assertTrue(s["ready"])
        self.assertEqual(s["state"], "listo")
        self.assertEqual(s["missing"], [])
        self.assertIn("Listo", s["label"])

    def test_esperando_when_nothing_posted(self):
        s = inputs_status(
            sp_away=False,
            sp_home=False,
            lineup_away=False,
            lineup_home=False,
            weather=False,
        )
        self.assertFalse(s["ready"])
        self.assertEqual(s["state"], "esperando")
        self.assertEqual(
            s["missing"],
            ["sp_away", "sp_home", "lineup_away", "lineup_home", "weather"],
        )

    def test_parcial_when_sp_but_no_lineups(self):
        s = inputs_status(
            sp_away=True,
            sp_home=True,
            lineup_away=False,
            lineup_home=False,
            weather=False,
        )
        self.assertFalse(s["ready"])
        self.assertEqual(s["state"], "parcial")
        self.assertIn("lineup_away", s["missing"])
        self.assertIn("weather", s["missing"])
        self.assertTrue(s["label"].startswith("Esperando:"))

    def test_weather_observed_needs_temp_from_api(self):
        self.assertFalse(weather_observed({}))
        self.assertFalse(weather_observed({"weather": {}}))
        self.assertFalse(weather_observed({"weather": {"wind": "10 mph, Out To CF"}}))
        self.assertTrue(weather_observed({"weather": {"temp": "72", "wind": "8 mph, L To R"}}))
        self.assertTrue(weather_observed({"weather": {"temp": 61, "condition": "Clear"}}))


if __name__ == "__main__":
    unittest.main()
