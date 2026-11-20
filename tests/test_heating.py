"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import datetime
import unittest

import numpy as np
import pandas as pd

from algo.heating import HeatingCurve, OffWindow, parse_heating_curve, parse_off_window

UTC = datetime.timezone.utc
STEP_NS = 10 * 60 * 10**9


def utc(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).replace(tzinfo=UTC)


def off_steps(window: str, first: str, last: str):
    """The UTC start times (HH:MM) of the 10-minute steps from first to last that touch the window."""
    times = np.arange(pd.Timestamp(first, tz="UTC").value, pd.Timestamp(last, tz="UTC").value + 1, STEP_NS,
                      dtype=np.int64)
    off = parse_off_window(window).steps(times, STEP_NS)
    return [pd.Timestamp(t, tz="UTC").strftime("%H:%M") for t in times[off]]


class TestHeatingCurve(unittest.TestCase):
    def test_flow_temperature(self):
        # Read off a boiler: slope 0.8, room 22, 10 °C outside gives 32 °C flow.
        self.assertAlmostEqual(HeatingCurve(0.8, 22.0).flow(10.0), 31.6)

    def test_parsing(self):
        self.assertIsNone(parse_heating_curve("", " "))
        self.assertIsNone(parse_heating_curve(None, None))
        self.assertEqual(parse_heating_curve("0.8", 22), HeatingCurve(0.8, 22.0))
        self.assertEqual(parse_heating_curve(1.2, "-3"), HeatingCurve(1.2, -3.0))
        for slope, room in (("0.8", ""), ("", "22")):
            with self.subTest(slope=slope, room=room):
                with self.assertRaisesRegex(ValueError, "together"):
                    parse_heating_curve(slope, room)
        for slope, room in (("0", "22"), ("-0.5", "22"), ("x", "22"), ("nan", "22"),
                            ("inf", "22"), ("0.8", "nan"), (True, "22"), ("0.8", "warm")):
            with self.subTest(slope=slope, room=room):
                with self.assertRaises(ValueError):
                    parse_heating_curve(slope, room)


class TestOffWindowParsing(unittest.TestCase):
    def test_valid(self):
        self.assertIsNone(parse_off_window(""))
        self.assertIsNone(parse_off_window(None))
        self.assertEqual(parse_off_window("23:30-03:00"), OffWindow(23 * 60 + 30, 180))
        self.assertEqual(parse_off_window(" 1:05 - 2:00 "), OffWindow(65, 120))
        self.assertEqual(parse_off_window("22:00-24:00"), OffWindow(22 * 60, 1440))
        self.assertEqual(parse_off_window("00:00-00:01"), OffWindow(0, 1))

    def test_invalid(self):
        for text in ("23:30", "23:30-", "24:00-03:00", "23:60-03:00", "23:30-24:01", "23:30-25:00", "3-4",
                     "03:00-03:00", "00:00-24:00", "23:30–03:00", "a:bc-03:00", "123:00-03:00"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_off_window(text)


class TestOffWindowSteps(unittest.TestCase):
    def test_a_step_touching_either_end_is_off(self):
        # 23:30-03:00 Berlin is 22:30-02:00 UTC in January, and [22:20, 22:30] and [02:00, 02:10] touch it.
        steps = off_steps("23:30-03:00", "2026-01-12 21:50", "2026-01-13 02:40")
        self.assertEqual(steps[0], "22:20")
        self.assertEqual(steps[-1], "02:00")
        self.assertEqual(len(steps), 23)

    def test_summer_time(self):
        steps = off_steps("23:30-03:00", "2026-07-12 20:50", "2026-07-13 01:40")
        self.assertEqual((steps[0], steps[-1]), ("21:20", "01:00"))

    def test_a_window_inside_one_step(self):
        self.assertEqual(off_steps("12:03-12:07", "2026-01-12 10:40", "2026-01-12 11:20"), ["11:00"])

    def test_a_window_within_the_day(self):
        self.assertEqual(off_steps("12:00-12:30", "2026-01-12 10:40", "2026-01-12 11:50"),
                         ["10:50", "11:00", "11:10", "11:20", "11:30"])

    def test_every_day(self):
        times = np.arange(pd.Timestamp("2026-01-12", tz="UTC").value, pd.Timestamp("2026-01-15", tz="UTC").value,
                          STEP_NS, dtype=np.int64)
        off = parse_off_window("23:30-03:00").steps(times, STEP_NS)
        # The night before until 02:00, two whole nights, the last one from 22:20 until the end at 23:50.
        self.assertEqual(int(off.sum()), 13 + 23 + 23 + 10)

    def test_the_repeated_autumn_hour_is_off_both_times(self):
        # 2026-10-25: 02:00-03:00 runs twice, 00:00-01:00 UTC in summer time and 01:00-02:00 UTC in winter time.
        steps = off_steps("02:15-02:45", "2026-10-24 23:30", "2026-10-25 02:30")
        self.assertIn("00:20", steps)
        self.assertIn("01:20", steps)
        self.assertEqual((steps[0], steps[-1]), ("00:10", "01:40"))

    def test_the_spring_gap(self):
        # 2026-03-29: 02:00-03:00 does not exist. Read in winter and in summer time the window spans 00:15-01:45 UTC.
        steps = off_steps("02:15-02:45", "2026-03-28 23:30", "2026-03-29 02:30")
        self.assertEqual((steps[0], steps[-1], len(steps)), ("00:10", "01:40", 10))

    def test_no_times(self):
        self.assertEqual(len(parse_off_window("23:30-03:00").steps(np.array([], dtype=np.int64), STEP_NS)), 0)


class TestOffWindowResumes(unittest.TestCase):
    def test_inside_the_window_its_end(self):
        off = parse_off_window("23:30-03:00")
        # 01:00 UTC is 02:00 Berlin, and the window ends 02:00 UTC.
        self.assertEqual(off.resumes(utc("2026-01-13T01:00:00")), utc("2026-01-13T02:00:00"))
        self.assertEqual(off.resumes(utc("2026-01-12T22:30:00")), utc("2026-01-13T02:00:00"))

    def test_outside_the_last_end(self):
        off = parse_off_window("23:30-03:00")
        self.assertEqual(off.resumes(utc("2026-01-13T02:00:00")), utc("2026-01-13T02:00:00"))
        self.assertEqual(off.resumes(utc("2026-01-13T12:00:00")), utc("2026-01-13T02:00:00"))
        self.assertEqual(off.resumes(utc("2026-01-13T22:29:00")), utc("2026-01-13T02:00:00"))


if __name__ == "__main__":
    unittest.main()
