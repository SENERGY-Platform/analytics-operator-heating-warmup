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
import json
import unittest

from algo.schedule import TIMEZONE, parse_plan

UTC = datetime.timezone.utc
H24 = datetime.timedelta(hours=24)


def berlin(text: str, fold: int = 0) -> datetime.datetime:
    """An aware UTC time from a Berlin wall time."""
    return datetime.datetime.fromisoformat(text).replace(tzinfo=TIMEZONE, fold=fold).astimezone(UTC)


def utc(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).replace(tzinfo=UTC)


def plan(*windows, setback=17):
    return parse_plan(json.dumps({"windows": list(windows)}), setback)


def window(start, end, temperature, weekdays=None):
    w = {"from": start, "to": end, "temperature": temperature}
    if weekdays is not None:
        w["weekdays"] = weekdays
    return w


class TestTarget(unittest.TestCase):
    def test_inside_and_outside_a_window(self):
        p = plan(window("06:00", "22:00", 21))
        # 2026-01-12 is a Monday.
        self.assertEqual(p.target(berlin("2026-01-12T05:59:59")), 17)
        self.assertEqual(p.target(berlin("2026-01-12T06:00:00")), 21)
        self.assertEqual(p.target(berlin("2026-01-12T21:59:00")), 21)
        self.assertEqual(p.target(berlin("2026-01-12T22:00:00")), 17)

    def test_the_highest_window_wins(self):
        p = plan(window("06:00", "22:00", 20), window("07:00", "09:00", 22), window("08:00", "10:00", 19))
        self.assertEqual(p.target(berlin("2026-01-12T08:30:00")), 22)
        self.assertEqual(p.target(berlin("2026-01-12T09:30:00")), 20)

    def test_a_window_below_the_setback_lowers_the_target(self):
        p = plan(window("00:00", "05:00", 15))
        self.assertEqual(p.target(berlin("2026-01-12T03:00:00")), 15)

    def test_a_window_past_midnight_belongs_to_its_first_day(self):
        # Friday 22:00 to Saturday 02:00 only.
        p = plan(window("22:00", "02:00", 21, weekdays=[4]))
        self.assertEqual(p.target(berlin("2026-01-16T23:00:00")), 21)  # Friday
        self.assertEqual(p.target(berlin("2026-01-17T01:59:00")), 21)  # Saturday morning
        self.assertEqual(p.target(berlin("2026-01-17T02:00:00")), 17)
        self.assertEqual(p.target(berlin("2026-01-17T23:00:00")), 17)  # Saturday evening is not Friday
        self.assertEqual(p.target(berlin("2026-01-16T01:00:00")), 17)  # Friday morning belongs to Thursday

    def test_from_equal_to_covers_the_whole_day(self):
        p = plan(window("00:00", "00:00", 19, weekdays=[5, 6]))
        self.assertEqual(p.target(berlin("2026-01-17T00:00:00")), 19)
        self.assertEqual(p.target(berlin("2026-01-18T23:59:00")), 19)
        self.assertEqual(p.target(berlin("2026-01-19T00:00:00")), 17)

    def test_24_00_ends_at_midnight(self):
        p = plan(window("18:00", "24:00", 21, weekdays=[0]))
        self.assertEqual(p.target(berlin("2026-01-12T23:59:00")), 21)
        self.assertEqual(p.target(berlin("2026-01-13T00:00:00")), 17)

    def test_empty_weekdays_mean_every_day(self):
        p = plan(window("06:00", "07:00", 21, weekdays=[]))
        for day in range(12, 19):
            self.assertEqual(p.target(berlin(f"2026-01-{day}T06:30:00")), 21, day)

    def test_wall_time_in_summer(self):
        p = plan(window("06:00", "22:00", 21))
        self.assertEqual(p.target(utc("2026-07-01T04:00:00")), 21)  # 06:00 CEST
        self.assertEqual(p.target(utc("2026-07-01T03:59:00")), 17)


class TestBoundaries(unittest.TestCase):
    def test_the_next_day_after_midnight(self):
        p = plan(window("06:00", "22:00", 21))
        found = p.boundaries(berlin("2026-01-12T23:00:00"), H24)
        self.assertEqual([(b.at, b.before, b.after) for b in found],
                         [(berlin("2026-01-13T06:00:00"), 17, 21), (berlin("2026-01-13T22:00:00"), 21, 17)])

    def test_a_window_past_midnight_gives_both_boundaries(self):
        p = plan(window("22:00", "02:00", 21, weekdays=[4]))
        found = p.boundaries(berlin("2026-01-16T12:00:00"), H24)
        self.assertEqual([(b.at, b.after) for b in found],
                         [(berlin("2026-01-16T22:00:00"), 21), (berlin("2026-01-17T02:00:00"), 17)])

    def test_a_whole_day_window_ends_at_midnight_whatever_its_from(self):
        # from == to == 22:00 covers all of Friday; the weekday changes at midnight, not at 22:00.
        p = plan(window("22:00", "22:00", 19, weekdays=[4]))
        found = p.boundaries(berlin("2026-01-16T12:00:00"), H24)
        self.assertEqual([(b.at, b.before, b.after) for b in found], [(berlin("2026-01-17T00:00:00"), 19, 17)])

    def test_the_horizon_is_open_at_now_and_closed_at_its_end(self):
        p = plan(window("06:00", "22:00", 21))
        self.assertEqual(p.boundaries(berlin("2026-01-12T06:00:00"), H24)[0].at, berlin("2026-01-12T22:00:00"))
        found = p.boundaries(berlin("2026-01-12T06:00:00") - datetime.timedelta(hours=24), H24)
        self.assertEqual(found[-1].at, berlin("2026-01-12T06:00:00"))

    def test_overlapping_windows_change_only_where_the_target_changes(self):
        p = plan(window("06:00", "22:00", 21), window("07:00", "09:00", 20))
        found = p.boundaries(berlin("2026-01-12T00:00:00"), H24)
        self.assertEqual([b.at for b in found], [berlin("2026-01-12T06:00:00"), berlin("2026-01-12T22:00:00")])

    def test_spring_forward_keeps_the_wall_time(self):
        # 2026-03-29: 02:00 CET jumps to 03:00 CEST.
        p = plan(window("06:00", "22:00", 21))
        found = p.boundaries(berlin("2026-03-28T23:00:00"), H24)
        self.assertEqual(found[0].at, utc("2026-03-29T04:00:00"))

    def test_fall_back_keeps_the_wall_time(self):
        # 2026-10-25: 03:00 CEST goes back to 02:00 CET.
        p = plan(window("06:00", "22:00", 21))
        found = p.boundaries(berlin("2026-10-24T23:00:00"), H24)
        self.assertEqual(found[0].at, utc("2026-10-25T05:00:00"))

    def test_a_boundary_in_the_spring_gap_takes_effect_at_the_change(self):
        # 02:30 does not exist that night; the window starts when the clock jumps to 03:00.
        p = plan(window("02:30", "05:00", 21))
        found = p.boundaries(berlin("2026-03-28T23:00:00"), datetime.timedelta(hours=6))
        self.assertEqual([(b.at, b.after) for b in found],
                         [(utc("2026-03-29T01:00:00"), 21), (utc("2026-03-29T03:00:00"), 17)])

    def test_a_boundary_in_the_repeated_hour_happens_twice(self):
        # As in the heating process: the clock going back from 03:00 to 02:00 leaves the window until 02:30 again.
        p = plan(window("02:30", "05:00", 21))
        found = p.boundaries(berlin("2026-10-24T23:00:00"), datetime.timedelta(hours=8))
        self.assertEqual([(b.at, b.after) for b in found],
                         [(utc("2026-10-25T00:30:00"), 21), (utc("2026-10-25T01:00:00"), 17),
                          (utc("2026-10-25T01:30:00"), 21), (utc("2026-10-25T04:00:00"), 17)])

    def test_a_window_ending_in_the_repeated_hour_ends_at_its_first_occurrence(self):
        # Ends at the first 02:30 (CEST); in the repeated hour 02:30 to 03:00 CET the minute is past `to` again.
        p = plan(window("01:00", "02:30", 21))
        found = p.boundaries(berlin("2026-10-24T23:00:00"), datetime.timedelta(hours=8))
        self.assertEqual([(b.at, b.after) for b in found],
                         [(berlin("2026-10-25T01:00:00"), 21), (berlin("2026-10-25T02:30:00", fold=0), 17),
                          (berlin("2026-10-25T02:00:00", fold=1), 21), (berlin("2026-10-25T02:30:00", fold=1), 17)])


class TestParse(unittest.TestCase):
    def test_empty_schedule_and_setback_give_no_plan(self):
        self.assertIsNone(parse_plan("", ""))
        self.assertIsNone(parse_plan(" ", None))

    def test_numbers_as_strings(self):
        p = parse_plan(json.dumps({"windows": [{"from": "6:00", "to": "7:30", "temperature": "21.5"}]}), "16.5")
        self.assertEqual((p.setback, p.windows[0].start, p.windows[0].end, p.windows[0].temperature), (16.5, 360, 450, 21.5))

    def test_invalid_input_raises_as_in_the_process(self):
        cases = [
            ("{\"windows\": []}", ""),  # setback empty
            ("{\"windows\": []}", "warm"),
            ("", "17"),
            ("not json", "17"),
            ("{}", "17"),
            (json.dumps({"windows": [window("6:00", "25:00", 21)]}), "17"),
            (json.dumps({"windows": [window("24:30", "06:00", 21)]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", None)]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", "")]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", True)]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", 21, weekdays=[7])]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", 21, weekdays=["1"])]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", 21, weekdays=[True])]}), "17"),
            (json.dumps({"windows": [window("06:00", "07:00", 21, weekdays=3)]}), "17"),
        ]
        for schedule, setback in cases:
            with self.subTest(schedule=schedule, setback=setback):
                with self.assertRaises(ValueError):
                    parse_plan(schedule, setback)


if __name__ == "__main__":
    unittest.main()
