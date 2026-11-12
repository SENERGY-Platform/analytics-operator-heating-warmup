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

from algo.forecast import KEPT_RUNS, Forecasts, parse_utc

UTC = datetime.timezone.utc


def at(text):
    return parse_utc(text)


def run(issued_hour: int, values, day="2026-11-12"):
    """A MOSMIX run issued at `issued_hour` UTC, hourly from one hour after."""
    start = at(f"{day}T00:00:00Z") + datetime.timedelta(hours=issued_hour + 1)
    times = [(start + datetime.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(values))]
    return list(values), times


class TestForecasts(unittest.TestCase):
    def test_the_latest_issue_wins(self):
        f = Forecasts()
        f.add(*run(3, [1, 2, 3, 4, 5, 6, 7, 8]))
        f.add(*run(6, [10, 11, 12, 13, 14]))
        self.assertEqual(f.at(at("2026-11-12T08:00:00Z")), 11)
        self.assertEqual(f.at(at("2026-11-12T05:00:00Z")), 2)  # before the latest run starts

    def test_the_order_of_arrival_does_not_matter(self):
        f = Forecasts()
        f.add(*run(6, [10, 11, 12]))
        f.add(*run(3, [1, 2, 3, 4, 5, 6]))
        self.assertEqual(f.at(at("2026-11-12T08:00:00Z")), 11)

    def test_a_null_hour_falls_back_to_the_previous_issue(self):
        f = Forecasts()
        f.add(*run(3, [1, 2, 3, 4, 5, 6]))
        f.add(*run(6, [10, None, 12]))
        self.assertEqual(f.at(at("2026-11-12T08:00:00Z")), 5)

    def test_zero_degrees_is_a_value(self):
        f = Forecasts()
        f.add(*run(3, [1, 2, 3, 4, 5, 6]))
        f.add(*run(6, [10, 0.0, 12]))
        self.assertEqual(f.at(at("2026-11-12T08:00:00Z")), 0.0)

    def test_no_run_covers_the_time(self):
        f = Forecasts()
        self.assertIsNone(f.at(at("2026-11-12T08:00:00Z")))
        f.add(*run(6, [10, 11, 12]))
        self.assertIsNone(f.at(at("2026-11-12T06:30:00Z")))
        self.assertIsNone(f.at(at("2026-11-12T09:01:00Z")))

    def test_between_two_hours_is_interpolated(self):
        f = Forecasts()
        f.add(*run(6, [10, 12, 13]))
        self.assertAlmostEqual(f.at(at("2026-11-12T07:15:00Z")), 10.5)
        self.assertIsNone(Forecasts().at(at("2026-11-12T07:15:00Z")))

    def test_a_null_neighbour_leaves_the_time_uncovered(self):
        f = Forecasts()
        f.add(*run(6, [10, None, 13]))
        self.assertIsNone(f.at(at("2026-11-12T07:30:00Z")))

    def test_hours_further_apart_than_one_are_not_interpolated(self):
        f = Forecasts()
        f.add([10, 13], ["2026-11-12T07:00:00Z", "2026-11-12T10:00:00Z"])
        self.assertIsNone(f.at(at("2026-11-12T08:00:00Z")))
        self.assertEqual(f.at(at("2026-11-12T10:00:00Z")), 13)

    def test_a_run_published_again_replaces_itself(self):
        f = Forecasts()
        f.add(*run(6, [10, 11, 12]))
        f.add(*run(6, [10, None, 12]))
        self.assertEqual(len(f), 1)
        self.assertIsNone(f.at(at("2026-11-12T08:00:00Z")))

    def test_a_run_published_twice_is_kept_once_and_old_runs_go(self):
        f = Forecasts()
        for hour in range(KEPT_RUNS + 2):
            f.add(*run(hour, [hour] * 3))
        f.add(*run(KEPT_RUNS + 1, [99] * 3))
        self.assertEqual(len(f), KEPT_RUNS)
        self.assertEqual(f.at(at("2026-11-12T07:00:00Z")), 99)
        self.assertIsNone(f.at(at("2026-11-12T01:00:00Z")))  # only the dropped first run reached it

    def test_malformed_messages_are_refused(self):
        f = Forecasts()
        values, times = run(6, [10, 11, 12])
        self.assertFalse(f.add(values[:2], times))
        self.assertFalse(f.add(10.0, times))
        self.assertFalse(f.add([], []))
        self.assertFalse(f.add(values, ["yesterday"] * 3))
        self.assertFalse(f.add(values, [times[0]] * 3))
        self.assertEqual(len(f), 0)


if __name__ == "__main__":
    unittest.main()
