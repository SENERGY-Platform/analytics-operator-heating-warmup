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

from algo.output import Throttle, iso_utc, round_to, whole_minutes

T0 = datetime.datetime(2026, 1, 12, 12, tzinfo=datetime.timezone.utc)


class TestRounding(unittest.TestCase):
    def test_half_away_from_zero_unlike_round(self):
        self.assertEqual(round(0.25, 1), 0.2)
        self.assertEqual(round_to(0.25, "0.1"), 0.3)
        self.assertEqual(round_to(-0.25, "0.1"), -0.3)
        self.assertEqual(round_to(2.675, "0.01"), 2.68)
        self.assertEqual(whole_minutes(0.5 / 60), 1)
        self.assertEqual(whole_minutes(2.5 / 60), 3)
        self.assertIsNone(round_to(None, "0.1"))

    def test_iso_utc(self):
        berlin = datetime.timezone(datetime.timedelta(hours=1))
        self.assertEqual(iso_utc(datetime.datetime(2026, 1, 12, 6, tzinfo=berlin)), "2026-01-12T05:00:00Z")


class TestThrottle(unittest.TestCase):
    def test_exactly_five_minutes_later_passes(self):
        throttle = Throttle(datetime.timedelta(minutes=5))
        self.assertEqual(throttle({"x": 1}, T0), {"x": 1})
        self.assertIsNone(throttle({"x": 2}, T0 + datetime.timedelta(minutes=5) - datetime.timedelta(microseconds=1)))
        self.assertEqual(throttle({"x": 2}, T0 + datetime.timedelta(minutes=5)), {"x": 2})

    def test_back_to_an_earlier_payload_is_a_change(self):
        throttle = Throttle(datetime.timedelta(minutes=5))
        throttle({"x": 1}, T0)
        throttle({"x": 2}, T0 + datetime.timedelta(minutes=5))
        self.assertIsNone(throttle({"x": 2}, T0 + datetime.timedelta(minutes=20)))
        self.assertEqual(throttle({"x": 1}, T0 + datetime.timedelta(minutes=20)), {"x": 1})


if __name__ == "__main__":
    unittest.main()
