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

import base64
import math
import pathlib
import pickle
import tempfile
import unittest

import mlflow

from algo.model import MAX_WARMUP_HOURS, WarmupModel, ZonesModel, warmup_hours


# cloudpickle, as MLflow stores it, of ZonesModel({"zone:0": WarmupModel(4, 0.15, 7, 0.2, trained_until=5)}, 9) from
# before dead time and outdoor-dependent heating power, when WarmupModel had only a, b, episodes, residual, trained_until.
OLD_MODEL = base64.b64decode(
    "gAWVpAAAAAAAAACMCmFsZ28ubW9kZWyUjApab25lc01vZGVslJOUKYGUfZQojAV6b25lc5R9lIwGem9uZTowlGgAjAtXYXJtdXBNb2RlbJSTlCmB"
    "lH2UKIwBYZRHQBAAAAAAAACMAWKURz/DMzMzMzMzjAhlcGlzb2Rlc5RLB4wIcmVzaWR1YWyURz/JmZmZmZmajA10cmFpbmVkX3VudGlslEsFdWJz"
    "aBBLCXViLg==")


def closed_form(a, b, start, target, outdoor):
    t_inf = outdoor + a / b
    return math.log((t_inf - start) / (t_inf - target)) / b


class TestWarmupHours(unittest.TestCase):
    def test_matches_the_closed_form(self):
        for a, b, start, target, outdoor in ((4, 0.15, 17, 21, 4), (2, 0.05, 16, 21, 5), (6, 0.3, 18, 22, 10)):
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(warmup_hours(a, b, start, target, outdoor),
                                       closed_form(a, b, start, target, outdoor), places=12)

    def test_b_zero_is_linear(self):
        self.assertEqual(warmup_hours(2.0, 0.0, 17, 21, 5), 2.0)

    def test_a_tiny_b_goes_over_into_the_linear_case(self):
        # The closed form loses every digit here: ln of a ratio within 1e-12 of one, divided by 1e-12.
        self.assertAlmostEqual(warmup_hours(2.0, 1e-12, 17, 21, 5), 2.0, places=9)
        self.assertAlmostEqual(warmup_hours(2.0, 1e-300, 17, 21, 5), 2.0, places=12)

    def test_unreachable_when_the_room_settles_at_or_below_the_target(self):
        # T_inf = 5 + 1.6 / 0.1 = 21
        self.assertIsNone(warmup_hours(1.6, 0.1, 17, 21, 5))
        self.assertIsNone(warmup_hours(1.6, 0.1, 17, 21.5, 5))
        self.assertIsNotNone(warmup_hours(1.6, 0.1, 17, 18, 5))

    def test_above_six_hours_is_unreachable(self):
        self.assertEqual(warmup_hours(1.0, 0.0, 15, 15 + MAX_WARMUP_HOURS, 0), MAX_WARMUP_HOURS)
        self.assertIsNone(warmup_hours(1.0, 0.0, 15, 15 + MAX_WARMUP_HOURS + 0.01, 0))

    def test_already_warm_needs_no_time(self):
        self.assertEqual(warmup_hours(4, 0.15, 21, 21, 4), 0.0)
        self.assertEqual(warmup_hours(4, 0.15, 22, 21, 4), 0.0)


class TestWarmupModel(unittest.TestCase):
    MODEL = WarmupModel(2.0, 0.05, 9, 0.1, a1=-0.1, dead_minutes=30, ta_min=-5.0, ta_max=10.0)

    def test_rate_is_held_at_the_ends_of_the_training_range(self):
        self.assertAlmostEqual(self.MODEL.rate(0.0), 2.0)
        self.assertAlmostEqual(self.MODEL.rate(10.0), 1.0)
        self.assertAlmostEqual(self.MODEL.rate(-5.0), 2.5)
        self.assertAlmostEqual(self.MODEL.rate(20.0), 1.0)
        self.assertAlmostEqual(self.MODEL.rate(-15.0), 2.5)
        # With a heating curve there is no range.
        self.assertAlmostEqual(WarmupModel(7.92, 0.18, 9, 0.1, a1=-0.36).rate(-15.0), 7.92 + 0.36 * 15)

    def test_hours_add_the_dead_time_not_yet_elapsed(self):
        # Below the training range the whole model holds -5 °C, the heating power and the loss alike.
        warmup = warmup_hours(2.5, 0.05, 17, 21, -5)
        self.assertAlmostEqual(self.MODEL.hours(17, 21, -10), 0.5 + warmup)
        self.assertAlmostEqual(self.MODEL.hours(17, 21, -10, elapsed_hours=0.2), 0.3 + warmup)
        self.assertAlmostEqual(self.MODEL.hours(17, 21, -10, elapsed_hours=2.0), warmup)
        self.assertAlmostEqual(self.MODEL.hours(17, 21, -10, elapsed_hours=-1.0), 0.5 + warmup)
        self.assertEqual(self.MODEL.hours(21, 21, -10), 0.0)

    def test_the_dead_time_counts_toward_the_limit(self):
        model = WarmupModel(1.0, 0.0, 9, 0.1, dead_minutes=60)
        self.assertEqual(model.hours(15, 20, 0), MAX_WARMUP_HOURS)
        self.assertIsNone(model.hours(15, 20.01, 0))
        self.assertAlmostEqual(model.hours(15, 20.01, 0, elapsed_hours=0.5), MAX_WARMUP_HOURS - 0.5 + 0.01)
        # Unreachable at the outdoor temperature: the room settles at 0 + 1.6/0.1 = 16.
        self.assertIsNone(WarmupModel(1.6, 0.1, 9, 0.1).hours(15, 17, 0))

    def test_invalid_fields(self):
        for kwargs in (dict(b=-0.01), dict(b=math.nan), dict(dead_minutes=15), dict(dead_minutes=70),
                       dict(ta_min=1.0), dict(ta_min=2.0, ta_max=1.0)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    WarmupModel(**{"a0": 1.0, "b": 0.1, "episodes": 3, "residual": 0.1, **kwargs})


class TestModel(unittest.TestCase):
    def test_pickle_round_trip(self):
        model = pickle.loads(pickle.dumps(ZonesModel({"zone:0": WarmupModel(
            4, 0.15, 7, 0.2, trained_until=5, a1=-0.1, dead_minutes=40, ta_min=-3.0, ta_max=9.5)}, 9)))
        zone = model.get("zone:0")
        self.assertEqual((zone.a0, zone.a1, zone.b, zone.dead_minutes, zone.ta_min, zone.ta_max, zone.episodes,
                          zone.residual, zone.trained_until), (4.0, -0.1, 0.15, 40, -3.0, 9.5, 7, 0.2, 5))
        self.assertEqual(model.trained_until, 9)
        self.assertIsNone(model.get("zone:1"))

    def test_a_model_registered_before_dead_time_loads(self):
        model = pickle.loads(OLD_MODEL)
        zone = model.get("zone:0")
        self.assertEqual(vars(zone), {"a0": 4.0, "a1": 0.0, "b": 0.15, "dead_minutes": 0, "ta_min": None,
                                      "ta_max": None, "episodes": 7, "residual": 0.2, "trained_until": 5})
        self.assertEqual(model.trained_until, 9)
        self.assertEqual(zone.hours(17, 21, 4), warmup_hours(4, 0.15, 17, 21, 4))
        self.assertEqual(zone.rate(-20.0), 4.0)

    def test_mlflow_round_trip_and_predict(self):
        model = ZonesModel({"zone:0": WarmupModel(4, 0.15, 7, 0.2)}, trained_until=5)
        with tempfile.TemporaryDirectory() as tmp:
            previous = mlflow.get_tracking_uri()
            self.addCleanup(mlflow.set_tracking_uri, previous)
            mlflow.set_tracking_uri(pathlib.Path(tmp, "mlruns").as_uri())
            with mlflow.start_run():
                info = mlflow.pyfunc.log_model(artifact_path="m", python_model=model)
            loaded = mlflow.pyfunc.load_model(info.model_uri)
            self.assertIsInstance(loaded.unwrap_python_model(), ZonesModel)
            hours, unknown = loaded.predict([{"zone": "zone:0", "start": 17, "target": 21, "outdoor": 4},
                                             {"zone": "zone:1", "start": 17, "target": 21, "outdoor": 4}])
            self.assertAlmostEqual(hours, closed_form(4, 0.15, 17, 21, 4))
            self.assertIsNone(unknown)


if __name__ == "__main__":
    unittest.main()
