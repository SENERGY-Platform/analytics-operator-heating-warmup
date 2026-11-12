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

import math
import pathlib
import pickle
import tempfile
import unittest

import mlflow

from algo.model import MAX_WARMUP_HOURS, WarmupModel, ZonesModel, warmup_hours


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


class TestModel(unittest.TestCase):
    def test_pickle_round_trip(self):
        model = pickle.loads(pickle.dumps(ZonesModel({"zone:0": WarmupModel(4, 0.15, 7, 0.2, trained_until=5)}, 9)))
        zone = model.get("zone:0")
        self.assertEqual((zone.a, zone.b, zone.episodes, zone.residual, zone.trained_until), (4.0, 0.15, 7, 0.2, 5))
        self.assertEqual(model.trained_until, 9)
        self.assertIsNone(model.get("zone:1"))

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
