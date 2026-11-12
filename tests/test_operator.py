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

import contextlib
import datetime
import json
import math
import unittest
from unittest import mock

import pandas as pd

from operator_lib.util import clock
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import import_filter
from operator_lib.util.model import InputTopic

import main
from main import (HISTORY_RETRIES, HISTORY_RETRY_S, IMPORT_FILTERS, CustomConfig, Operator, applicable_filters)
from algo.model import WarmupModel, ZonesModel, warmup_hours
from algo.training import fit_zones
from algo.zones import parse_zones
from tests.simulation import FakeDataset, UnreadableDataset, simulate

UTC = datetime.timezone.utc
BERLIN = datetime.timezone(datetime.timedelta(hours=1))  # January
SCHEDULE = {"windows": [{"from": "06:00", "to": "22:00", "temperature": 21}]}
MODEL = WarmupModel(4.0, 0.15, 9, 0.1, trained_until=0)


def zones_config(schedule=SCHEDULE, setback=17, devices=("d1", "d2")) -> str:
    """One zone "z" of the thermostats d1 and d2."""
    return json.dumps([{"key": "z", "name": "Wohnzimmer", "devices": list(devices), "schedule": schedule,
                        "setback": setback}])


def berlin(text: str) -> datetime.datetime:
    """An aware UTC time from a Berlin wall time in winter."""
    return datetime.datetime.fromisoformat(text).replace(tzinfo=BERLIN).astimezone(UTC)


def minutes(hours: float) -> int:
    return int(math.floor(hours * 60 + 0.5))


def topic(name, filter_type, filter_value, *dests):
    return InputTopic({"name": name, "filterType": filter_type, "filterValue": filter_value,
                       "mappings": [{"dest": d, "source": f"value.{d}"} for d in dests]})


THERMOSTATS = topic("urn_infai_ses_service_thermostat", "DeviceId", "d1,d2", "room_temperature", "target_temperature")
OUTDOOR_DEVICE = topic("urn_infai_ses_service_outdoor", "DeviceId", "d3", "outdoor_temperature")
STATIONS = topic("urn_infai_ses_import_stations", "ImportId", "i1", "outdoor_temperature")
MOSMIX = topic("urn_infai_ses_import_mosmix", "ImportId", "i2", "forecast_temperature", "forecasted_for")
STATIONS_META = topic("urn_infai_ses_import_stations", "ImportId", "i1", "outdoor_temperature", "outdoor_station_name",
                      "outdoor_station_lat", "outdoor_station_long")
MOSMIX_META = topic("urn_infai_ses_import_mosmix", "ImportId", "i2", "forecast_temperature", "forecasted_for",
                    "forecast_station_name", "forecast_station_lat", "forecast_station_long")


EXPORTS = {
    t.name: {"topic": t.name, "import_id": t.filterValue, "export_id": f"e-{t.filterValue}", "table": f"t_{t.filterValue}",
             "columns": {m.source: m.dest for m in t.mappings}}
    for t in (STATIONS, MOSMIX, STATIONS_META, MOSMIX_META)
}


class FakePyFuncModel:
    def __init__(self, python_model):
        self.python_model = python_model

    def unwrap_python_model(self):
        return self.python_model


class OperatorTestCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(clock.set_fixed, None)

    def operator(self, schedule=SCHEDULE, setback=17, **config):
        op = Operator()
        op.config = CustomConfig({"zones": zones_config(schedule, setback), **config})
        op.set_zones(parse_zones(op.config.zones))
        return op

    def send(self, op, at, model=MODEL, device="d1", **data):
        """The message of zone "z" that infer() lets out, or None."""
        wrapped = None if model is None else FakePyFuncModel(ZonesModel({"z": model}))
        _, results, new_model = op.infer(wrapped, data, None, device, at)
        self.assertIsNone(new_model)
        mine = [r for r in (results or []) if r["zone"] == "z"]
        self.assertLessEqual(len(mine), 1)
        return mine[0] if mine else None

    @staticmethod
    def current(op, model=MODEL):
        """The payload of zone "z" as of the newest message, whether or not the throttle let it out."""
        return op._payload(op.zones[0], model, op.now)

    def forecast(self, op, at, issued: str, values):
        start = datetime.datetime.fromisoformat(issued).replace(tzinfo=UTC) + datetime.timedelta(hours=1)
        times = [(start + datetime.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(values))]
        return self.send(op, at, device=None, forecast_temperature=values, forecasted_for=times)


class TestThrottle(OperatorTestCase):
    def test_at_most_one_message_per_five_minutes_and_only_on_a_change(self):
        op = self.operator()
        t0 = berlin("2026-01-12T12:00:00")
        first = self.send(op, t0, room_temperature=20.0, target_temperature=21.0, outdoor_temperature=4.0)
        self.assertIsNotNone(first)
        # A change within five minutes of the last message waits ...
        self.assertIsNone(self.send(op, t0 + datetime.timedelta(minutes=2), room_temperature=20.5))
        self.assertIsNone(self.send(op, t0 + datetime.timedelta(minutes=4, seconds=59), room_temperature=20.5))
        # ... and goes out with the first message five minutes after it.
        later = self.send(op, t0 + datetime.timedelta(minutes=5), room_temperature=20.5)
        self.assertIsNotNone(later)
        self.assertNotEqual(later["warmup_minutes"], first["warmup_minutes"])
        # Same values: nothing, whatever the time.
        self.assertIsNone(self.send(op, t0 + datetime.timedelta(minutes=30), room_temperature=20.5))

    def test_a_change_below_the_rounding_is_no_change(self):
        op = self.operator()
        t0 = berlin("2026-01-12T12:00:00")
        self.send(op, t0, room_temperature=20.0, target_temperature=21.0, outdoor_temperature=4.0)
        # 0.01 K less outdoor changes the warm-up by seconds, not by a whole minute.
        self.assertIsNone(self.send(op, t0 + datetime.timedelta(minutes=6), outdoor_temperature=3.99))


class TestWarmup(OperatorTestCase):
    def test_remaining_time_with_the_measured_outdoor_temperature(self):
        op = self.operator()
        result = self.send(op, berlin("2026-01-12T12:00:00"), room_temperature=19.0, target_temperature=21.0,
                           outdoor_temperature=4.0)
        self.assertEqual(result["warmup_minutes"], minutes(warmup_hours(4.0, 0.15, 19.0, 21.0, 4.0)))
        self.assertEqual((result["a"], result["b"], result["episodes"]), (4.0, 0.15, 9))

    def test_several_thermostats_are_averaged(self):
        op = self.operator()
        at = berlin("2026-01-12T12:00:00")
        self.send(op, at, device="d1", room_temperature=18.0, target_temperature=21.0, outdoor_temperature=4.0)
        result = self.send(op, at + main.THROTTLE, device="d2", room_temperature=20.0, target_temperature=21.0)
        self.assertEqual(result["warmup_minutes"], minutes(warmup_hours(4.0, 0.15, 19.0, 21.0, 4.0)))

    def test_a_stale_thermostat_no_longer_counts(self):
        op = self.operator()
        at = berlin("2026-01-12T06:00:00")
        self.send(op, at, device="d1", room_temperature=16.0, target_temperature=21.0, outdoor_temperature=4.0)
        later = at + main.LIVE_MAX_AGE + datetime.timedelta(minutes=1)
        result = self.send(op, later, device="d2", room_temperature=20.0, target_temperature=21.0, outdoor_temperature=4.0)
        self.assertEqual(result["warmup_minutes"], minutes(warmup_hours(4.0, 0.15, 20.0, 21.0, 4.0)))

    def test_null_when_warm_without_model_or_outdoor(self):
        at = berlin("2026-01-12T12:00:00")
        result = self.send(self.operator(), at, room_temperature=21.0, target_temperature=21.0, outdoor_temperature=4.0)
        self.assertIsNone(result["warmup_minutes"])
        result = self.send(self.operator(), at, model=None, room_temperature=19.0, target_temperature=21.0,
                           outdoor_temperature=4.0)
        self.assertEqual({k: result[k] for k in ("a", "b", "episodes", "warmup_minutes", "next_lead_minutes")},
                         dict.fromkeys(("a", "b", "episodes", "warmup_minutes", "next_lead_minutes")))
        result = self.send(self.operator(), at, room_temperature=19.0, target_temperature=21.0)
        self.assertIsNone(result["warmup_minutes"])

    def test_unreachable_is_minus_one(self):
        weak = WarmupModel(1.0, 0.15, 9, 0.1)
        result = self.send(self.operator(), berlin("2026-01-12T12:00:00"), model=weak, room_temperature=17.0,
                           target_temperature=21.0, outdoor_temperature=4.0)
        self.assertEqual(result["warmup_minutes"], -1)

    def test_a_late_outdoor_value_does_not_replace_a_newer_one(self):
        op = self.operator()
        at = berlin("2026-01-12T12:00:00")
        self.send(op, at, device=None, outdoor_temperature=4.0)
        self.send(op, at - datetime.timedelta(minutes=30), device=None, outdoor_temperature=9.0)
        self.assertEqual(op.outdoor, (4.0, at))

    def test_dwd_missing_value_is_no_outdoor_temperature(self):
        op = self.operator()
        at = berlin("2026-01-12T12:00:00")
        self.send(op, at, room_temperature=19.0, target_temperature=21.0, outdoor_temperature=4.0)
        result = self.send(op, at + datetime.timedelta(minutes=10), device=None, outdoor_temperature=-999.0)
        self.assertIsNone(result)  # nothing changed
        self.assertEqual(op.outdoor[0], 4.0)


class TestZones(OperatorTestCase):
    def two_zones(self):
        op = Operator()
        op.config = CustomConfig({"zones": json.dumps([
            {"key": "zone:0", "name": "Salon", "devices": ["d1"], "schedule": SCHEDULE, "setback": 17},
            {"key": "zone:1", "name": "Bad", "devices": ["d2"], "schedule": SCHEDULE, "setback": 18}])})
        op.set_zones(parse_zones(op.config.zones))
        return op

    def infer(self, op, at, device, models=None, **data):
        wrapped = None if models is None else FakePyFuncModel(ZonesModel(models))
        _, results, _ = op.infer(wrapped, data, None, device, at)
        return {r["zone"]: r for r in (results or [])}

    def test_one_message_per_zone_with_its_own_model_and_thermostats(self):
        op = self.two_zones()
        at = berlin("2026-01-12T12:00:00")
        out = self.infer(op, at, None, outdoor_temperature=4.0)
        out.update(self.infer(op, at + main.THROTTLE, "d2", room_temperature=20.0, target_temperature=21.0))
        out.update(self.infer(op, at + 2 * main.THROTTLE, "d1", models={"zone:0": MODEL},
                              room_temperature=19.0, target_temperature=21.0))
        self.assertEqual(set(out), {"zone:0", "zone:1"})
        self.assertEqual((out["zone:0"]["zone_name"], out["zone:1"]["zone_name"]), ("Salon", "Bad"))
        self.assertEqual(out["zone:0"]["warmup_minutes"], minutes(warmup_hours(4.0, 0.15, 19.0, 21.0, 4.0)))
        # zone:1 has no model of its own.
        self.assertEqual((out["zone:1"]["a"], out["zone:1"]["warmup_minutes"]), (None, None))
        for result in out.values():
            self.assertIn("measure_station", result)
            self.assertIn("forecast_station", result)

    def test_a_thermostat_in_no_zone_is_ignored(self):
        op = self.two_zones()
        at = berlin("2026-01-12T12:00:00")
        self.infer(op, at, "d9", room_temperature=10.0, target_temperature=25.0)
        self.assertEqual((op.room, op.setpoint), ({"zone:0": {}, "zone:1": {}}, {"zone:0": {}, "zone:1": {}}))

    def test_each_zone_is_throttled_on_its_own(self):
        op = self.two_zones()
        t0 = berlin("2026-01-12T12:00:00")
        models = {"zone:0": MODEL, "zone:1": MODEL}
        minute = datetime.timedelta(minutes=1)
        self.assertEqual(set(self.infer(op, t0, None, models, outdoor_temperature=4.0)), {"zone:0", "zone:1"})
        self.assertEqual(set(self.infer(op, t0 + 5 * minute, "d1", models, room_temperature=19.0,
                                        target_temperature=21.0)), {"zone:0"})
        # zone:1 last went out at t0, zone:0 at t0 + 5 min.
        self.assertEqual(set(self.infer(op, t0 + 7 * minute, "d2", models, room_temperature=19.0,
                                        target_temperature=21.0)), {"zone:1"})
        self.assertEqual(self.infer(op, t0 + 8 * minute, "d1", models, room_temperature=19.5), {})
        self.assertEqual(set(self.infer(op, t0 + 10 * minute, "d1", models, room_temperature=19.5)), {"zone:0"})

    def test_zones_config_is_checked(self):
        for zones in ([{"key": "a", "devices": ["d1"]}, {"key": "a", "devices": ["d2"]}],
                      [{"key": "a", "devices": ["d1"]}, {"key": "b", "devices": ["d1"]}],
                      [{"key": "", "devices": ["d1"]}],
                      [{"key": "a", "devices": "d1"}],
                      [{"key": "a", "devices": ["d1"], "schedule": SCHEDULE}]):
            with self.subTest(zones=zones):
                with self.assertRaises(ValueError):
                    CustomConfig({"zones": json.dumps(zones)})
        for empty in ("", "[]"):
            with self.assertRaises(ValueError):
                parse_zones(empty)
        [zone] = parse_zones(json.dumps([{"key": " k ", "devices": [" d1 ", "d1"]}]))
        self.assertEqual((zone.key, zone.name, zone.devices, zone.plan), ("k", "k", ("d1",), None))


class TestNextWindow(OperatorTestCase):
    def test_next_start_with_the_forecast_at_its_time(self):
        op = self.operator()
        now = berlin("2026-01-12T23:00:00")
        # Issued 21:00 UTC: hours from 22:00 UTC; 06:00 Berlin is 05:00 UTC, the 8th entry.
        self.forecast(op, now - main.THROTTLE, "2026-01-12T21:00:00", [5, 5, 5, 5, 5, 5, 5, 3.0, 5])
        self.send(op, now, room_temperature=17.0, target_temperature=17.0, outdoor_temperature=6.0)
        result = self.current(op)
        self.assertEqual((result["next_start"], result["next_target"]), ("2026-01-13T05:00:00Z", 21.0))
        self.assertEqual(result["next_lead_minutes"], minutes(warmup_hours(4.0, 0.15, 17.0, 21.0, 3.0)))
        self.assertEqual(json.loads(result["windows"]), [
            {"start": "2026-01-13T05:00:00Z", "target": 21.0, "outdoor": 3.0,
             "lead_minutes": result["next_lead_minutes"]}])

    def test_zero_degrees_forecast_is_used(self):
        op = self.operator()
        now = berlin("2026-01-12T23:00:00")
        self.forecast(op, now - main.THROTTLE, "2026-01-12T21:00:00", [5, 5, 5, 5, 5, 5, 5, 0.0, 5])
        self.send(op, now, room_temperature=17.0, target_temperature=17.0, outdoor_temperature=6.0)
        result = self.current(op)
        self.assertEqual(json.loads(result["windows"])[0]["outdoor"], 0.0)
        self.assertEqual(result["next_lead_minutes"], minutes(warmup_hours(4.0, 0.15, 17.0, 21.0, 0.0)))

    def test_without_forecast_the_measured_value(self):
        op = self.operator()
        now = berlin("2026-01-12T23:00:00")
        # This run ends before 05:00 UTC.
        self.forecast(op, now - main.THROTTLE, "2026-01-12T21:00:00", [5, 5, 5])
        self.send(op, now, room_temperature=17.0, target_temperature=17.0, outdoor_temperature=6.0)
        result = self.current(op)
        self.assertEqual(json.loads(result["windows"])[0]["outdoor"], 6.0)
        self.assertEqual(result["next_lead_minutes"], minutes(warmup_hours(4.0, 0.15, 17.0, 21.0, 6.0)))

    def test_the_lead_is_capped(self):
        op = self.operator(max_lead_minutes="30")
        result = self.send(op, berlin("2026-01-12T23:00:00"), room_temperature=17.0, target_temperature=17.0,
                           outdoor_temperature=6.0)
        self.assertEqual(result["next_lead_minutes"], 30)

    def test_an_unreachable_target_takes_the_whole_lead(self):
        op = self.operator()
        result = self.send(op, berlin("2026-01-12T23:00:00"), model=WarmupModel(1.0, 0.15, 9, 0.1),
                           room_temperature=17.0, target_temperature=17.0, outdoor_temperature=6.0)
        self.assertEqual(result["next_lead_minutes"], 240)

    def test_a_window_closer_than_its_lead_starts_from_the_room(self):
        op = self.operator()
        from_setback = minutes(warmup_hours(4.0, 0.15, 17.0, 21.0, 6.0))
        now = berlin("2026-01-13T06:00:00") - datetime.timedelta(minutes=from_setback - 10)
        result = self.send(op, now, room_temperature=19.5, target_temperature=17.0, outdoor_temperature=6.0)
        self.assertEqual(result["next_lead_minutes"], minutes(warmup_hours(4.0, 0.15, 19.5, 21.0, 6.0)))

        far = berlin("2026-01-13T06:00:00") - datetime.timedelta(minutes=from_setback + 10)
        result = self.send(self.operator(), far, room_temperature=19.5, target_temperature=17.0, outdoor_temperature=6.0)
        self.assertEqual(result["next_lead_minutes"], from_setback)

    def test_a_step_up_from_a_lower_window_starts_from_that_window(self):
        schedule = json.dumps({"windows": [{"from": "06:00", "to": "22:00", "temperature": 19},
                                           {"from": "16:00", "to": "22:00", "temperature": 21}]})
        op = self.operator(schedule=schedule)
        result = self.send(op, berlin("2026-01-12T10:00:00"), room_temperature=19.0, target_temperature=19.0,
                           outdoor_temperature=6.0)
        self.assertEqual((result["next_start"], result["next_target"]), ("2026-01-12T15:00:00Z", 21.0))
        self.assertEqual(result["next_lead_minutes"], minutes(warmup_hours(4.0, 0.15, 19.0, 21.0, 6.0)))

    def test_next_start_is_the_first_rise_above_the_current_plan(self):
        # At 21:00 the plan is 21; the 23:00 rise to 19 stays below it, the next morning's 22 does not.
        schedule = json.dumps({"windows": [{"from": "06:00", "to": "22:00", "temperature": 21},
                                           {"from": "23:00", "to": "01:00", "temperature": 19},
                                           {"from": "06:00", "to": "08:00", "temperature": 22}]})
        op = self.operator(schedule=schedule)
        result = self.send(op, berlin("2026-01-12T21:00:00"), room_temperature=21.0, target_temperature=21.0,
                           outdoor_temperature=6.0)
        self.assertEqual((result["next_start"], result["next_target"]), ("2026-01-13T05:00:00Z", 22.0))
        self.assertEqual([w["start"] for w in json.loads(result["windows"])],
                         ["2026-01-12T22:00:00Z", "2026-01-13T05:00:00Z"])

    def test_nothing_above_the_current_plan_within_a_day(self):
        op = self.operator()
        result = self.send(op, berlin("2026-01-12T07:00:00"), room_temperature=21.0, target_temperature=21.0,
                           outdoor_temperature=6.0)
        self.assertEqual((result["next_start"], result["next_target"], result["next_lead_minutes"]), (None, None, None))
        # The next morning is still a rising boundary for the day view.
        self.assertEqual([w["start"] for w in json.loads(result["windows"])], ["2026-01-13T05:00:00Z"])
        result = self.send(self.operator(), berlin("2026-01-12T05:00:00"), room_temperature=21.0)
        self.assertEqual([w["start"] for w in json.loads(result["windows"])], ["2026-01-12T05:00:00Z"])

    def test_a_late_message_does_not_move_the_plan_back(self):
        op = self.operator()
        self.send(op, berlin("2026-01-12T07:00:00"), room_temperature=21.0, target_temperature=21.0,
                  outdoor_temperature=6.0)
        result = self.send(op, berlin("2026-01-12T05:00:00"), room_temperature=20.0)
        self.assertEqual(op.now, berlin("2026-01-12T07:00:00"))
        self.assertIsNone(result)

    def test_without_a_plan_nothing_is_ahead(self):
        op = self.operator(schedule=None, setback=None)
        result = self.send(op, berlin("2026-01-12T07:00:00"), room_temperature=20.0, target_temperature=21.0,
                           outdoor_temperature=6.0)
        self.assertEqual((result["next_start"], result["windows"]), (None, "[]"))
        self.assertIsNotNone(result["warmup_minutes"])


class _LoggerMock:
    def __init__(self, *args, **kwargs):
        pass

    def trace(self, name):
        return contextlib.nullcontext()

    def log_metrics(self, metrics, step=None):
        pass


@contextlib.contextmanager
def training_without_ray(history=None, history_error=None):
    if history_error is not None:
        request = mock.patch("main.provide_historic_data", side_effect=history_error)
    else:
        request = mock.patch("main.provide_historic_data", return_value=history)
    with request, \
         mock.patch("main.train_zone_models", mock.Mock(remote=fit_zones)), \
         mock.patch("main.ray.get", side_effect=lambda result: result):
        yield


def outdoor_by_day(day):
    return [-4.0, 8.0, 2.0, 11.0, -1.0][day % 5]


class TestStart(unittest.TestCase):
    """init() with the topics of a deployment, the library's MLflow and Ray mocked."""

    def setUp(self):
        import_filter.clear()
        self.addCleanup(import_filter.clear)
        self.addCleanup(clock.set_fixed, None)
        # The library checks the export's columns in the database at start; there is none here.
        patch = mock.patch.object(import_filter, "_check_columns")
        self.check_columns = patch.start()
        self.addCleanup(patch.stop)

    def init(self, topics, history=None, **config):
        config.setdefault("import_exports", json.dumps([EXPORTS[t.name] for t in topics if t.name in EXPORTS]))
        op = Operator()
        registry = []
        with mock.patch("operator_lib.util.op_ml.ray"), \
             mock.patch("operator_lib.util.op_ml.mlflow") as mock_mlflow, \
             mock.patch("operator_lib.util.op_ml.MlflowClient"), \
             mock.patch("operator_lib.util.op_ml.TrainMlflowLogger", _LoggerMock), \
             training_without_ray(history if history is not None else []):
            def load_model(uri):
                if not registry:
                    raise RuntimeError("no registered model")
                return FakePyFuncModel(registry[-1])

            mock_mlflow.pyfunc.load_model.side_effect = load_model
            mock_mlflow.pyfunc.log_model.side_effect = lambda artifact_path, python_model: (
                registry.append(python_model), mock.MagicMock())[1]
            op.set_topics(topics, [])
            op.init(kafka_consumer=None, kafka_producer=None, filter_handler=None, output_topic=None,
                    pipeline_id="p", operator_id="o",
                    config=CustomConfig({"zones": zones_config(),
                                         "training_end": "2026-01-10T12:00:00Z", **config}))
        return op, registry

    def test_an_outdoor_device_needs_no_filter(self):
        op, _ = self.init([THERMOSTATS, OUTDOOR_DEVICE, MOSMIX], forecast_station="10469")
        self.assertEqual(op.import_filters, [IMPORT_FILTERS[1]])
        self.assertEqual(import_filter.filter_for(MOSMIX).value, "10469")
        self.assertIsNone(import_filter.filter_for(OUTDOOR_DEVICE))

    def test_the_library_alone_refuses_a_filter_on_a_device_input(self):
        # Why init() drops such filters: resolved as declared, the start fails.
        with self.assertRaisesRegex(MissingConfigValueError, "none of which is an import input"):
            import_filter.resolve_import_filters(
                list(IMPORT_FILTERS), CustomConfig({"measure_station": "1", "forecast_station": "2",
                                                    "import_exports": json.dumps([EXPORTS[MOSMIX.name]])}),
                mock.Mock(), [OUTDOOR_DEVICE, MOSMIX], [])

    def test_no_forecast_input_and_no_outdoor_import_need_no_filter(self):
        op, _ = self.init([THERMOSTATS, OUTDOOR_DEVICE])
        self.assertEqual(op.import_filters, [])

    def test_both_imports_get_their_station(self):
        self.init([THERMOSTATS, STATIONS, MOSMIX], measure_station="02932", forecast_station="10469")
        self.assertEqual(import_filter.filter_for(STATIONS).value, "02932")
        self.assertEqual(import_filter.filter_for(MOSMIX).value, "10469")
        self.assertEqual({c.args[-1][0] for c in self.check_columns.call_args_list}, {"station_id"})

    def test_a_shared_import_without_station_or_location_stops_the_start(self):
        with self.assertRaisesRegex(ValueError, "measure_station"):
            self.init([THERMOSTATS, STATIONS, MOSMIX], forecast_station="10469")
        with self.assertRaisesRegex(ValueError, "forecast_station"):
            self.init([THERMOSTATS, STATIONS, MOSMIX], measure_station="1", lat="51.3")

    def test_the_stations_in_use_are_reported(self):
        op, _ = self.init([THERMOSTATS, STATIONS_META, MOSMIX_META], measure_station="02932",
                          forecast_station="10469", lat="52.0", long="12.0")
        at = datetime.datetime(2026, 1, 12, 12, tzinfo=UTC)
        payload = op._payload(op.zones[0], None, at)
        self.assertEqual(json.loads(payload["measure_station"]), {"id": "02932", "name": None, "distance_km": None})
        self.assertEqual(json.loads(payload["forecast_station"]), {"id": "10469", "name": None, "distance_km": None})

        op.infer(None, {"outdoor_temperature": 3.0, "outdoor_station_name": "Leipzig/Halle",
                        "outdoor_station_lat": 51.0, "outdoor_station_long": 12.0}, None, None, at)
        op.infer(None, {"forecast_temperature": [1.0], "forecasted_for": ["2026-01-12T13:00:00Z"],
                        "forecast_station_name": "LEIPZIG", "forecast_station_lat": "52.0",
                        "forecast_station_long": "12.0"}, None, None, at)
        payload = op._payload(op.zones[0], None, at)
        # One degree of latitude is 111.2 km.
        self.assertEqual(json.loads(payload["measure_station"]), {"id": "02932", "name": "Leipzig/Halle", "distance_km": 111.2})
        self.assertEqual(json.loads(payload["forecast_station"]), {"id": "10469", "name": "LEIPZIG", "distance_km": 0.0})

    def test_a_nearest_station_keeps_the_distance_the_library_found(self):
        found = [import_filter.ResolvedImportFilter(IMPORT_FILTERS[0].path, IMPORT_FILTERS[0].inputs, "02932", "near",
                                                    distance_km=8.04, latitude=51.43, longitude=12.24),
                 import_filter.ResolvedImportFilter(IMPORT_FILTERS[1].path, IMPORT_FILTERS[1].inputs, None, "none")]
        with mock.patch.object(Operator, "resolved_import_filters", return_value=found):
            op, _ = self.init([THERMOSTATS, STATIONS_META, MOSMIX_META], measure_station="02932",
                              forecast_station="10469", lat="52.0", long="12.0")
        payload = op._payload(op.zones[0], None, datetime.datetime(2026, 1, 12, 12, tzinfo=UTC))
        self.assertEqual(json.loads(payload["measure_station"]), {"id": "02932", "name": None, "distance_km": 8.0})
        self.assertIsNone(payload["forecast_station"])

    def test_a_device_as_outdoor_source_has_no_station(self):
        op, _ = self.init([THERMOSTATS, OUTDOOR_DEVICE, MOSMIX_META], forecast_station="10469")
        payload = op._payload(op.zones[0], None, datetime.datetime(2026, 1, 12, 12, tzinfo=UTC))
        self.assertIsNone(payload["measure_station"])
        self.assertEqual(json.loads(payload["forecast_station"])["id"], "10469")

    def test_the_forecast_history_is_not_read(self):
        thermostat, weather = simulate(4.0, 0.15, days=5, outdoor=outdoor_by_day)
        op, registry = self.init([THERMOSTATS, OUTDOOR_DEVICE, MOSMIX], forecast_station="1",
                                 history=[FakeDataset(thermostat), FakeDataset(weather), UnreadableDataset()])
        self.assertEqual(op.consume, [True, True, False])
        self.assertEqual(len(registry), 1)
        self.assertAlmostEqual(registry[0].get("z").a, 4.0, delta=0.4)
        self.assertEqual(op.trained_day, datetime.date(2026, 1, 10))


class TestRetraining(unittest.TestCase):
    def setUp(self):
        self.addCleanup(clock.set_fixed, None)

    def train(self, op, history=None, history_error=None, monotonic=1000.0, now=datetime.datetime(2026, 1, 12, 23, 30, tzinfo=UTC)):
        clock.set_fixed(now)
        logger = mock.Mock()
        with training_without_ray(history, history_error), mock.patch("main.time.monotonic", return_value=monotonic):
            return op.train(None, logger), logger

    def needs(self, op, monotonic=1000.0, now=None, model=None):
        if now is not None:
            clock.set_fixed(now)
        with mock.patch("main.time.monotonic", return_value=monotonic):
            return op.need_retraining(model)

    def operator(self):
        op = Operator()
        op.config = CustomConfig({"zones": zones_config()})
        op.set_zones(parse_zones(op.config.zones))
        return op

    def test_once_a_day_by_the_berlin_date(self):
        op = self.operator()
        # 23:30 UTC is already the next day in Berlin.
        self.train(op, history=[], now=datetime.datetime(2026, 1, 12, 23, 30, tzinfo=UTC))
        self.assertEqual(op.trained_day, datetime.date(2026, 1, 13))
        self.assertFalse(self.needs(op, now=datetime.datetime(2026, 1, 13, 22, 59, tzinfo=UTC)))
        self.assertTrue(self.needs(op, now=datetime.datetime(2026, 1, 13, 23, 0, tzinfo=UTC)))

    def test_a_registered_model_counts_for_its_day(self):
        op = self.operator()
        trained = int(datetime.datetime(2026, 1, 13, 3, tzinfo=UTC).timestamp()) * 10**9
        model = FakePyFuncModel(ZonesModel({"z": WarmupModel(4, 0.15, 9, 0.1)}, trained_until=trained))
        self.assertFalse(self.needs(op, now=datetime.datetime(2026, 1, 13, 12, tzinfo=UTC), model=model))
        self.assertTrue(self.needs(op, now=datetime.datetime(2026, 1, 14, 12, tzinfo=UTC), model=model))

    def test_an_unreadable_history_is_read_again_after_a_pause(self):
        op = self.operator()
        model, _ = self.train(op, history=[UnreadableDataset()], monotonic=1000.0)
        self.assertIsNone(model)
        self.assertIsNone(op.trained_day)
        self.assertFalse(self.needs(op, monotonic=1000.0 + HISTORY_RETRY_S - 1))
        self.assertTrue(self.needs(op, monotonic=1000.0 + HISTORY_RETRY_S))

    def test_the_day_counts_as_trained_after_the_last_retry(self):
        op = self.operator()
        for attempt in range(HISTORY_RETRIES):
            self.train(op, history=[UnreadableDataset()], monotonic=1000.0 * attempt)
            self.assertIsNone(op.trained_day, attempt)
        self.train(op, history=[UnreadableDataset()], monotonic=1000.0 * HISTORY_RETRIES)
        self.assertIsNotNone(op.trained_day)

    def test_a_history_that_cannot_be_requested_is_not_retried(self):
        op = self.operator()
        self.train(op, history_error=RuntimeError("no deployment configuration"))
        self.assertIsNotNone(op.trained_day)
        self.assertIsNone(op._retry_at)

    def test_a_zone_without_a_new_fit_keeps_its_previous_model(self):
        op = Operator()
        op.config = CustomConfig({"zones": json.dumps([{"key": "z", "devices": ["d1"]}, {"key": "y", "devices": ["d2"]}])})
        op.set_zones(parse_zones(op.config.zones))
        living, weather = simulate(4.0, 0.15, days=5, outdoor=outdoor_by_day, device="d1")
        bath, _ = simulate(4.0, 0.15, days=1, outdoor=outdoor_by_day, device="d2")
        old_y = WarmupModel(3.0, 0.1, 4, 0.1)
        previous = FakePyFuncModel(ZonesModel({"z": WarmupModel(1.0, 0.1, 4, 0.1), "y": old_y, "gone": old_y}))
        clock.set_fixed(datetime.datetime(2026, 1, 12, 12, tzinfo=UTC))
        with training_without_ray([FakeDataset(pd.concat([living, bath], ignore_index=True)), FakeDataset(weather)]):
            model = op.train(previous, None)
        self.assertAlmostEqual(model.get("z").a, 4.0, delta=0.4)
        self.assertIs(model.get("y"), old_y)
        self.assertIsNone(model.get("gone"))

        # No zone with a new fit: nothing new to register.
        with training_without_ray([FakeDataset(bath), FakeDataset(weather)]):
            self.assertIsNone(op.train(previous, None))

    def test_a_zone_key_becomes_a_valid_metric_name(self):
        op = Operator()
        op.config = CustomConfig({"zones": json.dumps([{"key": "zone:0", "devices": ["d1"]}])})
        op.set_zones(parse_zones(op.config.zones))
        _, logger = self.train(op, history=[])
        logger.log_metrics.assert_called_once_with({"zone_0.episodes": 0})

    def test_metrics_go_to_mlflow(self):
        op = self.operator()
        thermostat, weather = simulate(4.0, 0.15, days=5, outdoor=outdoor_by_day)
        model, logger = self.train(op, history=[FakeDataset(thermostat), FakeDataset(weather)])
        zone = model.get("z")
        logger.log_metrics.assert_called_once_with(
            {"z.episodes": 5, "z.a": zone.a, "z.b": zone.b, "z.residual": zone.residual})

        _, logger = self.train(self.operator(), history=[FakeDataset(thermostat.iloc[:300]), FakeDataset(weather)])
        logger.log_metrics.assert_called_once_with({"z.episodes": 1})


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        config = CustomConfig({})
        self.assertEqual((config.history_days, config.max_lead_minutes, config.zones, config.lat), (60, 240, "", ""))

    def test_strings_are_parsed(self):
        config = CustomConfig({"history_days": "30", "max_lead_minutes": "120", "history_days ": ""})
        self.assertEqual((config.history_days, config.max_lead_minutes), (30.0, 120))
        self.assertEqual(CustomConfig({"history_days": "", "max_lead_minutes": " "}).max_lead_minutes, 240)

    def test_invalid_values_stop_the_start(self):
        for values in ({"history_days": "x"}, {"history_days": "0"}, {"history_days": "nan"},
                       {"max_lead_minutes": "-1"}, {"max_lead_minutes": "1.5"},
                       {"zones": zones_config(setback="warm")}, {"zones": "{"}, {"zones": "[]"}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CustomConfig(values)


class TestTopics(unittest.TestCase):
    def test_per_input_whether_it_is_read_and_its_single_device(self):
        op = Operator()
        single = topic("urn_infai_ses_service_thermostat", "DeviceId", " d5 ", "room_temperature", "target_temperature")
        op.set_topics([single, THERMOSTATS, MOSMIX, OUTDOOR_DEVICE], [])
        self.assertEqual(op.consume, [True, True, False, True])
        self.assertEqual(op.devices, ["d5", None, None, "d3"])


class TestApplicableFilters(unittest.TestCase):
    def test_by_mapping_of_import_topics(self):
        self.assertEqual(applicable_filters(IMPORT_FILTERS, [STATIONS, MOSMIX]), list(IMPORT_FILTERS))
        self.assertEqual(applicable_filters(IMPORT_FILTERS, [STATIONS]), [IMPORT_FILTERS[0]])
        self.assertEqual(applicable_filters(IMPORT_FILTERS, [THERMOSTATS, OUTDOOR_DEVICE]), [])


if __name__ == "__main__":
    unittest.main()
