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

__all__ = ("Operator", "CustomConfig")

import datetime
import json
import math
import re
import time
import typing

import ray

from operator_lib.util import MLOperator, Config, ImportFilter, Inputs, Outputs, logger
from operator_lib.util import clock
from operator_lib.util.helpers import provide_historic_data

from algo.forecast import Forecasts, parse_utc, plain_number
from algo.model import WarmupModel, ZonesModel
from algo.output import Throttle, iso_utc, round_to, whole_minutes
from algo.schedule import TIMEZONE
from algo.training import OUTDOOR, ROOM, SETPOINT, TRAINING_COLUMNS, to_utc_ns, train_zone_models, valid_temperature
from algo.zones import Zone, parse_zones

FORECAST_TEMPERATURE = "forecast_temperature"
FORECASTED_FOR = "forecasted_for"
# Optional inputs from value.meta of the two imports, for the station outputs.
STATION_INPUTS = {
    "measure_station": ("outdoor_station_name", "outdoor_station_lat", "outdoor_station_long"),
    "forecast_station": ("forecast_station_name", "forecast_station_lat", "forecast_station_long"),
}

THROTTLE = datetime.timedelta(minutes=5)
HORIZON = datetime.timedelta(hours=24)
# A live value older than this, relative to the newest message, no longer counts.
LIVE_MAX_AGE = datetime.timedelta(hours=6)
# A history that cannot be read, as when the Ray cluster has no CPU free, is read again this often, this far apart.
HISTORY_RETRY_S = 600
HISTORY_RETRIES = 6

# Message paths and export columns of the station in import-dwd-stations and import-dwd-mosmix.
STATION = dict(path="value.meta.id", column="station_id", near=("lat", "long"),
               near_paths=("value.meta.lat", "value.meta.long"), near_columns=("lat", "long"))
IMPORT_FILTERS = (
    ImportFilter(inputs=[OUTDOOR, *STATION_INPUTS["measure_station"]], value="measure_station", **STATION),
    ImportFilter(inputs=[FORECAST_TEMPERATURE, FORECASTED_FOR, *STATION_INPUTS["forecast_station"]],
                 value="forecast_station", **STATION),
)


def _blank(value) -> bool:
    return value is None or not str(value).strip()


def _number(value) -> typing.Optional[float]:
    try:
        return plain_number(str(value).strip())
    except ValueError:
        return None


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, with the earth radius the library uses for the nearest station."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(h)))


class CustomConfig(Config):
    # Redeclared so the manifest lists it, with the library's default.
    logger_level: str = "warning"

    lat: float = ""
    long: float = ""
    measure_station: str = ""
    forecast_station: str = ""
    # JSON list of {"key", "name", "devices", "schedule", "setback"}; schedule in the format of the heating process.
    zones: str = ""
    history_days: float = 60
    max_lead_minutes: int = 240

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.history_days = 60 if _blank(self.history_days) else float(self.history_days)
        if not math.isfinite(self.history_days) or self.history_days <= 0:
            raise ValueError(f"history_days must be a positive number, got {self.history_days}")
        self.max_lead_minutes = 240 if _blank(self.max_lead_minutes) else int(self.max_lead_minutes)
        if self.max_lead_minutes < 0:
            raise ValueError(f"max_lead_minutes must not be negative, got {self.max_lead_minutes}")
        # Invalid zones stop the operator at start; the library builds an empty config first, so empty passes here.
        if not _blank(self.zones):
            parse_zones(self.zones)


class OperatorInputs(Inputs):
    room_temperature: float
    target_temperature: float
    outdoor_temperature: float
    # Lists of one MOSMIX run; operator-lib knows no list ports.
    forecast_temperature: str
    forecasted_for: str
    outdoor_station_name: str
    outdoor_station_lat: float
    outdoor_station_long: float
    forecast_station_name: str
    forecast_station_lat: float
    forecast_station_long: float


class OperatorOutputs(Outputs):
    # One message per zone.
    zone: str
    zone_name: str
    a: float
    b: float
    episodes: int
    warmup_minutes: int
    next_start: str
    next_target: float
    next_lead_minutes: int
    # JSON list of {"start", "target", "outdoor", "lead_minutes"}.
    windows: str
    # JSON {"id", "name", "distance_km"} of the station in use; null for a device or while unknown.
    measure_station: str
    forecast_station: str


def _dests(topic) -> typing.Set[str]:
    return {m.dest for m in (topic.mappings or [])}


def applicable_filters(declared, topics) -> typing.List[ImportFilter]:
    """
    The filters that have an import input to act on. The library refuses to start on a filter whose inputs are all
    mapped to devices or not mapped at all, which is how a user's own outdoor sensor is wired.
    """
    imported = set()
    for topic in topics:
        if topic.filterType == "ImportId":
            imported |= _dests(topic)
    return [f for f in declared if imported & set(f.inputs)]


def check_station_config(config, filters):
    # Unfiltered, a shared import mixes every station into the outdoor temperature without an error.
    for f in filters:
        if _blank(getattr(config, f.value, None)) and any(_blank(getattr(config, n, None)) for n in f.near):
            raise ValueError(f"{', '.join(f.inputs)} come from a shared import: set '{f.value}' or "
                             f"'{f.near[0]}' and '{f.near[1]}'")


def _unwrap(model) -> typing.Optional[ZonesModel]:
    if model is None:
        return None
    if hasattr(model, "unwrap_python_model"):
        model = model.unwrap_python_model()
    if not isinstance(model, ZonesModel):
        logger.warning(f"ignoring a model of unexpected type {type(model).__name__}")
        return None
    return model


def _maybe_list(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _berlin_day(at: datetime.datetime) -> datetime.date:
    return at.astimezone(TIMEZONE).date()


def _metric_name(key: str) -> str:
    # MLflow allows letters, digits, "_", "-", ".", " " and "/" in a metric name; a zone key like "zone:0" has more.
    return re.sub(r"[^A-Za-z0-9_\-. /]", "_", key)


def _single_device(topic) -> typing.Optional[str]:
    if topic.filterType != "DeviceId":
        return None
    devices = [d.strip() for d in (topic.filterValue or "").split(",") if d.strip()]
    return devices[0] if len(devices) == 1 else None


class Operator(MLOperator):
    configType = CustomConfig
    inputs = OperatorInputs
    outputs = OperatorOutputs
    import_filters = IMPORT_FILTERS

    def __init__(self):
        # Without topics no filter applies; set_topics() decides.
        self.import_filters = []
        self.topics = []
        # Per filter (its config field): the resolved station id and what its messages say about it.
        self.stations: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self.zones: typing.List[Zone] = []
        self.zone_of: typing.Dict[str, str] = {}  # device id -> zone key
        # Per input index: whether its history is read at training, and its device if it has exactly one.
        self.consume: typing.Optional[typing.List[bool]] = None
        self.devices: typing.Optional[typing.List[typing.Optional[str]]] = None
        # Per zone key, per device: the latest value and its time.
        self.room: typing.Dict[str, typing.Dict[str, typing.Tuple[float, datetime.datetime]]] = {}
        self.setpoint: typing.Dict[str, typing.Dict[str, typing.Tuple[float, datetime.datetime]]] = {}
        self.outdoor: typing.Optional[typing.Tuple[float, datetime.datetime]] = None
        self.forecasts = Forecasts()
        self.now: typing.Optional[datetime.datetime] = None
        self.throttles: typing.Dict[str, Throttle] = {}
        self.trained_day: typing.Optional[datetime.date] = None
        self._retry_at = None  # monotonic time of the next training after an unreadable history
        self._history_retries = 0

    def set_topics(self, input_topics, training_topics=()):
        # Called by OperatorLib before init(). The filters are an instance attribute, which super().init() reads.
        super().set_topics(input_topics, training_topics)
        self.topics = list(input_topics) + list(training_topics)
        self.import_filters = applicable_filters(IMPORT_FILTERS, list(input_topics) + list(training_topics))
        inputs = list(input_topics)
        # provide_historic_data returns one dataset per input topic, in this order.
        self.consume = [bool(_dests(t) & set(TRAINING_COLUMNS)) for t in inputs] if inputs else None
        self.devices = [_single_device(t) for t in inputs] if inputs else None

    def set_zones(self, zones: typing.List[Zone]):
        self.zones = list(zones)
        self.zone_of = {d: z.key for z in self.zones for d in z.devices}
        self.room = {z.key: {} for z in self.zones}
        self.setpoint = {z.key: {} for z in self.zones}
        self.throttles = {z.key: Throttle(THROTTLE) for z in self.zones}

    def init(self, *args, **kwargs):
        config = kwargs.get("config") or CustomConfig({})
        check_station_config(config, self.import_filters)
        self.set_zones(parse_zones(config.zones))
        super().init(*args, **kwargs)
        self._resolved_stations()

    def _resolved_stations(self):
        self.stations = {}
        by_inputs = {tuple(r.inputs): r for r in self.resolved_import_filters()}
        for f in self.import_filters:
            resolved = by_inputs.get(tuple(f.inputs))
            if resolved is None or resolved.value is None:
                continue
            self.stations[f.value] = {"id": resolved.value, "name": None, "lat": resolved.latitude,
                                      "long": resolved.longitude, "distance_km": resolved.distance_km}

    def _request_history(self):
        try:
            return provide_historic_data(datetime.timedelta(days=self.config.history_days))
        except Exception as ex:
            logger.info(f"no history available: {type(ex).__name__}: {ex}")
            return None

    def train(self, model, train_logger):
        now = clock.now()
        previous = _unwrap(model)
        datasets = self._request_history()
        if datasets is None:
            results, notes, history_failed = {}, ["no model: no history"], False
        else:
            results, notes, history_failed = ray.get(train_zone_models.remote(
                datasets, self.consume, self.devices, {z.key: list(z.devices) for z in self.zones}, to_utc_ns(now)))
        for note in notes:
            logger.info(note)
        # A zone without a new fit keeps its previous model.
        merged, metrics = {}, {}
        for zone in self.zones:
            fitted, episodes = results.get(zone.key, (None, 0))
            name = _metric_name(zone.key)
            metrics[f"{name}.episodes"] = episodes
            if fitted is not None:
                merged[zone.key] = fitted
                metrics.update({f"{name}.a": fitted.a, f"{name}.b": fitted.b, f"{name}.residual": fitted.residual})
            elif previous is not None and previous.get(zone.key) is not None:
                merged[zone.key] = previous.get(zone.key)
        if train_logger is not None:
            train_logger.log_metrics(metrics)
        new_fits = any(results.get(z.key, (None, 0))[0] is not None for z in self.zones)
        trained = ZonesModel(merged, trained_until=to_utc_ns(now)) if new_fits else None
        if history_failed and self._history_retries < HISTORY_RETRIES:
            # The day stays untrained, so the history is read again.
            self._history_retries += 1
            self._retry_at = time.monotonic() + HISTORY_RETRY_S
            logger.warning(f"history unreadable, training again in {HISTORY_RETRY_S // 60} minutes "
                           f"(retry {self._history_retries} of {HISTORY_RETRIES})")
        else:
            self._history_retries = 0
            self._retry_at = None
            self.trained_day = _berlin_day(now)
        # None keeps the previous model.
        return trained

    def need_retraining(self, model) -> bool:
        if self._retry_at is not None:
            if time.monotonic() < self._retry_at:
                return False
            self._retry_at = None
            return True
        if self.trained_day is None:
            current = _unwrap(model)
            if current is not None and current.trained_until is not None:
                self.trained_day = _berlin_day(datetime.datetime.fromtimestamp(current.trained_until / 1e9, datetime.timezone.utc))
        return self.trained_day != _berlin_day(clock.now())

    @staticmethod
    def _live_mean(values, now) -> typing.Optional[float]:
        fresh = [v for v, at in values.values() if now - at <= LIVE_MAX_AGE]
        return sum(fresh) / len(fresh) if fresh else None

    def _update(self, data, device_id, at: datetime.datetime):
        # A thermostat in no zone is ignored.
        zone = self.zone_of.get(str(device_id).strip()) if device_id is not None else None
        if ROOM in data and zone is not None:
            value = valid_temperature(data[ROOM])
            if value is not None:
                self.room[zone][device_id] = (value, at)
        if SETPOINT in data and zone is not None:
            value = valid_temperature(data[SETPOINT])
            if value is not None:
                self.setpoint[zone][device_id] = (value, at)
        if OUTDOOR in data:
            value = valid_temperature(data[OUTDOOR])
            if value is not None and (self.outdoor is None or at >= self.outdoor[1]):
                self.outdoor = (value, at)
        for key, (name, lat, long) in STATION_INPUTS.items():
            station = self.stations.get(key)
            if station is None:
                continue
            if isinstance(data.get(name), str) and data[name].strip():
                station["name"] = data[name].strip()
            if plain_number(data.get(lat)) is not None and plain_number(data.get(long)) is not None:
                station["lat"], station["long"] = plain_number(data[lat]), plain_number(data[long])
        if FORECAST_TEMPERATURE in data or FORECASTED_FOR in data:
            if not self.forecasts.add(_maybe_list(data.get(FORECAST_TEMPERATURE)), _maybe_list(data.get(FORECASTED_FOR))):
                logger.warning("ignoring a forecast message without two lists of equal length")

    def _lead(self, model: WarmupModel, before: float, target: float, start: datetime.datetime,
              now: datetime.datetime, room: typing.Optional[float], outdoor: typing.Optional[float]) -> typing.Optional[int]:
        if outdoor is None:
            return None
        cap = self.config.max_lead_minutes

        def lead_from(t0):
            hours = model.hours(t0, target, outdoor)
            return cap if hours is None else min(whole_minutes(hours), cap)

        # The room sits at the plan temperature before the boundary, the setback unless another window precedes.
        lead = lead_from(before)
        if room is not None and start - now < datetime.timedelta(minutes=lead):
            lead = lead_from(room)
        return lead

    def _station(self, key: str) -> typing.Optional[str]:
        station = self.stations.get(key)
        if station is None:
            return None
        # The library gives the distance of a station it chose as the nearest; one named in the config is measured here.
        distance = station["distance_km"]
        lat, long = _number(self.config.lat), _number(self.config.long)
        if distance is None and lat is not None and long is not None and station["lat"] is not None:
            distance = distance_km(lat, long, station["lat"], station["long"])
        distance = round_to(distance, "0.1")
        return json.dumps({"id": station["id"], "name": station["name"], "distance_km": distance})

    def _payload(self, zone: Zone, model: typing.Optional[WarmupModel], now: datetime.datetime) -> dict:
        room = self._live_mean(self.room[zone.key], now)
        setpoint = self._live_mean(self.setpoint[zone.key], now)
        measured = self.outdoor[0] if self.outdoor is not None and now - self.outdoor[1] <= LIVE_MAX_AGE else None

        warmup = None
        if model is not None and room is not None and setpoint is not None and setpoint > room and measured is not None:
            hours = model.hours(room, setpoint, measured)
            warmup = -1 if hours is None else whole_minutes(hours)

        next_start = next_target = next_lead = None
        windows = []
        if zone.plan is not None:
            current = zone.plan.target(now)
            for boundary in zone.plan.boundaries(now, HORIZON):
                if boundary.after <= boundary.before:
                    continue
                forecast = self.forecasts.at(boundary.at)
                outdoor = forecast if forecast is not None else measured
                lead = None if model is None else self._lead(
                    model, boundary.before, boundary.after, boundary.at, now, room, outdoor)
                windows.append({"start": iso_utc(boundary.at), "target": round_to(boundary.after, "0.1"),
                                "outdoor": round_to(outdoor, "0.1"), "lead_minutes": lead})
                if next_start is None and boundary.after > current:
                    next_start, next_target, next_lead = iso_utc(boundary.at), round_to(boundary.after, "0.1"), lead

        return {
            "zone": zone.key,
            "zone_name": zone.name,
            "a": None if model is None else round_to(model.a, "0.01"),
            "b": None if model is None else round_to(model.b, "0.001"),
            "episodes": None if model is None else model.episodes,
            "warmup_minutes": warmup,
            "next_start": next_start,
            "next_target": next_target,
            "next_lead_minutes": next_lead,
            "windows": json.dumps(windows),
            "measure_station": self._station("measure_station"),
            "forecast_station": self._station("forecast_station"),
        }

    def infer(self, model, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        at = parse_utc(timestamp)
        # A late message updates the values but does not move the clock of the plan back.
        self.now = at if self.now is None or at > self.now else self.now
        self._update(data, device_id, at)
        models = _unwrap(model)
        results = []
        for zone in self.zones:
            payload = self._payload(zone, None if models is None else models.get(zone.key), self.now)
            passed = self.throttles[zone.key](payload, self.now)
            if passed is not None:
                results.append(passed)
        return None, (results or None), None


from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="analytics-operator-heating-warmup", git_info_file='git_commit')
