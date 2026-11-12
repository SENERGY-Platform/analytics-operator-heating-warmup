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

__all__ = ("WarmupModel", "ZonesModel", "warmup_hours", "MAX_WARMUP_HOURS")

import math
import typing

import mlflow.pyfunc

# A warm-up that would take longer counts as unreachable.
MAX_WARMUP_HOURS = 6.0


def warmup_hours(a: float, b: float, start: float, target: float, outdoor: float) -> typing.Optional[float]:
    """
    Hours from `start` to `target` under dT/dt = a - b (T - outdoor) at constant outdoor; None if the
    target is not reached or only after MAX_WARMUP_HOURS. 0 when the room is already there.
    """
    if start >= target:
        return 0.0
    # Heating rate at the target; the room approaches outdoor + a/b, which lies above the target iff rate > 0.
    rate = a - b * (target - outdoor)
    if rate <= 0:
        return None
    # ln((T_inf - T0)/(T_inf - T_set)) / b, written with log1p so that it goes over into (T_set - T0)/a as b -> 0.
    x = b * (target - start) / rate
    hours = (target - start) / rate * (math.log1p(x) / x if x != 0 else 1.0)
    if not math.isfinite(hours) or hours > MAX_WARMUP_HOURS:
        return None
    return hours


class WarmupModel:
    """1R1C model of one zone: a in K/h, b in 1/h, fitted on `episodes` warm-ups."""

    def __init__(self, a: float, b: float, episodes: int, residual: float, trained_until: typing.Optional[int] = None):
        self.a = float(a)
        self.b = float(b)
        self.episodes = int(episodes)
        # RMS of the slope residuals, K/h.
        self.residual = float(residual)
        # Wall clock of the training, UTC nanoseconds.
        self.trained_until = None if trained_until is None else int(trained_until)

    def hours(self, start: float, target: float, outdoor: float) -> typing.Optional[float]:
        return warmup_hours(self.a, self.b, start, target, outdoor)


class ZonesModel(mlflow.pyfunc.PythonModel):
    """The models of all zones of one operator, registered in MLflow as one model; a zone without one is absent."""

    def __init__(self, zones: typing.Dict[str, WarmupModel], trained_until: typing.Optional[int] = None):
        self.zones = dict(zones)
        # Wall clock of the training, UTC nanoseconds.
        self.trained_until = None if trained_until is None else int(trained_until)

    def get(self, key: str) -> typing.Optional[WarmupModel]:
        return self.zones.get(key)

    def predict(self, context, model_input, params=None):
        """Warm-up hours (None if unreachable or the zone has no model) for rows of zone, start, target and outdoor."""
        rows = model_input
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict(orient="records")
        if isinstance(rows, dict):
            rows = [rows]
        result = []
        for r in rows:
            zone = self.zones.get(str(r["zone"]))
            result.append(None if zone is None else zone.hours(float(r["start"]), float(r["target"]), float(r["outdoor"])))
        return result
