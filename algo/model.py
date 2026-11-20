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

__all__ = ("WarmupModel", "ZonesModel", "warmup_hours", "MAX_WARMUP_HOURS", "DEAD_MINUTES")

import math
import typing

import mlflow.pyfunc

# A warm-up that would take longer counts as unreachable.
MAX_WARMUP_HOURS = 6.0
# Dead times the training tries, minutes.
DEAD_MINUTES = (0, 10, 20, 30, 40, 50, 60)


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
    """
    One zone: after `dead_minutes` the room follows dT/dt = a0 + a1 T_out - b (T - T_out), with a0 in K/h, a1 in K/h
    per K and b in 1/h, fitted on `episodes` warm-ups.
    """

    def __init__(self, a0: float, b: float, episodes: int, residual: float, trained_until: typing.Optional[int] = None,
                 *, a1: float = 0.0, dead_minutes: int = 0, ta_min: typing.Optional[float] = None,
                 ta_max: typing.Optional[float] = None):
        if not b >= 0:
            raise ValueError(f"b must not be negative, got {b}")
        if dead_minutes not in DEAD_MINUTES:
            raise ValueError(f"dead_minutes must be one of {DEAD_MINUTES}, got {dead_minutes}")
        if (ta_min is None) != (ta_max is None) or (ta_min is not None and ta_min > ta_max):
            raise ValueError(f"ta_min and ta_max are a range or both None, got {ta_min}, {ta_max}")
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.b = float(b)
        self.dead_minutes = int(dead_minutes)
        # Outdoor range of the training, rate() does not extrapolate beyond it. None with a heating curve.
        self.ta_min = None if ta_min is None else float(ta_min)
        self.ta_max = None if ta_max is None else float(ta_max)
        self.episodes = int(episodes)
        # RMS of the slope residuals, K/h.
        self.residual = float(residual)
        # Wall clock of the training, UTC nanoseconds.
        self.trained_until = None if trained_until is None else int(trained_until)

    def __setstate__(self, state):
        state = dict(state)
        if "a0" not in state:
            # Registered before dead time and outdoor-dependent heating power: a constant a.
            state["a0"] = state.pop("a")
            state.update(a1=0.0, dead_minutes=0, ta_min=None, ta_max=None)
        self.__dict__.update(state)

    def _outdoor(self, outdoor: float) -> float:
        # Without a heating curve the model knows only the trained outdoor range and holds its edge beyond it.
        if self.ta_min is not None:
            return min(max(outdoor, self.ta_min), self.ta_max)
        return outdoor

    def rate(self, outdoor: float) -> float:
        """Heating power over heat capacity at this outdoor temperature, K/h."""
        return self.a0 + self.a1 * self._outdoor(outdoor)

    def hours(self, start: float, target: float, outdoor: float, elapsed_hours: float = 0.0) -> typing.Optional[float]:
        """
        Dead time not yet elapsed plus the warm-up from `start` to `target`. None if unreachable or longer than
        MAX_WARMUP_HOURS. `elapsed_hours` is the time since the heating was asked for.
        """
        if start >= target:
            return 0.0
        outdoor = self._outdoor(outdoor)
        warmup = warmup_hours(self.rate(outdoor), self.b, start, target, outdoor)
        if warmup is None:
            return None
        total = max(0.0, self.dead_minutes / 60 - max(0.0, elapsed_hours)) + warmup
        return None if total > MAX_WARMUP_HOURS else total


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
