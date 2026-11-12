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

__all__ = ("Forecasts", "parse_utc", "plain_number")

import bisect
import datetime
import math
import typing

UTC = datetime.timezone.utc
# Runs kept besides the latest, for hours the latest leaves null.
KEPT_RUNS = 4
# Two forecast hours further apart than this are not interpolated.
MAX_STEP = datetime.timedelta(hours=1)


def parse_utc(value) -> datetime.datetime:
    """An ISO 8601 time as aware UTC; one without zone counts as UTC."""
    if isinstance(value, datetime.datetime):
        parsed = value
    else:
        text = str(value).strip()
        # fromisoformat of Python 3.10 knows no Z.
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        parsed = datetime.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def plain_number(value) -> typing.Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class _Run:
    def __init__(self, times: typing.List[datetime.datetime], values: typing.List[typing.Optional[float]]):
        self.times = times
        self.values = values
        # The forecast hours start one hour after the issue time, so the first one orders the runs.
        self.key = times[0]

    def at(self, t: datetime.datetime) -> typing.Optional[float]:
        i = bisect.bisect_left(self.times, t)
        if i < len(self.times) and self.times[i] == t:
            return self.values[i]
        if i == 0 or i == len(self.times):
            return None
        t0, t1 = self.times[i - 1], self.times[i]
        v0, v1 = self.values[i - 1], self.values[i]
        if v0 is None or v1 is None or t1 - t0 > MAX_STEP:
            return None
        return v0 + (v1 - v0) * ((t - t0) / (t1 - t0))


class Forecasts:
    """The latest MOSMIX runs of one station; a lookup takes the latest run that has a value for the time."""

    def __init__(self):
        self._runs: typing.List[_Run] = []

    def add(self, temperatures, forecasted_for) -> bool:
        """Stores one run; False if the message is no run (no lists, lists of unequal length, bad times)."""
        if not isinstance(temperatures, list) or not isinstance(forecasted_for, list):
            return False
        if len(temperatures) != len(forecasted_for) or not forecasted_for:
            return False
        try:
            pairs = sorted(((parse_utc(t), plain_number(v)) for t, v in zip(forecasted_for, temperatures)),
                           key=lambda p: p[0])
        except (TypeError, ValueError):
            return False
        times = [t for t, _ in pairs]
        if len(set(times)) != len(times):
            return False
        run = _Run(times, [v for _, v in pairs])
        # A run published twice replaces itself.
        self._runs = [r for r in self._runs if r.key != run.key]
        self._runs.append(run)
        self._runs.sort(key=lambda r: r.key)
        del self._runs[:-KEPT_RUNS]
        return True

    def at(self, t: datetime.datetime) -> typing.Optional[float]:
        for run in reversed(self._runs):
            value = run.at(t)
            if value is not None:
                return value
        return None

    def __len__(self):
        return len(self._runs)
