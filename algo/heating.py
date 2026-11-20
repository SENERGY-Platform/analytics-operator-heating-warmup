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

__all__ = ("HeatingCurve", "OffWindow", "parse_heating_curve", "parse_off_window")

import dataclasses
import datetime
import math
import re
import typing

import numpy as np

from .schedule import TIMEZONE

UTC = datetime.timezone.utc
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=UTC)
_DAY = datetime.timedelta(days=1)
_RANGE = re.compile(r"^([0-9]{1,2}):([0-5][0-9])\s*-\s*([0-9]{1,2}):([0-5][0-9])$")


def _blank(value) -> bool:
    return value is None or not str(value).strip()


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


@dataclasses.dataclass(frozen=True)
class HeatingCurve:
    """
    Weather-compensated flow temperature of the boiler as the line through its values at -10 and +10 °C outdoors.
    Two points describe the curve of any make, and a curved one stays within about 1 K of the line between them.
    """
    cold: float  # flow at -10 °C, °C
    mild: float  # flow at +10 °C, °C

    def flow(self, outdoor):
        return self.mild + (self.cold - self.mild) * (10.0 - outdoor) / 20.0


def parse_heating_curve(cold, mild) -> typing.Optional[HeatingCurve]:
    """None when neither is set. One without the other or an invalid value raises ValueError."""
    if _blank(cold) and _blank(mild):
        return None
    if _blank(cold) or _blank(mild):
        raise ValueError("flow_at_minus10 and flow_at_plus10 are set together or not at all")
    c, m = _finite(cold, "flow_at_minus10"), _finite(mild, "flow_at_plus10")
    if c < m:
        raise ValueError(f"flow_at_minus10 must not be below flow_at_plus10, got {cold!r} and {mild!r}")
    return HeatingCurve(c, m)


def _ns(at: datetime.datetime) -> int:
    # Integer arithmetic: a float timestamp loses the nanoseconds.
    return (at - _EPOCH) // datetime.timedelta(microseconds=1) * 1000


@dataclasses.dataclass(frozen=True)
class OffWindow:
    """Daily switch-off of the heating plant in Europe/Berlin wall time. An end at or before the start is the next day."""
    start: int  # minute of the day, 0..1439
    end: int  # minute of the day, 0..1440

    def _instances(self, first: datetime.date, last: datetime.date) -> typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]:
        """The windows that start on the local days first..last, as UTC instants."""
        result = []
        day = first
        while day <= last:
            midnight = datetime.datetime(day.year, day.month, day.day)
            begin = midnight + datetime.timedelta(minutes=self.start)
            end = midnight + datetime.timedelta(minutes=self.end) + (_DAY if self.end <= self.start else datetime.timedelta())
            # A wall time in the spring gap or the repeated autumn hour widens the window to both readings.
            begins = [begin.replace(tzinfo=TIMEZONE, fold=f).astimezone(UTC) for f in (0, 1)]
            ends = [end.replace(tzinfo=TIMEZONE, fold=f).astimezone(UTC) for f in (0, 1)]
            result.append((min(begins), max(ends)))
            day += _DAY
        return result

    def steps(self, times: np.ndarray, step_ns: int) -> np.ndarray:
        """Per grid time t (int64 UTC ns, ascending): whether [t, t + step_ns] touches a window, both ends included."""
        off = np.zeros(len(times), dtype=bool)
        if len(times) == 0:
            return off
        first = datetime.datetime.fromtimestamp(int(times[0]) // 10**9, UTC).astimezone(TIMEZONE).date() - _DAY
        last = datetime.datetime.fromtimestamp(int(times[-1]) // 10**9, UTC).astimezone(TIMEZONE).date() + _DAY
        for begin, end in self._instances(first, last):
            lo = np.searchsorted(times, _ns(begin) - step_ns, side="left")
            hi = np.searchsorted(times, _ns(end), side="right")
            off[lo:hi] = True
        return off

    def resumes(self, now: datetime.datetime) -> typing.Optional[datetime.datetime]:
        """When the heating last came back or, if `now` lies in a window, will come back. None if not within two days."""
        today = now.astimezone(TIMEZONE).date()
        # The window that began last ends last, whether now lies in it or after it.
        ends = [end for begin, end in self._instances(today - 2 * _DAY, today) if begin <= now]
        return max(ends) if ends else None


def parse_off_window(text) -> typing.Optional[OffWindow]:
    """`"HH:MM-HH:MM"`, None when blank, ValueError when invalid or when start and end are the same time of day."""
    if _blank(text):
        return None
    m = _RANGE.match(str(text).strip())
    if not m:
        raise ValueError(f"heating_off must be HH:MM-HH:MM, got {text!r}")
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    if h1 > 23 or h2 > 24 or (h2 == 24 and m2 != 0):
        raise ValueError(f"heating_off must be HH:MM-HH:MM, got {text!r}")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if end % 1440 == start:
        raise ValueError(f"heating_off must not start and end at the same time, got {text!r}")
    return OffWindow(start, end)
