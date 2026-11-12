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

__all__ = ("Plan", "Window", "Boundary", "parse_plan", "TIMEZONE")

import dataclasses
import datetime
import json
import math
import re
import typing
import zoneinfo

TIMEZONE = zoneinfo.ZoneInfo("Europe/Berlin")
UTC = datetime.timezone.utc
ALL_DAYS = (0, 1, 2, 3, 4, 5, 6)
_TIME = re.compile(r"^([01]?[0-9]|2[0-4]):([0-5][0-9])$")
_STEP = datetime.timedelta(seconds=1)


@dataclasses.dataclass(frozen=True)
class Window:
    days: typing.Tuple[int, ...]  # 0 Monday .. 6 Sunday
    start: int  # minute of the day, 0..1440
    end: int
    temperature: float


@dataclasses.dataclass(frozen=True)
class Boundary:
    at: datetime.datetime  # aware UTC
    before: float  # plan temperature just before
    after: float


def _minute(text, name: str) -> int:
    m = _TIME.match(str(text))
    if not m or (int(m.group(1)) == 24 and int(m.group(2)) != 0):
        raise ValueError(f"window {name}: time hh:mm expected, not {text!r}")
    return int(m.group(1)) * 60 + int(m.group(2))


def _number(value) -> typing.Optional[float]:
    # As JavaScript's Number() in the heating process, without its reading of true as 1.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _weekday(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= 6
    return isinstance(value, float) and value.is_integer() and 0 <= value <= 6


def parse_plan(schedule, setback) -> typing.Optional["Plan"]:
    """
    The plan of the heating process, from its JSON text or the parsed object; None when neither is configured.
    Invalid input raises ValueError.
    """
    if not str(schedule or "").strip():
        if str(setback if setback is not None else "").strip():
            raise ValueError("setback is set but schedule is empty")
        return None
    low = None if str(setback if setback is not None else "").strip() == "" else _number(setback)
    if low is None:
        raise ValueError(f"setback is not a number: {setback!r}")
    if isinstance(schedule, dict):
        plan = schedule
    else:
        try:
            plan = json.loads(schedule)
        except (TypeError, ValueError) as ex:
            raise ValueError(f"schedule is no JSON: {ex}") from None
    if not isinstance(plan, dict) or not isinstance(plan.get("windows"), list):
        raise ValueError("schedule needs a list windows")
    windows = []
    for k, w in enumerate(plan["windows"]):
        name = str(k + 1)
        if not isinstance(w, dict):
            raise ValueError(f"window {name}: object expected")
        days = w.get("weekdays")
        if days is None or (isinstance(days, list) and len(days) == 0):
            days = list(ALL_DAYS)
        if not isinstance(days, list):
            raise ValueError(f"window {name}: weekdays must be a list")
        for d in days:
            if not _weekday(d):
                raise ValueError(f"window {name}: weekday 0 (Monday) to 6 (Sunday) expected, not {d!r}")
        temperature = _number(w.get("temperature"))
        if temperature is None:
            raise ValueError(f"window {name}: temperature is not a number")
        windows.append(Window(tuple(int(d) for d in days), _minute(w.get("from"), name),
                              _minute(w.get("to"), name), temperature))
    return Plan(tuple(windows), low)


@dataclasses.dataclass(frozen=True)
class Plan:
    windows: typing.Tuple[Window, ...]
    setback: float

    def target(self, at: datetime.datetime) -> float:
        """The highest temperature of the windows covering `at`, else the setback; a window with from > to runs past midnight."""
        local = at.astimezone(TIMEZONE)
        day = local.weekday()
        prev = (day + 6) % 7
        m = local.hour * 60 + local.minute
        best = None
        for w in self.windows:
            if w.start == w.end:
                on = day in w.days
            elif w.start < w.end:
                on = day in w.days and w.start <= m < w.end
            else:
                on = (day in w.days and m >= w.start) or (prev in w.days and m < w.end)
            if on and (best is None or w.temperature > best):
                best = w.temperature
        return self.setback if best is None else best

    def _candidates(self, start: datetime.datetime, end: datetime.datetime) -> typing.List[datetime.datetime]:
        # The plan changes only at a window boundary or a midnight in local time (the weekday changes there, which
        # ends a whole-day window), or at a change of the UTC offset. A local time in the spring gap or the
        # repeated autumn hour is tried with both folds; boundaries() keeps only the instants where the target changes.
        found = set()
        first = start.astimezone(TIMEZONE).date() - datetime.timedelta(days=1)
        last = end.astimezone(TIMEZONE).date() + datetime.timedelta(days=1)
        minutes = {0} | {b for w in self.windows for b in (w.start, w.end)}
        day = first
        while day <= last:
            midnight = datetime.datetime(day.year, day.month, day.day)
            for b in minutes:
                naive = midnight + datetime.timedelta(minutes=b)
                for fold in (0, 1):
                    found.add(naive.replace(tzinfo=TIMEZONE, fold=fold).astimezone(UTC))
            day += datetime.timedelta(days=1)
        hour = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        while hour <= end:
            if hour.astimezone(TIMEZONE).utcoffset() != (hour - _STEP).astimezone(TIMEZONE).utcoffset():
                found.add(hour)
            hour += datetime.timedelta(hours=1)
        return sorted(t for t in found if start < t <= end)

    def boundaries(self, now: datetime.datetime, horizon: datetime.timedelta) -> typing.List[Boundary]:
        """Every change of the plan temperature in (now, now + horizon]."""
        result = []
        for t in self._candidates(now, now + horizon):
            before, after = self.target(t - _STEP), self.target(t)
            if before != after:
                result.append(Boundary(t, before, after))
        return result
