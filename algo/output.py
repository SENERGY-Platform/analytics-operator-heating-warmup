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

__all__ = ("round_to", "whole_minutes", "iso_utc", "Throttle")

import datetime
import decimal
import typing


def round_to(value: typing.Optional[float], step: str) -> typing.Optional[float]:
    """Half away from zero on the shortest decimal form, so 0.25 K gives 0.3 and -0.25 K gives -0.3, unlike round()."""
    if value is None:
        return None
    return float(decimal.Decimal(repr(float(value))).quantize(decimal.Decimal(step), rounding=decimal.ROUND_HALF_UP))


def whole_minutes(hours: float) -> int:
    return int(round_to(hours * 60, "1"))


def iso_utc(at: datetime.datetime) -> str:
    return at.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Throttle:
    """Passes a payload when it differs from the last one passed and that was at least `interval` ago."""

    def __init__(self, interval: datetime.timedelta):
        self.interval = interval
        self._last = None
        self._at: typing.Optional[datetime.datetime] = None

    def __call__(self, payload: dict, now: datetime.datetime) -> typing.Optional[dict]:
        if self._last is not None and payload == self._last:
            return None
        if self._at is not None and now - self._at < self.interval:
            return None
        self._last = dict(payload)
        self._at = now
        return dict(payload)
