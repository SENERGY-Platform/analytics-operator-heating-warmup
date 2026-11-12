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

__all__ = ("Zone", "parse_zones")

import dataclasses
import json
import typing

from .schedule import Plan, parse_plan


@dataclasses.dataclass(frozen=True)
class Zone:
    key: str
    name: str
    devices: typing.Tuple[str, ...]
    plan: typing.Optional[Plan]


def parse_zones(text) -> typing.List[Zone]:
    """The zones of the config value `zones`, a JSON list; invalid input raises ValueError."""
    if not str(text or "").strip():
        raise ValueError("zones is empty")
    try:
        raw = json.loads(text)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"zones is no JSON: {ex}") from None
    if not isinstance(raw, list) or not raw:
        raise ValueError("zones must be a non-empty list")
    zones, owner = [], {}
    for i, z in enumerate(raw):
        if not isinstance(z, dict):
            raise ValueError(f"zone {i + 1}: object expected")
        key = z.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"zone {i + 1}: key must be a non-empty string")
        key = key.strip()
        if any(other.key == key for other in zones):
            raise ValueError(f"zone {key}: key used twice")
        devices = z.get("devices") or []
        if not isinstance(devices, list) or not all(isinstance(d, str) and d.strip() for d in devices):
            raise ValueError(f"zone {key}: devices must be a list of device ids")
        devices = tuple(dict.fromkeys(d.strip() for d in devices))
        for d in devices:
            # A thermostat in two zones would be averaged into both.
            if d in owner:
                raise ValueError(f"device {d} is in zone {owner[d]} and zone {key}")
            owner[d] = key
        try:
            plan = parse_plan(z.get("schedule"), z.get("setback"))
        except ValueError as ex:
            raise ValueError(f"zone {key}: {ex}") from None
        name = z.get("name")
        zones.append(Zone(key, name.strip() if isinstance(name, str) else key, devices, plan))
    return zones
