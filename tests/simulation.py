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
import random
import typing

import pandas as pd

START = pd.Timestamp("2026-01-05 00:00:00")  # naive UTC, as the readers return it


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()


class UnreadableDataset:
    """A history whose read fails, as when Ray finds no CPU for it."""

    def to_pandas(self):
        raise RuntimeError("Failed to get allocated resources for data-dataset_407_0 after 10 consecutive failures.")


def simulate(a: float, b: float, days: int, outdoor: typing.Callable[[int], float], comfort=21.0, setback=17.0,
             on_minute=6 * 60, off_minute=22 * 60, report_minutes=5, decimals: typing.Optional[int] = None,
             seed=1, device="d1"):
    """
    A zone heated by a thermostat that holds its setpoint: dT/dt = a - b (T - T_out) while it is below, -b (T - T_out)
    otherwise, integrated exactly minute by minute. The setpoint is `comfort` from on_minute to off_minute (UTC), else
    `setback`. Returns the thermostat frame (room and setpoint every `report_minutes`, as the history of an input with
    several devices, with device_id) and the outdoor frame (every 10 minutes); `outdoor(day)` is the outdoor
    temperature of a day.
    """
    rng = random.Random(seed)
    T = setback
    rows, outdoor_rows = [], []
    for minute in range(days * 1440):
        day, of_day = divmod(minute, 1440)
        out = outdoor(day)
        setpoint = comfort if on_minute <= of_day < off_minute else setback
        at = START + pd.Timedelta(minutes=minute)
        if minute % report_minutes == 0:
            value = T if decimals is None else round(T + rng.uniform(-0.05, 0.05), decimals)
            rows.append((at, value, setpoint))
        if minute % 10 == 0:
            outdoor_rows.append((at, out))
        dt = 1 / 60
        if T < setpoint:
            t_inf = out + a / b
            T_next = t_inf + (T - t_inf) * math.exp(-b * dt)
            T = min(T_next, setpoint)
        else:
            # Cooling towards outdoor until the thermostat holds the setpoint.
            T = max(out + (T - out) * math.exp(-b * dt), setpoint)
    thermostat = pd.DataFrame(rows, columns=["time", "room_temperature", "target_temperature"]).assign(device_id=device)
    weather = pd.DataFrame(outdoor_rows, columns=["time", "outdoor_temperature"])
    return thermostat, weather
