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

__all__ = ("ROOM", "SETPOINT", "OUTDOOR", "TRAINING_COLUMNS", "Grid", "build_grid", "find_episodes", "fit_episodes",
           "zone_frames", "fit_zone", "fit_zones", "train_zone_models", "valid_temperature", "to_utc_ns")

import dataclasses
import typing

import numpy as np
import pandas as pd
import ray

from .model import WarmupModel

ROOM = "room_temperature"
SETPOINT = "target_temperature"
OUTDOOR = "outdoor_temperature"
TRAINING_COLUMNS = (ROOM, SETPOINT, OUTDOOR)

STEP_NS = 10 * 60 * 10**9
STEP_H = 1 / 6
# Samples further apart are not interpolated across; a setpoint older than its limit is unknown.
ROOM_MAX_GAP_NS = 2 * 3600 * 10**9
OUTDOOR_MAX_GAP_NS = 3 * 3600 * 10**9
SETPOINT_MAX_AGE_NS = 24 * 3600 * 10**9

RISE_K = 1.0
REACHED_K = 0.3
MAX_EPISODE_STEPS = 36  # 6 h
MIN_EPISODE_STEPS = 2
MIN_EPISODES = 3
# Temperatures in tenths of a kelvin do not subtract exactly: 16.4 - 15.4 < 1.0 in floating point.
EPS = 1e-9


def to_utc_ns(value) -> int:
    """Nanoseconds since the epoch; a time without zone counts as UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value)


def valid_temperature(value) -> typing.Optional[float]:
    """A finite value above absolute zero; DWD marks a missing value with -999."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= -273.15:
        return None
    return number


@dataclasses.dataclass
class Grid:
    times: np.ndarray  # int64 ns, STEP_NS apart
    room: np.ndarray
    setpoint: np.ndarray
    outdoor: np.ndarray


def _device_series(frame: pd.DataFrame, column: str) -> typing.List[typing.Tuple[np.ndarray, np.ndarray]]:
    """Per device, the sorted sample times and values of one column; equal times are averaged."""
    parts = [frame] if "device_id" not in frame.columns else [p for _, p in frame.groupby("device_id", sort=False, dropna=False)]
    result = []
    for part in parts:
        values = pd.to_numeric(part[column], errors="coerce")
        values = values.where(values > -273.15)
        times = pd.DatetimeIndex(pd.to_datetime(part["time"], utc=True)).asi8
        series = pd.Series(values.to_numpy(dtype=float), index=times).dropna()
        if series.empty:
            continue
        series = series.groupby(level=0).mean().sort_index()
        result.append((series.index.to_numpy(dtype=np.int64), series.to_numpy(dtype=float)))
    return result


def _interpolate(times: np.ndarray, values: np.ndarray, grid: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.interp(grid, times, values)
    after = np.searchsorted(times, grid, side="left")
    exact = (after < len(times)) & (times[np.minimum(after, len(times) - 1)] == grid)
    inside = (after > 0) & (after < len(times))
    gap = np.full(len(grid), np.iinfo(np.int64).max)
    gap[inside] = times[after[inside]] - times[after[inside] - 1]
    out[~(exact | (inside & (gap <= max_gap)))] = np.nan
    return out


def _last_value(times: np.ndarray, values: np.ndarray, grid: np.ndarray, max_age: int) -> np.ndarray:
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    known = idx >= 0
    out[known] = values[idx[known]]
    out[known & (grid - times[np.maximum(idx, 0)] > max_age)] = np.nan
    return out


def _mean(arrays: typing.List[np.ndarray], n: int) -> np.ndarray:
    if not arrays:
        return np.full(n, np.nan)
    stacked = np.vstack(arrays)
    count = np.sum(np.isfinite(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    out = np.full(n, np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def build_grid(frames: typing.List[pd.DataFrame]) -> typing.Optional[Grid]:
    """Room, setpoint and outdoor temperature on a 10-minute grid, each the mean over the devices that have a value."""
    series = {c: [] for c in TRAINING_COLUMNS}
    for frame in frames:
        if "time" not in frame.columns:
            continue
        for column in TRAINING_COLUMNS:
            if column in frame.columns:
                series[column] += _device_series(frame, column)
    if not series[ROOM] or not series[SETPOINT] or not series[OUTDOOR]:
        return None
    first = min(int(t[0]) for s in series.values() for t, _ in s)
    last = max(int(t[-1]) for s in series.values() for t, _ in s)
    start = first - first % STEP_NS
    grid = np.arange(start, last + 1, STEP_NS, dtype=np.int64)
    n = len(grid)
    return Grid(
        times=grid,
        room=_mean([_interpolate(t, v, grid, ROOM_MAX_GAP_NS) for t, v in series[ROOM]], n),
        setpoint=_mean([_last_value(t, v, grid, SETPOINT_MAX_AGE_NS) for t, v in series[SETPOINT]], n),
        outdoor=_mean([_interpolate(t, v, grid, OUTDOOR_MAX_GAP_NS) for t, v in series[OUTDOOR]], n),
    )


def find_episodes(grid: Grid) -> typing.List[typing.Tuple[int, int]]:
    """
    Warm-ups as (first, last) grid index: from a rise of the setpoint to at least RISE_K above the room until the
    room reaches the setpoint minus REACHED_K, 6 h pass, the setpoint drops or a value is missing.
    """
    T, S, O = grid.room, grid.setpoint, grid.outdoor
    known = np.isfinite(T) & np.isfinite(S) & np.isfinite(O)
    n = len(T)
    episodes = []
    k = 1
    while k < n:
        if not (known[k] and np.isfinite(S[k - 1]) and S[k] > S[k - 1] and S[k] - T[k] >= RISE_K - EPS):
            k += 1
            continue
        last = k
        j = k + 1
        while j < n and known[j] and S[j] >= S[j - 1]:
            # A drop of the setpoint or a gap ends the episode before j: the step to j was not heated throughout.
            last = j
            if T[j] >= S[j] - REACHED_K - EPS or j - k >= MAX_EPISODE_STEPS:
                break
            j += 1
        if last - k >= MIN_EPISODE_STEPS:
            episodes.append((k, last))
        k = last + 1 if last > k else k + 1
    return episodes


@dataclasses.dataclass
class Fit:
    a: float
    b: float
    residual: float
    steps: int


def fit_episodes(grid: Grid, episodes: typing.List[typing.Tuple[int, int]]) -> typing.Optional[Fit]:
    """Least squares of dT/dt = a - b (T - T_out) over the steps of all episodes; None if a and b are not determined."""
    slopes, deltas = [], []
    for first, last in episodes:
        T = grid.room[first:last + 1]
        O = grid.outdoor[first:last + 1]
        slopes.append(np.diff(T) / STEP_H)
        # The mean slope over a step belongs to the mean temperature difference over it.
        deltas.append((T[:-1] + T[1:]) / 2 - (O[:-1] + O[1:]) / 2)
    if not slopes:
        return None
    y = np.concatenate(slopes)
    x = np.concatenate(deltas)
    A = np.column_stack((np.ones_like(x), -x))
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    if rank < 2:
        return None
    a, b = float(coef[0]), float(coef[1])
    residual = float(np.sqrt(np.mean((y - (a - b * x)) ** 2)))
    return Fit(a, b, residual, len(y))


def _read_frames(datasets, consume, devices, notes):
    """The frames of the datasets to consume, each with its devices; with whether a requested history could not be read."""
    frames = []
    for i, ds in enumerate(datasets):
        if consume is not None and (i >= len(consume) or not consume[i]):
            continue
        try:
            frame = (ray.get(ds) if isinstance(ds, ray.ObjectRef) else ds).to_pandas()
        except Exception as ex:
            notes.append(f"history of input {i} unreadable: {type(ex).__name__}: {ex}")
            return None, True
        # An input with one device comes without the column device_id.
        if "device_id" not in frame.columns and devices is not None and i < len(devices) and devices[i] is not None:
            frame = frame.assign(device_id=devices[i])
        frames.append(frame)
    return frames, False


def zone_frames(frames: typing.List[pd.DataFrame], devices: typing.Iterable[str]) -> typing.List[pd.DataFrame]:
    """The room and setpoint rows of the zone's devices, and every outdoor row; rows of no known device are left out."""
    wanted = set(devices)
    result = []
    for frame in frames:
        if "time" not in frame.columns:
            continue
        own = [c for c in (ROOM, SETPOINT) if c in frame.columns]
        if own and "device_id" in frame.columns:
            part = frame.loc[frame["device_id"].astype(str).str.strip().isin(wanted), ["time", "device_id", *own]]
            if not part.empty:
                result.append(part)
        if OUTDOOR in frame.columns:
            result.append(frame[["time", OUTDOOR] + (["device_id"] if "device_id" in frame.columns else [])])
    return result


def fit_zone(frames: typing.List[pd.DataFrame], now_ns: int) -> typing.Tuple[typing.Optional[WarmupModel], int, str]:
    """The model of one zone from its frames, or None, with the number of usable episodes and a log line."""
    grid = build_grid(frames)
    if grid is None:
        return None, 0, "no model: the history lacks room temperature, setpoint or outdoor temperature"
    episodes = find_episodes(grid)
    if len(episodes) < MIN_EPISODES:
        return None, len(episodes), f"no model: {len(episodes)} usable warm-up episodes, at least {MIN_EPISODES} needed"
    fit = fit_episodes(grid, episodes)
    if fit is None:
        return None, len(episodes), f"no model: {len(episodes)} episodes do not determine a and b (no spread in T - T_out)"
    if fit.a <= 0 or fit.b < 0:
        return None, len(episodes), (f"no model: fit rejected, a={fit.a:.4f} K/h, b={fit.b:.4f} 1/h "
                                     f"from {len(episodes)} episodes")
    return (WarmupModel(fit.a, fit.b, len(episodes), fit.residual, trained_until=now_ns), len(episodes),
            f"model: a={fit.a:.4f} K/h, b={fit.b:.4f} 1/h from {len(episodes)} episodes, {fit.steps} steps, "
            f"residual {fit.residual:.4f} K/h")


def fit_zones(datasets, consume: typing.Optional[typing.List[bool]], devices: typing.Optional[typing.List[typing.Optional[str]]],
              zones: typing.Dict[str, typing.Sequence[str]], now_ns: int):
    """
    Per zone key the model (None if its history gives none) and its number of usable episodes, the log lines, and
    whether the requested history could not be read; runs as one Ray task for all zones.
    """
    notes = []
    frames, failed = _read_frames(datasets, consume, devices, notes)
    if failed:
        return {}, notes, True
    results = {}
    for key, zone_devices in zones.items():
        model, episodes, note = fit_zone(zone_frames(frames, zone_devices), now_ns)
        notes.append(f"zone {key}: {note}")
        results[key] = (model, episodes)
    return results, notes, False


train_zone_models = ray.remote(fit_zones)
