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

__all__ = ("ROOM", "SETPOINT", "OUTDOOR", "TRAINING_COLUMNS", "Grid", "Fit", "build_grid", "find_episodes",
           "fit_episodes", "warmup_error", "zone_frames", "fit_zone", "fit_zones", "train_zone_models",
           "valid_temperature", "to_utc_ns")

import dataclasses
import typing

import numpy as np
import pandas as pd
import ray

from .heating import HeatingCurve, OffWindow
from .model import DEAD_MINUTES, MAX_WARMUP_HOURS, WarmupModel

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
# Below this outdoor spread over the fitted steps a1 is not determined and stays 0.
MIN_OUTDOOR_SPAN_K = 3.0
# Warm-up errors closer than this count as equal, so float noise does not pick a longer dead time.
ERROR_TIE_H = 1e-9
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


def find_episodes(grid: Grid, off: typing.Optional[np.ndarray] = None) -> typing.List[typing.Tuple[int, int]]:
    """
    Warm-ups as (first, last) grid index: from a rise of the setpoint to at least RISE_K above the room until the
    room reaches the setpoint minus REACHED_K, 6 h pass, the setpoint drops or a value is missing. `off[i]` marks the
    step from i to i + 1 as unknown: a running episode ends at i, so one starting at i has no step and is dropped.
    """
    T, S, O = grid.room, grid.setpoint, grid.outdoor
    known = np.isfinite(T) & np.isfinite(S) & np.isfinite(O)
    n = len(T)
    usable = np.ones(n, dtype=bool) if off is None else ~np.asarray(off, dtype=bool)
    episodes = []
    k = 1
    while k < n:
        if not (known[k] and np.isfinite(S[k - 1]) and S[k] > S[k - 1] and S[k] - T[k] >= RISE_K - EPS):
            k += 1
            continue
        last = k
        j = k + 1
        while j < n and known[j] and usable[j - 1] and S[j] >= S[j - 1]:
            # A drop of the setpoint, a gap or a switch-off ends the episode before j: the step to j was not heated throughout.
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
    a0: float
    a1: float
    b: float
    ta_min: typing.Optional[float]
    ta_max: typing.Optional[float]
    residual: float
    steps: int
    # The coefficients of the heating curve form, None without one.
    k: typing.Optional[float] = None
    c: typing.Optional[float] = None


def _lstsq(columns: typing.List[np.ndarray], y: np.ndarray) -> typing.Optional[np.ndarray]:
    A = np.column_stack(columns)
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    return None if rank < A.shape[1] else coef


def fit_episodes(grid: Grid, episodes: typing.List[typing.Tuple[int, int]], dead_steps: int = 0,
                 curve: typing.Optional[HeatingCurve] = None) -> typing.Tuple[typing.Optional[Fit], str]:
    """
    Least squares of dT/dt = a0 + a1 T_out - b (T - T_out) over the steps of all episodes from `dead_steps` after
    their start, with a heating curve as k (flow - T) - c (T - T_out). The fit, or None and the reason.
    """
    index = np.array([i for first, last in episodes for i in range(first + dead_steps, last)], dtype=np.int64)
    if len(index) == 0:
        return None, "no steps after the dead time"
    T, O = grid.room, grid.outdoor
    y = (T[index + 1] - T[index]) / STEP_H
    # The mean slope over a step belongs to the mean temperatures over it.
    room = (T[index] + T[index + 1]) / 2
    outdoor = (O[index] + O[index + 1]) / 2
    x = room - outdoor
    ones = np.ones_like(y)
    if curve is not None:
        heat = curve.flow(outdoor) - room
        coef = _lstsq([heat, -x], y)
        if coef is None:
            return None, "k and c not determined"
        k, c = float(coef[0]), float(coef[1])
        if c < 0:
            coef = _lstsq([heat], y)
            if coef is None:
                return None, "k not determined"
            k, c = float(coef[0]), 0.0
        if k <= 0:
            return None, f"rejected, k={k:.4f} 1/h"
        a0, a1, b = k * curve.room * (1 + curve.slope), -k * (1 + curve.slope), k + c
        ta_min = ta_max = None
    else:
        k = c = None
        ta_min, ta_max = float(outdoor.min()), float(outdoor.max())
        with_a1 = ta_max - ta_min >= MIN_OUTDOOR_SPAN_K - EPS
        heat = [ones, outdoor] if with_a1 else [ones]
        coef = _lstsq(heat + [-x], y)
        if coef is None:
            return None, "a and b not determined (no spread in T - T_out)"
        b = float(coef[-1])
        if b < 0:
            # With one bound the constrained optimum lies on it.
            coef = _lstsq(heat, y)
            if coef is None:
                return None, "a not determined"
            b = 0.0
        a0, a1 = float(coef[0]), (float(coef[1]) if with_a1 else 0.0)
        low = min((a0 + a1 * ta_min, ta_min), (a0 + a1 * ta_max, ta_max))
        if low[0] <= 0:
            return None, f"rejected, rate {low[0]:.4f} K/h at {low[1]:.1f} °C"
    residual = float(np.sqrt(np.mean((y - (a0 + a1 * outdoor - b * x)) ** 2)))
    return Fit(a0, a1, b, ta_min, ta_max, residual, len(y), k, c), ""


def warmup_error(grid: Grid, episodes: typing.List[typing.Tuple[int, int]], model: WarmupModel) -> float:
    """Mean absolute error of the predicted against the actual episode duration in hours, unreachable counting as the maximum."""
    errors = []
    for first, last in episodes:
        outdoor = float(np.mean(grid.outdoor[first:last + 1]))
        predicted = model.hours(float(grid.room[first]), float(grid.room[last]), outdoor)
        errors.append(abs((MAX_WARMUP_HOURS if predicted is None else predicted) - (last - first) * STEP_H))
    return float(np.mean(errors))


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


def fit_zone(frames: typing.List[pd.DataFrame], now_ns: int, curve: typing.Optional[HeatingCurve] = None,
             off: typing.Optional[OffWindow] = None) -> typing.Tuple[typing.Optional[WarmupModel], int, str]:
    """
    The model of one zone from its frames, or None, with the number of usable episodes and a log line. Each dead time
    gets its own fit, and the one that predicts the episode durations best wins, the shorter on a tie.
    """
    grid = build_grid(frames)
    if grid is None:
        return None, 0, "no model: the history lacks room temperature, setpoint or outdoor temperature"
    episodes = find_episodes(grid, None if off is None else off.steps(grid.times, STEP_NS))
    n = len(episodes)
    if n < MIN_EPISODES:
        return None, n, f"no model: {n} usable warm-up episodes, at least {MIN_EPISODES} needed"
    best, reasons = None, []
    for dead in DEAD_MINUTES:
        fit, reason = fit_episodes(grid, episodes, dead * 60 * 10**9 // STEP_NS, curve)
        if fit is None:
            reasons.append(f"{dead} min: {reason}")
            continue
        model = WarmupModel(fit.a0, fit.b, n, fit.residual, trained_until=now_ns, a1=fit.a1, dead_minutes=dead,
                            ta_min=fit.ta_min, ta_max=fit.ta_max)
        error = warmup_error(grid, episodes, model)
        if best is None or error < best[1] - ERROR_TIE_H:
            best = (model, error, fit)
    if best is None:
        return None, n, f"no model: no dead time gives a fit from {n} episodes ({'; '.join(reasons)})"
    model, error, fit = best
    return model, n, (f"model: a0={model.a0:.4f} K/h, a1={model.a1:.4f} K/h/K, b={model.b:.4f} 1/h, "
                      f"dead time {model.dead_minutes} min, {n} episodes, {fit.steps} steps, "
                      f"warm-up error {error * 60:.1f} min, residual {model.residual:.4f} K/h")


def fit_zones(datasets, consume: typing.Optional[typing.List[bool]], devices: typing.Optional[typing.List[typing.Optional[str]]],
              zones: typing.Dict[str, typing.Sequence[str]], now_ns: int, curve: typing.Optional[HeatingCurve] = None,
              off: typing.Optional[OffWindow] = None):
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
        model, episodes, note = fit_zone(zone_frames(frames, zone_devices), now_ns, curve, off)
        notes.append(f"zone {key}: {note}")
        results[key] = (model, episodes)
    return results, notes, False


train_zone_models = ray.remote(fit_zones)
