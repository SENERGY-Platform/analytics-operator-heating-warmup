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
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from algo.heating import HeatingCurve, parse_off_window
from algo.model import WarmupModel, warmup_hours
from algo.training import STEP_H, STEP_NS, Grid, build_grid, find_episodes, fit_episodes, fit_zone, fit_zones
from tests.simulation import START, FakeDataset, UnreadableDataset, simulate, simulate_curve

A, B = 4.0, 0.15
NOW = int(pd.Timestamp("2026-03-01", tz="UTC").value)


def outdoor_by_day(day: int) -> float:
    return [-4.0, 8.0, 2.0, 11.0, -1.0, 5.0, 0.0][day % 7]


def fit(datasets, consume=None, curve=None, off=None):
    """One zone of every device: (model, notes, history failed, episodes) as fit_zones gives it for that zone."""
    results, notes, failed = fit_zones(datasets, consume, None, {"z": ["d1", "a", "b"]}, NOW, curve, off)
    if failed:
        return None, notes, True, 0
    model, episodes = results["z"]
    return model, notes, False, episodes


def grid_of(room, setpoint, outdoor) -> Grid:
    n = len(room)
    return Grid(np.arange(n, dtype=np.int64) * STEP_NS, np.array(room, float), np.array(setpoint, float),
                np.array(outdoor, float))


def exact_episodes(episodes, a1=0.0) -> Grid:
    """
    A grid of warm-ups, each (a0, b, outdoor, start, steps) on the exact curve of dT/dt = a0 + a1 T_out - b (T - T_out),
    with the setpoint 4 K above the start and one step of setback between them.
    """
    room, setpoint, outdoor = [], [], []
    for a0, b, out, start, steps in episodes:
        room.append(start)
        setpoint.append(start)
        outdoor.append(out)
        rate = a0 + a1 * out
        for i in range(steps + 1):
            t = i * STEP_H
            room.append(start + rate * t if b == 0 else out + rate / b + (start - out - rate / b) * math.exp(-b * t))
            setpoint.append(start + 4)
            outdoor.append(out)
    return grid_of(room, setpoint, outdoor)


class TestFit(unittest.TestCase):
    def assert_recovered(self, model):
        self.assertIsInstance(model, WarmupModel)
        self.assertLess(abs(model.a0 - A) / A, 0.10, model.a0)
        # A heating power that does not depend on the outdoor temperature, across the range trained on.
        for outdoor in (model.ta_min, model.ta_max):
            self.assertLess(abs(model.rate(outdoor) - A) / A, 0.10, (outdoor, model.a1))
        self.assertLess(abs(model.b - B) / B, 0.10, model.b)
        self.assertEqual(model.dead_minutes, 0)

    def test_exact_curves_give_back_a_and_b(self):
        thermostat, weather = simulate(A, B, days=14, outdoor=outdoor_by_day)
        model, notes, failed, episodes = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assert_recovered(model)
        self.assertFalse(failed)
        self.assertEqual((episodes, model.episodes), (14, 14))
        self.assertEqual(model.trained_until, NOW)
        self.assertLess(model.residual, 0.05)
        # Pairing each step's slope with its mean temperature difference, not the one at its start, is what
        # makes exact curves come back this closely (about 1 % off otherwise).
        self.assertLess(abs(model.a0 - A) / A, 0.002)
        self.assertLess(abs(model.a1), 0.0005)
        self.assertLess(abs(model.b - B) / B, 0.002)
        self.assertEqual((model.ta_min, model.ta_max), (-4.0, 11.0))

    def test_rounded_noisy_readings_every_ten_minutes(self):
        thermostat, weather = simulate(A, B, days=21, outdoor=outdoor_by_day, report_minutes=10, decimals=1)
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assert_recovered(model)

    def test_another_zone(self):
        thermostat, weather = simulate(2.5, 0.08, days=14, outdoor=lambda d: [3.0, -6.0, 9.0][d % 3])
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertLess(abs(model.a0 - 2.5) / 2.5, 0.10)
        self.assertLess(abs(model.b - 0.08) / 0.08, 0.10)

    def test_several_thermostats_are_averaged(self):
        thermostat, weather = simulate(A, B, days=10, outdoor=outdoor_by_day)
        # A second device reporting the same zone half a degree warmer at other moments, as one frame with device_id.
        other = thermostat.assign(time=thermostat["time"] + pd.Timedelta(minutes=2),
                                  room_temperature=thermostat["room_temperature"] + 0.5)
        both = pd.concat([thermostat.assign(device_id="a"), other.assign(device_id="b")], ignore_index=True)
        model, _, _, episodes = fit([FakeDataset(both), FakeDataset(weather)])
        self.assertEqual(episodes, 10)
        # The first step of each warm-up averages in a device that has not seen the rise yet. With a1 free that
        # tilts b (13 % off here), so the check is on what the model is for, the warm-up time.
        for outdoor in (0.0, 5.0, 11.0):
            self.assertLess(abs(model.hours(17.25, 21.0, outdoor) / warmup_hours(A, B, 17.25, 21.0, outdoor) - 1), 0.03)
        for outdoor in (model.ta_min, model.ta_max):
            self.assertLess(abs(model.rate(outdoor) - A) / A, 0.10)

    def test_missing_outdoor_values_of_dwd_are_ignored(self):
        thermostat, weather = simulate(A, B, days=10, outdoor=outdoor_by_day)
        weather.loc[weather.index % 7 == 3, "outdoor_temperature"] = -999.0
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assert_recovered(model)

    def test_aware_and_naive_times_mix(self):
        thermostat, weather = simulate(A, B, days=10, outdoor=outdoor_by_day)
        weather["time"] = weather["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assert_recovered(model)

    def test_fewer_than_three_episodes_give_no_model(self):
        thermostat, weather = simulate(A, B, days=2, outdoor=outdoor_by_day)
        model, notes, failed, episodes = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertIsNone(model)
        self.assertFalse(failed)
        self.assertEqual(episodes, 2)
        self.assertTrue(any("2 usable warm-up episodes" in n for n in notes), notes)

        thermostat, weather = simulate(A, B, days=3, outdoor=outdoor_by_day)
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertIsNotNone(model)

    def test_a_cooling_fit_is_rejected(self):
        # The room falls while the setpoint is up: a <= 0.
        thermostat, weather = simulate(A, B, days=5, outdoor=outdoor_by_day)
        thermostat["room_temperature"] = 30 - thermostat.index * 0.001
        thermostat["target_temperature"] = np.where((thermostat["time"].dt.hour >= 6), 35.0, 17.0)
        model, notes, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertIsNone(model)
        self.assertTrue(any("rejected" in n for n in notes), notes)

    def test_without_outdoor_history_there_is_no_model(self):
        thermostat, _ = simulate(A, B, days=5, outdoor=outdoor_by_day)
        model, notes, failed, _ = fit([FakeDataset(thermostat)])
        self.assertIsNone(model)
        self.assertFalse(failed)
        self.assertTrue(any("lacks" in n for n in notes), notes)

    def test_an_unreadable_history_is_reported(self):
        thermostat, _ = simulate(A, B, days=1, outdoor=outdoor_by_day)
        model, _, failed, _ = fit([FakeDataset(thermostat), UnreadableDataset()])
        self.assertIsNone(model)
        self.assertTrue(failed)

    def test_inputs_not_to_consume_are_not_read(self):
        thermostat, weather = simulate(A, B, days=5, outdoor=outdoor_by_day)
        model, _, failed, _ = fit([FakeDataset(thermostat), UnreadableDataset(), FakeDataset(weather)],
                                  consume=[True, False, True])
        self.assertFalse(failed)
        self.assertIsNotNone(model)


class TestDeadTimeAndHeatingPower(unittest.TestCase):
    CURVE = HeatingCurve(0.8, 22.0)

    def test_the_heating_curve_gives_back_k_c_and_the_dead_time(self):
        k, c = 0.12, 0.06
        thermostat, weather = simulate_curve(k, c, 0.8, 22.0, days=14, outdoor=outdoor_by_day, dead_minutes=30)
        model, notes, _, episodes = fit([FakeDataset(thermostat), FakeDataset(weather)], curve=self.CURVE)
        self.assertEqual((model.dead_minutes, episodes), (30, 14))
        # a1 = -k (1 + s), b = k + c
        self.assertAlmostEqual(-model.a1 / 1.8, k, delta=k * 0.01)
        self.assertAlmostEqual(model.b + model.a1 / 1.8, c, delta=c * 0.01)
        self.assertAlmostEqual(model.a0, k * 22.0 * 1.8, delta=0.01)
        self.assertEqual((model.ta_min, model.ta_max), (None, None))
        self.assertTrue(any("dead time 30 min" in n and "a0=" in n and "warm-up error" in n for n in notes), notes)

        grid = build_grid([thermostat, weather])
        found, _ = fit_episodes(grid, find_episodes(grid), 3, self.CURVE)
        self.assertAlmostEqual(found.k, k, delta=k * 0.01)
        self.assertAlmostEqual(found.c, c, delta=c * 0.01)

    def test_without_curve_the_heating_power_follows_the_outdoor_temperature(self):
        thermostat, weather = simulate(3.0, 0.12, days=14, outdoor=outdoor_by_day, a1=-0.1, dead_minutes=40)
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertEqual(model.dead_minutes, 40)
        self.assertAlmostEqual(model.a0, 3.0, delta=0.01)
        self.assertAlmostEqual(model.a1, -0.1, delta=0.001)
        self.assertAlmostEqual(model.b, 0.12, delta=0.001)
        self.assertEqual((model.ta_min, model.ta_max), (-4.0, 11.0))

    def test_a_dead_time_too_short_or_too_long_loses(self):
        thermostat, weather = simulate(A, B, days=10, outdoor=outdoor_by_day, dead_minutes=20)
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertEqual(model.dead_minutes, 20)

    def test_on_a_tie_the_shorter_dead_time_wins(self):
        thermostat, weather = simulate(A, B, days=10, outdoor=outdoor_by_day, dead_minutes=20)
        with mock.patch("algo.training.warmup_error", return_value=0.25):
            model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertEqual(model.dead_minutes, 0)
        # Float noise below the tie tolerance does not make a longer one win either.
        errors = iter([0.25, 0.25 - 1e-12, 0.25 - 2e-12, 0.25, 0.25, 0.25, 0.25])
        with mock.patch("algo.training.warmup_error", side_effect=lambda *_: next(errors)):
            model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertEqual(model.dead_minutes, 0)

    def test_a_negative_b_is_fitted_again_with_b_zero(self):
        # Accelerating warm-ups: the slope grows with T - T_out, which reads as b < 0.
        g = grid_of(room=[17, 17, 17.1, 17.3, 17.6, 18.0, 18.5], setpoint=[17] + [21] * 6, outdoor=[0] * 7)
        found, _ = fit_episodes(g, [(1, 6)])
        self.assertEqual((found.b, found.a1), (0.0, 0.0))
        self.assertAlmostEqual(found.a0, 1.8)  # mean slope of 0.6, 1.2, 1.8, 2.4, 3.0 K/h

    def test_a_negative_c_is_fitted_again_with_c_zero(self):
        # Exact curves of k (flow - T) - c (T - T_out) with k = 0.2, c = -0.05, as a0 + a1 T_out - b (T - T_out).
        k, c, s, r = 0.2, -0.05, 0.8, 22.0
        g = exact_episodes([(k * r * (1 + s), k + c, out, 17.0, 6) for out in (-2.0, 4.0, 9.0)], a1=-k * (1 + s))
        episodes = find_episodes(g)
        self.assertEqual(len(episodes), 3)
        found, _ = fit_episodes(g, episodes, 0, HeatingCurve(s, r))
        self.assertEqual(found.c, 0.0)
        # k alone by least squares, and the model it converts to.
        index = np.array([i for first, last in episodes for i in range(first, last)])
        y = (g.room[index + 1] - g.room[index]) / STEP_H
        room = (g.room[index] + g.room[index + 1]) / 2
        heat = r + s * (r - g.outdoor[index]) - room
        self.assertAlmostEqual(found.k, float(heat @ y / (heat @ heat)), places=12)
        self.assertEqual((found.a1, found.b), (-found.k * (1 + s), found.k))

    def test_a1_is_zero_below_three_kelvin_of_outdoor_spread(self):
        for days, a1 in (((2.0, 4.9), 0.0), ((2.0, 5.0), None)):
            with self.subTest(days=days):
                thermostat, weather = simulate(3.0, 0.12, days=10, outdoor=lambda d: days[d % 2], a1=-0.1)
                model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
                self.assertEqual((model.ta_min, model.ta_max), days)
                if a1 is not None:
                    self.assertEqual(model.a1, a1)
                else:
                    self.assertAlmostEqual(model.a1, -0.1, delta=0.001)

    def test_a_rate_at_or_below_zero_within_the_training_range_is_no_fit(self):
        # Exact warm-ups of a0 = 2, a1 = -0.25: the heating power is gone at 8 °C.
        g = exact_episodes([(2.0, 0.05, out, 17.0, 6) for out in (-2.0, 3.0, 7.0)], a1=-0.25)
        found, _ = fit_episodes(g, find_episodes(g))
        self.assertAlmostEqual(found.a1, -0.25, delta=1e-3)
        g = exact_episodes([(2.0, 0.05, out, 17.0, 6) for out in (-2.0, 3.0, 9.0)], a1=-0.25)
        found, reason = fit_episodes(g, find_episodes(g))
        self.assertIsNone(found)
        self.assertIn("rejected, rate", reason)
        self.assertIn("9.0 °C", reason)

    def test_no_dead_time_with_a_fit_gives_no_model(self):
        thermostat, weather = simulate(A, B, days=5, outdoor=outdoor_by_day)
        # Room equal to outdoor: no spread in T - T_out for any dead time.
        weather = thermostat[["time", "room_temperature"]].rename(columns={"room_temperature": "outdoor_temperature"})
        model, notes, _, episodes = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertIsNone(model)
        self.assertEqual(episodes, 5)
        self.assertTrue(any("no dead time gives a fit from 5 episodes" in n and "60 min:" in n for n in notes), notes)


class TestSwitchOff(unittest.TestCase):
    def test_a_switch_off_over_every_warm_up_leaves_no_episode(self):
        # The heating comes on at 06:00 UTC, 07:00 in Berlin in January.
        thermostat, weather = simulate(A, B, days=5, outdoor=outdoor_by_day)
        model, _, _, episodes = fit([FakeDataset(thermostat), FakeDataset(weather)], off=parse_off_window("06:50-07:20"))
        self.assertEqual((model, episodes), (None, 0))
        model, _, _, episodes = fit([FakeDataset(thermostat), FakeDataset(weather)], off=parse_off_window("23:30-03:00"))
        self.assertEqual(episodes, 5)

    def test_a_switch_off_cuts_the_warm_ups(self):
        thermostat, weather = simulate(A, B, days=5, outdoor=outdoor_by_day)
        grid = build_grid([thermostat, weather])
        whole = find_episodes(grid)
        # 07:30 Berlin is 06:30 UTC, three steps after the start. The step from 06:20 touches it.
        cut = find_episodes(grid, parse_off_window("07:30-08:00").steps(grid.times, STEP_NS))
        self.assertEqual([(first, first + 2) for first, _ in whole], cut)


class TestZones(unittest.TestCase):
    """One training for all zones, each from the thermostats listed for it."""

    def test_two_zones_are_learned_separately(self):
        living, weather = simulate(A, B, days=10, outdoor=outdoor_by_day, device="t1")
        bath, _ = simulate(2.5, 0.08, days=10, outdoor=outdoor_by_day, device="t2")
        both = pd.concat([living, bath], ignore_index=True)
        results, notes, failed = fit_zones([FakeDataset(both), FakeDataset(weather)], None, None,
                                           {"zone:0": ["t1"], "zone:1": ["t2"]}, NOW)
        self.assertFalse(failed)
        (first, n0), (second, n1) = results["zone:0"], results["zone:1"]
        self.assertEqual((n0, n1), (10, 10))
        self.assertLess(abs(first.a0 - A) / A, 0.10)
        self.assertLess(abs(first.b - B) / B, 0.10)
        self.assertLess(abs(second.a0 - 2.5) / 2.5, 0.10)
        self.assertLess(abs(second.b - 0.08) / 0.08, 0.10)

    def test_a_device_in_no_zone_is_left_out(self):
        living, weather = simulate(A, B, days=10, outdoor=outdoor_by_day, device="t1")
        # A thermostat of another room, much warmer, that no zone lists.
        stray = living.assign(device_id="elsewhere", room_temperature=living["room_temperature"] + 3)
        results, _, _ = fit_zones([FakeDataset(pd.concat([living, stray], ignore_index=True)), FakeDataset(weather)],
                                  None, None, {"zone:0": ["t1"]}, NOW)
        model, _ = results["zone:0"]
        self.assertLess(abs(model.a0 - A) / A, 0.01)

    def test_too_few_episodes_concern_only_their_zone(self):
        living, weather = simulate(A, B, days=10, outdoor=outdoor_by_day, device="t1")
        bath, _ = simulate(A, B, days=2, outdoor=outdoor_by_day, device="t2")
        results, notes, _ = fit_zones([FakeDataset(pd.concat([living, bath], ignore_index=True)), FakeDataset(weather)],
                                      None, None, {"zone:0": ["t1"], "zone:1": ["t2"]}, NOW)
        self.assertIsNotNone(results["zone:0"][0])
        self.assertEqual(results["zone:1"], (None, 2))
        self.assertTrue(any(n.startswith("zone zone:1: no model: 2 usable") for n in notes), notes)

    def test_an_input_with_one_device_takes_it_from_its_topic(self):
        living, weather = simulate(A, B, days=5, outdoor=outdoor_by_day)
        results, _, _ = fit_zones([FakeDataset(living.drop(columns="device_id")), FakeDataset(weather)], None,
                                  ["urn:t1", None], {"zone:0": ["urn:t1"]}, NOW)
        self.assertIsNotNone(results["zone:0"][0])
        results, _, _ = fit_zones([FakeDataset(living.drop(columns="device_id")), FakeDataset(weather)], None,
                                  None, {"zone:0": ["urn:t1"]}, NOW)
        self.assertEqual(results["zone:0"], (None, 0))


class TestEpisodes(unittest.TestCase):
    def test_a_rise_of_exactly_one_kelvin_in_tenths_starts_an_episode(self):
        # 16.4 - 15.4 is 0.9999999999999982 in floating point.
        g = grid_of(room=[15.4, 15.4, 15.8, 16.2], setpoint=[15.0, 16.4, 16.4, 16.4], outdoor=[0, 0, 0, 0])
        self.assertEqual(find_episodes(g), [(1, 3)])

    def test_a_rise_of_less_than_one_kelvin_is_no_episode(self):
        g = grid_of(room=[15.5, 15.5, 15.8, 16.1], setpoint=[15.0, 16.4, 16.4, 16.4], outdoor=[0, 0, 0, 0])
        self.assertEqual(find_episodes(g), [])

    def test_the_episode_ends_when_the_room_is_close_to_the_setpoint(self):
        g = grid_of(room=[17, 17, 18, 19, 20, 20.7, 21, 21], setpoint=[17] + [21] * 7, outdoor=[0] * 8)
        self.assertEqual(find_episodes(g), [(1, 5)])

    def test_a_drop_of_the_setpoint_ends_the_episode_before_it(self):
        g = grid_of(room=[17, 17, 17.5, 18, 18.5, 18.6], setpoint=[17, 21, 21, 21, 17, 17], outdoor=[0] * 6)
        self.assertEqual(find_episodes(g), [(1, 3)])

    def test_a_gap_ends_the_episode_before_it(self):
        g = grid_of(room=[17, 17, 17.5, 18, np.nan, 19], setpoint=[17, 21, 21, 21, 21, 21], outdoor=[0] * 6)
        self.assertEqual(find_episodes(g), [(1, 3)])

    def test_six_hours_end_an_episode(self):
        n = 50
        g = grid_of(room=[10 + 0.01 * i for i in range(n)], setpoint=[10] + [21] * (n - 1), outdoor=[0] * n)
        self.assertEqual(find_episodes(g), [(1, 37)])

    def test_a_short_episode_is_not_usable(self):
        g = grid_of(room=[17, 17, 20.8, 21], setpoint=[17, 21, 21, 21], outdoor=[0] * 4)
        self.assertEqual(find_episodes(g), [])

    def test_a_further_rise_continues_the_episode(self):
        g = grid_of(room=[17, 17, 17.5, 18, 18.5, 19], setpoint=[17, 19, 19, 21, 21, 21], outdoor=[0] * 6)
        self.assertEqual(find_episodes(g), [(1, 5)])

    def test_fit_needs_spread_in_the_temperature_difference(self):
        g = grid_of(room=[17, 17, 17.5, 18], setpoint=[17, 21, 21, 21], outdoor=[17, 17, 17.5, 18])
        fit, reason = fit_episodes(g, [(1, 3)])
        self.assertIsNone(fit)
        self.assertIn("not determined", reason)

    def test_a_switch_off_ends_the_episode_before_it(self):
        g = grid_of(room=[17, 17, 17.5, 18, 18.5, 19, 19.5], setpoint=[17] + [21] * 6, outdoor=[0] * 7)
        self.assertEqual(find_episodes(g), [(1, 6)])
        # The step from 3 to 4 touches the switch-off: the episode ends at 3.
        off = np.zeros(7, dtype=bool)
        off[3] = True
        self.assertEqual(find_episodes(g, off), [(1, 3)])

    def test_no_episode_starts_at_a_switched_off_step(self):
        g = grid_of(room=[17, 17, 17.5, 18, 18.5], setpoint=[17, 21, 21, 21, 21], outdoor=[0] * 5)
        off = np.zeros(5, dtype=bool)
        off[1] = True
        self.assertEqual(find_episodes(g, off), [])
        # The step before the rise does not matter.
        off = np.zeros(5, dtype=bool)
        off[0] = True
        self.assertEqual(find_episodes(g, off), [(1, 4)])


class TestGrid(unittest.TestCase):
    def frame(self, minutes, room=None, setpoint=None, outdoor=None):
        times = [START + pd.Timedelta(minutes=m) for m in minutes]
        columns = {"time": times}
        for name, values in (("room_temperature", room), ("target_temperature", setpoint),
                             ("outdoor_temperature", outdoor)):
            if values is not None:
                columns[name] = values
        return pd.DataFrame(columns)

    def test_room_is_interpolated_and_not_across_long_gaps(self):
        g = build_grid([self.frame([0, 20, 260], room=[17.0, 19.0, 19.0], setpoint=[21, 21, 21], outdoor=[0, 0, 0])])
        self.assertEqual(list(g.room[:3]), [17.0, 18.0, 19.0])
        self.assertTrue(np.isnan(g.room[3]))
        self.assertEqual(g.room[-1], 19.0)

    def test_the_setpoint_holds_its_last_value_for_a_day(self):
        g = build_grid([self.frame([0, 1440, 2900], room=[17, 17, 17], setpoint=[21, 18, 18], outdoor=[0, 0, 0])])
        self.assertEqual(g.setpoint[143], 21)
        self.assertEqual(g.setpoint[144], 18)
        self.assertEqual(g.setpoint[288], 18)
        self.assertTrue(np.isnan(g.setpoint[289]))

    def test_inputs_of_the_same_kind_are_averaged(self):
        g = build_grid([self.frame([0, 1440], room=[17, 17], setpoint=[21, 18], outdoor=[0, 0]),
                        self.frame([1500], setpoint=[19])])
        self.assertEqual(g.setpoint[144], 18)
        self.assertEqual(g.setpoint[150], 18.5)

    def test_each_device_is_interpolated_on_its_own(self):
        # Two thermostats 2 K apart reporting alternately: the zone is their mean at every grid point.
        a = self.frame([0, 10, 20], room=[20.0, 20.0, 20.0], setpoint=[21, 21, 21], outdoor=[0, 0, 0]).assign(device_id="a")
        b = self.frame([5, 15, 25], room=[22.0, 22.0, 22.0], setpoint=[21, 21, 21]).assign(device_id="b")
        g = build_grid([pd.concat([a, b], ignore_index=True)])
        # At 00:00 only a has reported.
        self.assertEqual(list(g.room[:3]), [20.0, 21.0, 21.0])

    def test_values_at_or_below_absolute_zero_are_missing(self):
        g = build_grid([self.frame([0, 10, 20], room=[17, 17, 17], setpoint=[21, 21, 21], outdoor=[1.0, -999.0, 3.0])])
        self.assertEqual(list(g.outdoor), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
