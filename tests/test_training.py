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

import unittest

import numpy as np
import pandas as pd

from algo.model import WarmupModel
from algo.training import STEP_NS, Grid, build_grid, find_episodes, fit_episodes, fit_zone, fit_zones
from tests.simulation import START, FakeDataset, UnreadableDataset, simulate

A, B = 4.0, 0.15
NOW = int(pd.Timestamp("2026-03-01", tz="UTC").value)


def outdoor_by_day(day: int) -> float:
    return [-4.0, 8.0, 2.0, 11.0, -1.0, 5.0, 0.0][day % 7]


def fit(datasets, consume=None):
    """One zone of every device: (model, notes, history failed, episodes) as fit_zones gives it for that zone."""
    results, notes, failed = fit_zones(datasets, consume, None, {"z": ["d1", "a", "b"]}, NOW)
    if failed:
        return None, notes, True, 0
    model, episodes = results["z"]
    return model, notes, False, episodes


def grid_of(room, setpoint, outdoor) -> Grid:
    n = len(room)
    return Grid(np.arange(n, dtype=np.int64) * STEP_NS, np.array(room, float), np.array(setpoint, float),
                np.array(outdoor, float))


class TestFit(unittest.TestCase):
    def assert_recovered(self, model):
        self.assertIsInstance(model, WarmupModel)
        self.assertLess(abs(model.a - A) / A, 0.10, model.a)
        self.assertLess(abs(model.b - B) / B, 0.10, model.b)

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
        self.assertLess(abs(model.a - A) / A, 0.002)
        self.assertLess(abs(model.b - B) / B, 0.002)

    def test_rounded_noisy_readings_every_ten_minutes(self):
        thermostat, weather = simulate(A, B, days=21, outdoor=outdoor_by_day, report_minutes=10, decimals=1)
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assert_recovered(model)

    def test_another_zone(self):
        thermostat, weather = simulate(2.5, 0.08, days=14, outdoor=lambda d: [3.0, -6.0, 9.0][d % 3])
        model, _, _, _ = fit([FakeDataset(thermostat), FakeDataset(weather)])
        self.assertLess(abs(model.a - 2.5) / 2.5, 0.10)
        self.assertLess(abs(model.b - 0.08) / 0.08, 0.10)

    def test_several_thermostats_are_averaged(self):
        thermostat, weather = simulate(A, B, days=10, outdoor=outdoor_by_day)
        # A second device reporting the same zone half a degree warmer at other moments, as one frame with device_id.
        other = thermostat.assign(time=thermostat["time"] + pd.Timedelta(minutes=2),
                                  room_temperature=thermostat["room_temperature"] + 0.5)
        both = pd.concat([thermostat.assign(device_id="a"), other.assign(device_id="b")], ignore_index=True)
        model, _, _, episodes = fit([FakeDataset(both), FakeDataset(weather)])
        self.assert_recovered(model)
        self.assertEqual(episodes, 10)

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
        self.assertLess(abs(first.a - A) / A, 0.10)
        self.assertLess(abs(first.b - B) / B, 0.10)
        self.assertLess(abs(second.a - 2.5) / 2.5, 0.10)
        self.assertLess(abs(second.b - 0.08) / 0.08, 0.10)

    def test_a_device_in_no_zone_is_left_out(self):
        living, weather = simulate(A, B, days=10, outdoor=outdoor_by_day, device="t1")
        # A thermostat of another room, much warmer, that no zone lists.
        stray = living.assign(device_id="elsewhere", room_temperature=living["room_temperature"] + 3)
        results, _, _ = fit_zones([FakeDataset(pd.concat([living, stray], ignore_index=True)), FakeDataset(weather)],
                                  None, None, {"zone:0": ["t1"]}, NOW)
        model, _ = results["zone:0"]
        self.assertLess(abs(model.a - A) / A, 0.01)

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
        self.assertIsNone(fit_episodes(g, [(1, 3)]))


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
