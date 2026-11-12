# analytics-operator-heating-warmup

Learns how fast each heating zone of a home warms up and tells how long before a plan window the heating has to start, so the room is warm when the window begins. One operator serves all zones of a smart service instance: one pod and one training a day, however many zones.

## Model

One zone as a 1R1C model: dT/dt = a - b (T - T_out), with `a` in K/h (heating power over heat capacity) and `b` in 1/h (1/tau).

The time from T0 to T_set at constant T_out: the room approaches T_inf = T_out + a/b. If T_inf <= T_set the target is unreachable; otherwise t = ln((T_inf - T0)/(T_inf - T_set))/b, and (T_set - T0)/a for b = 0. A warm-up longer than 6 h counts as unreachable.

## Inputs

- `room_temperature`: measured room temperature of the thermostats of all zones (value role `Reading`). The device of a message assigns it to its zone; a device in no zone is ignored. Several devices of a zone are averaged.
- `target_temperature`: setpoint of the same thermostats (value role `Target`), assigned and averaged likewise.
- `outdoor_temperature`: measured outdoor temperature, one for all zones, either a device of the user or `value.temperature_2m` of the shared import of all DWD stations (import-dwd-stations). DWD's -999 for a missing value is ignored.
- `forecast_temperature` and `forecasted_for`: `value.temperature_2m` and `value.forecasted_for` of the shared DWD MOSMIX import (import-dwd-mosmix), one message per station and run, both lists of equal length. Optional; without it the measured outdoor temperature is used.
- `outdoor_station_name`, `outdoor_station_lat`, `outdoor_station_long` and `forecast_station_name`, `forecast_station_lat`, `forecast_station_long`: `value.meta.name`, `value.meta.lat`, `value.meta.long` of the same two imports, for the name and distance in the station outputs. Optional. A mapping the import's export does not carry as a column leaves the measured import without an export, so its history is then read from Kafka (days, not `history_days`).

The two shared imports are narrowed to one station each with the import filters of analytics-operator-lib-python: the station nearest to `lat`/`long`, unless `measure_station` or `forecast_station` names one. A device as outdoor source needs no filter, and an input that is not mapped to an import gets none. A shared import with neither a station nor a location stops the operator at start, since it would mix every station.

The operator declares no selectors and tells the inputs apart by the mapped names, so any subset of them can be mapped.

## Configuration

| Name | Default | Meaning |
|---|---|---|
| `lat`, `long` | empty | location of the zone; picks the nearest stations and gives their distance |
| `measure_station` | empty | DWD station id (`meta.id`) of the measured outdoor temperature, instead of the nearest |
| `forecast_station` | empty | MOSMIX station id (`meta.id`) of the forecast, instead of the nearest |
| `zones` | (required) | JSON list of the zones, see below |
| `history_days` | 60 | days of history read at each training |
| `max_lead_minutes` | 240 | upper limit of a lead time |

`zones` lists each zone as `{"key": "zone:0", "name": "Salon", "devices": ["<device id>", ...], "schedule": {...}, "setback": 17}`. `key` must be unique and a device may belong to one zone only. `schedule` is the heating plan of the zone in the format of the heating process, `{"windows":[{"weekdays":[0..6],"from":"hh:mm","to":"hh:mm","temperature":21}]}`, as an object or as JSON text, and `setback` the temperature outside its windows; a zone without both has no plan.

The plan follows the heating process: Europe/Berlin wall time, weekday 0 is Monday, no or empty `weekdays` means every day, the highest covering window wins, `from` after `to` runs past midnight and belongs to the day it starts, `from` equal to `to` covers the whole day, `24:00` ends at midnight. Invalid values stop the operator at start, and so do a `setback` without a `schedule` or the reverse, a key used twice, a device in two zones, and an empty `zones`.

## Output

One message per zone, each at most once per 5 minutes and only when one of its values changed after rounding (minutes to whole minutes, temperatures to 0.1 K, `a` to 0.01, `b` to 0.001, half away from zero):

- `zone`, `zone_name`: `key` and `name` of the zone.
- `a`, `b`, `episodes`: the model of the zone; null without one.
- `warmup_minutes`: while the setpoint is above the room temperature, the remaining warm-up with the latest measured outdoor temperature; `-1` if unreachable; null otherwise or without model.
- `next_start` (ISO UTC), `next_target`: the next plan boundary within 24 h where the plan temperature rises above the current one.
- `next_lead_minutes`: warm-up time for that boundary; null without model or outdoor temperature.
- `windows`: JSON text, every rising boundary of the next 24 h as `{"start", "target", "outdoor", "lead_minutes"}`.
- `measure_station`, `forecast_station`: JSON text `{"id", "name", "distance_km"}` of the station in use. `id` is the station the filter resolved at start; `name` comes from the station's latest message, `distance_km` from its location to `lat`/`long` (0.1 km); each null while unknown. The field is null when the input is no shared import, such as an outdoor device. Both are the same in the message of every zone.

A lead time starts from the plan temperature just before the boundary (the setback unless another window precedes it); if the boundary is closer than that lead, it starts from the current room temperature instead. It uses the MOSMIX forecast at the boundary, from the latest run that has a value for that time (linear between two forecast hours), else the latest measured outdoor temperature. It is capped at `max_lead_minutes`; an unreachable target gives the cap.

Room temperature, setpoint and the measured outdoor temperature count for 6 h after their last message. The plan is evaluated at the time of the newest message; a late message does not move it back.

## Training

The operator is an `MLOperator` of analytics-operator-lib-python. At start without a model in MLflow, and then once a day (Berlin date), it trains all zones in one Ray task on the cluster (`algo/training.py`), each on `history_days` of its own thermostats and the shared outdoor temperature; the image starts with `uv run`, so Ray builds the environment of `uv.lock` on the workers. The forecast history is not read.

1. Room temperature and outdoor temperature are interpolated onto a 10-minute grid (not across gaps of more than 2 h and 3 h), the setpoint holds its last value for up to 24 h; several devices are averaged.
2. A warm-up episode starts where the setpoint rises to at least 1 K above the room temperature, and runs until the room reaches the setpoint minus 0.3 K, the setpoint drops, a value is missing, or 6 h pass. An episode needs at least two steps.
3. `a` and `b` are fitted by least squares on the slope of each step against the mean of T - T_out over it.

Fewer than 3 episodes, a fit with `a` <= 0 or `b` < 0, or no spread in T - T_out give a zone no model; it keeps its previous one and the reason is logged. The models of all zones are registered in MLflow as one model (`ZonesModel`); a training in which no zone got a new fit registers nothing. Per zone, `a`, `b`, the episode count and the RMS residual of the slopes (K/h) are logged to MLflow as `<zone key>.a` and so on, with characters MLflow does not allow in a name replaced by `_`. A history that could not be read, as when the Ray cluster has no CPU free, is read again after 10 minutes, up to six times.

## Versions

Every build on `main` pushes `:vX.Y.Z`; a branch `release/vN` builds only `:vN.x.y`. See `.github/workflows/prod.yml`. `.github/workflows/retag.yml` points a tag such as `:prod` at an existing version or digest.

Tests: `uv run --frozen python -m unittest discover -s tests -t .`
