"""Tests for the evaluation report and run labels. Track B, owner: babarchechi."""
import csv

import pytest

from sentra.evaluation import SCENARIOS, load_sessions, markdown_table, score, summary
from sentra.events import PersonStatus, PostureState
from sentra.pipeline import TransitionLog

_run_counter = 0


def write_run(directory, scenario, rows, suffix=None):
    """Write a transitions CSV the way TransitionLog does. rows: (timestamp, track_id, from_state, to_state)."""
    global _run_counter
    _run_counter += 1
    label = f"_s{scenario}" if scenario is not None else ""
    name = suffix or f"transitions_20261107_18{_run_counter:04d}{label}.csv"
    path = directory / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TransitionLog.COLUMNS)
        for timestamp, track_id, from_state, to_state in rows:
            writer.writerow([f"{timestamp:.3f}", "2026-11-07T18:00:00.000", track_id, from_state, to_state, "0.80", "r"])
    return path


FALL = [  # upright, fast drop at t=10, alert at t=15.2
    (9.0, 1, "", "upright"),
    (10.0, 1, "upright", "descending"),
    (10.6, 1, "descending", "on_ground"),
    (15.2, 1, "on_ground", "possible_fall"),
]
NO_ALERT = [(1.0, 1, "", "upright")]


def results_for(tmp_path):
    sessions, unlabelled = load_sessions(tmp_path)
    return score(sessions), unlabelled


def test_fall_scenario_with_alert_is_correct_and_measures_time_to_alert(tmp_path):
    write_run(tmp_path, 6, FALL)
    results, _ = results_for(tmp_path)
    r = results[6]
    assert (r.runs, r.correct_alerts, r.misses, r.false_alarms) == (1, 1, 0, 0)
    assert r.latencies == [pytest.approx(5.2)]


def test_fall_scenario_without_alert_is_a_miss(tmp_path):
    write_run(tmp_path, 6, NO_ALERT)
    r = results_for(tmp_path)[0][6]
    assert (r.runs, r.correct_alerts, r.misses) == (1, 0, 1)


def test_alert_in_quiet_scenario_is_a_false_alarm(tmp_path):
    write_run(tmp_path, 1, FALL)
    write_run(tmp_path, 1, NO_ALERT)
    r = results_for(tmp_path)[0][1]
    assert (r.runs, r.false_alarms) == (2, 1)


def test_two_people_alerting_in_scenario_8_counts_one_extra_false_alarm(tmp_path):
    write_run(tmp_path, 8, FALL + [(16.0, 2, "upright", "possible_fall")])
    r = results_for(tmp_path)[0][8]
    assert (r.correct_alerts, r.false_alarms, r.misses) == (1, 1, 0)


def test_time_to_alert_falls_back_to_on_ground_and_spans_track_id_changes(tmp_path):
    # Track 1 drops, then ByteTrack gives the person a new ID (3) that goes straight to the alert.
    write_run(tmp_path, 6, [(10.0, 1, "upright", "on_ground"), (15.5, 3, "", "possible_fall")])
    r = results_for(tmp_path)[0][6]
    assert r.latencies == [pytest.approx(5.5)]


def test_unlabelled_and_unknown_scenario_files_are_skipped(tmp_path):
    write_run(tmp_path, None, FALL)                                        # recorded without --scenario
    write_run(tmp_path, 12, FALL, suffix="transitions_20261107_180000_s12.csv")  # no scenario 12
    write_run(tmp_path, 6, FALL)
    sessions, unlabelled = load_sessions(tmp_path)
    assert [s.scenario for s in sessions] == [6]
    assert len(unlabelled) == 2


def test_markdown_table_lists_every_scenario_and_marks_columns_that_do_not_apply(tmp_path):
    write_run(tmp_path, 5, [(1.0, 1, "", "upright"), (4.0, 1, "upright", "on_ground")])
    write_run(tmp_path, 6, FALL)
    table = markdown_table(results_for(tmp_path)[0])
    lines = table.splitlines()
    assert len(lines) == 2 + len(SCENARIOS)
    row5 = next(line for line in lines if line.startswith("| 5 |"))
    assert "(expect ON_GROUND, no alert)" in row5
    assert "| 1 | n/a | 0 | n/a |" in row5 and "ON_GROUND in 1/1 runs" in row5
    assert table.isascii()  # Windows consoles garble non-ASCII characters
    row6 = next(line for line in lines if line.startswith("| 6 |"))
    assert "| 1 | 1 | 0 | 0 |" in row6 and "median time to alert 5.2 s" in row6
    row1 = next(line for line in lines if line.startswith("| 1 |"))
    assert row1.endswith("|  |  |  |  |  |")  # no runs yet: blank cells


def test_summary_headline(tmp_path):
    write_run(tmp_path, 6, FALL)
    write_run(tmp_path, 6, NO_ALERT)
    write_run(tmp_path, 1, FALL)
    text = summary(results_for(tmp_path)[0])
    assert "Falls flagged: 1 of 2 runs (50%)." in text
    assert "False alarms: 1 across 1 non-fall runs" in text
    assert "Median time from drop to alert: 5.2 s." in text


def test_transition_log_label_round_trips_into_the_report(tmp_path):
    log = TransitionLog(tmp_path, label="s6")
    log.record([PersonStatus(1, PostureState.DESCENDING, 0.9, "r", 10.0)], timestamp=10.0)
    log.record([PersonStatus(1, PostureState.POSSIBLE_FALL, 0.9, "r", 15.0)], timestamp=15.0)
    log.close()
    assert log.path.name.endswith("_s6.csv")
    sessions, unlabelled = load_sessions(tmp_path)
    assert [s.scenario for s in sessions] == [6] and not unlabelled
    assert sessions[0].alert_latency() == pytest.approx(5.0)


def test_runs_started_in_the_same_second_do_not_overwrite_each_other(tmp_path):
    first = TransitionLog(tmp_path, label="s1")
    second = TransitionLog(tmp_path, label="s1")
    first.close()
    second.close()
    assert first.path != second.path
    assert len(load_sessions(tmp_path)[0]) == 2


def test_main_accepts_only_known_scenarios():
    import main

    assert main.build_parser().parse_args(["--scenario", "6"]).scenario == 6
    assert main.build_parser().parse_args([]).scenario is None
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--scenario", "12"])
