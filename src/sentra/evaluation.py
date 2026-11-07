"""Evaluation report. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 11.

Each evaluation run is one `python src/main.py --scenario N` session. That writes
outputs/transitions_<time>_s<N>.csv, and this module scores those files against what each
scenario should do: alert (a fall) or stay quiet (everything else).

Only the analyzer's state changes are used, so nothing here depends on the dashboard.
"""
from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

ALERT_STATE = "possible_fall"


@dataclass(frozen=True)
class Scenario:
    number: int
    description: str
    expected_alerts: int  # 0: nobody should alert. 1: exactly one person should.
    expect: str           # expected outcome, as worded in the plan


# Mirrors the table in docs/PHASE1_PLAN.md, Section 11. Keep them in sync.
SCENARIOS: dict[int, Scenario] = {
    s.number: s
    for s in (
        Scenario(1, "Walk across the frame", 0, "no alert"),
        Scenario(2, "Sit down on a chair normally", 0, "no alert"),
        Scenario(3, "Bend down to pick something up", 0, "no alert"),
        Scenario(4, "Sit on the floor slowly", 0, "no alert"),
        Scenario(5, "Lie down slowly on the mat", 0, "ON_GROUND, no alert"),
        Scenario(6, "Controlled quick drop onto the mat, stay down 8 s", 1, "alert after ~5 s"),
        Scenario(7, "Quick drop, get up within 2 s", 0, "no alert"),
        Scenario(8, "Two people, one does #6", 1, "alert for that person only"),
        Scenario(9, "Walk partly out of frame / behind a chair", 0, "no alert"),
    )
}

# transitions_20261107_183000_s6.csv, or transitions_20261107_183000_2_s6.csv for a same-second clash.
_LABELLED = re.compile(r"^transitions_\d{8}_\d{6}(?:_\d+)?_s(\d+)\.csv$")


def scenario_label(number: int) -> str:
    """The run label main.py passes to the pipeline, which ends up in the CSV filename."""
    return f"s{number}"


@dataclass(frozen=True)
class Transition:
    timestamp: float
    track_id: int
    from_state: str
    to_state: str


@dataclass(frozen=True)
class Session:
    """One evaluation run: the state changes from one transitions CSV."""

    path: Path
    scenario: int
    transitions: tuple[Transition, ...]

    @property
    def alerted_tracks(self) -> set[int]:
        return {t.track_id for t in self.transitions if t.to_state == ALERT_STATE}

    def reached(self, state: str) -> bool:
        return any(t.to_state == state for t in self.transitions)

    def alert_latency(self) -> float | None:
        """Seconds from the drop to the first alert, or None if there was no alert.

        The drop is the last DESCENDING change before the alert (from any track, since a track ID can
        change mid-fall), or the last ON_GROUND change if no DESCENDING was logged.
        """
        alerts = [t.timestamp for t in self.transitions if t.to_state == ALERT_STATE]
        if not alerts:
            return None
        first_alert = min(alerts)
        for state in ("descending", "on_ground"):
            starts = [t.timestamp for t in self.transitions if t.to_state == state and t.timestamp <= first_alert]
            if starts:
                return first_alert - max(starts)
        return None


def read_session(path: Path) -> Session | None:
    """Read a labelled transitions CSV, or return None if the file has no known scenario label."""
    match = _LABELLED.match(path.name)
    if not match or int(match.group(1)) not in SCENARIOS:
        return None
    with path.open(newline="", encoding="utf-8") as f:
        transitions = tuple(
            Transition(
                timestamp=float(row["timestamp"]),
                track_id=int(row["track_id"]),
                from_state=row["from_state"],
                to_state=row["to_state"],
            )
            for row in csv.DictReader(f)
        )
    return Session(path=path, scenario=int(match.group(1)), transitions=transitions)


def load_sessions(directory: Path) -> tuple[list[Session], list[Path]]:
    """Return (labelled sessions, transition files without a scenario label) found in `directory`."""
    sessions: list[Session] = []
    unlabelled: list[Path] = []
    for path in sorted(directory.glob("transitions_*.csv")):
        session = read_session(path)
        if session is None:
            unlabelled.append(path)
        else:
            sessions.append(session)
    return sessions, unlabelled


@dataclass
class ScenarioResult:
    scenario: Scenario
    runs: int = 0
    correct_alerts: int = 0   # fall scenarios: runs where the faller was flagged
    misses: int = 0           # fall scenarios: runs with no alert at all
    false_alarms: int = 0     # people flagged who shouldn't have been
    reached_on_ground: int = 0
    latencies: list[float] = field(default_factory=list)


def score(sessions: list[Session]) -> dict[int, ScenarioResult]:
    """Score every session against its scenario's expected outcome. Every scenario gets a result."""
    results = {number: ScenarioResult(scenario) for number, scenario in SCENARIOS.items()}
    for session in sessions:
        result = results[session.scenario]
        result.runs += 1
        alerted = len(session.alerted_tracks)
        if result.scenario.expected_alerts == 0:
            result.false_alarms += alerted
        elif alerted:
            result.correct_alerts += 1
            result.false_alarms += alerted - 1  # anyone besides the one faller
            latency = session.alert_latency()
            if latency is not None:
                result.latencies.append(latency)
        else:
            result.misses += 1
        if session.reached("on_ground"):
            result.reached_on_ground += 1
    return results


def _notes(result: ScenarioResult) -> str:
    notes = []
    if result.latencies:
        notes.append(f"median time to alert {statistics.median(result.latencies):.1f} s")
    if result.scenario.number == 5 and result.runs:
        notes.append(f"ON_GROUND in {result.reached_on_ground}/{result.runs} runs")
    return "; ".join(notes)


def markdown_table(results: dict[int, ScenarioResult]) -> str:
    """The scenario table from docs/evaluation.md, filled in. "n/a" marks columns that don't apply.

    Plain ASCII on purpose: Windows consoles and redirected output garble characters like en dashes.
    """
    lines = [
        "| # | Scenario | Runs | Correct alerts | False alarms | Misses | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for number in sorted(results):
        r = results[number]
        if not r.runs:
            cells = ["", "", "", ""]
        elif r.scenario.expected_alerts > 0:
            cells = [str(r.runs), str(r.correct_alerts), str(r.false_alarms), str(r.misses)]
        else:
            cells = [str(r.runs), "n/a", str(r.false_alarms), "n/a"]
        lines.append(f"| {number} | {r.scenario.description} (expect {r.scenario.expect}) | " + " | ".join(cells)
                     + f" | {_notes(r)} |")
    return "\n".join(lines)


def summary(results: dict[int, ScenarioResult]) -> str:
    """One-paragraph headline numbers, e.g. for the pitch."""
    fall = [r for r in results.values() if r.scenario.expected_alerts > 0]
    quiet = [r for r in results.values() if r.scenario.expected_alerts == 0]
    fall_runs = sum(r.runs for r in fall)
    detected = sum(r.correct_alerts for r in fall)
    quiet_runs = sum(r.runs for r in quiet)
    false_alarms = sum(r.false_alarms for r in results.values())
    latencies = [x for r in fall for x in r.latencies]

    parts = []
    if fall_runs:
        parts.append(f"Falls flagged: {detected} of {fall_runs} runs ({100 * detected / fall_runs:.0f}%).")
    else:
        parts.append("Falls flagged: no fall runs recorded yet.")
    parts.append(f"False alarms: {false_alarms} across {quiet_runs} non-fall runs"
                 + (" and the fall runs." if fall_runs else "."))
    if latencies:
        parts.append(f"Median time from drop to alert: {statistics.median(latencies):.1f} s.")
    return " ".join(parts)
