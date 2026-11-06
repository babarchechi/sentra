# Sentra evaluation results

Owner: babarchechi. Protocol: docs/PHASE1_PLAN.md, Section 11. Consenting teammates only, mat for every drop,
no video recorded (CSV event logs only).

**Session date/time:** ____  **Camera position:** ____  **Average fps:** ____  **Thresholds (commit):** ____

| # | Scenario | Runs | Correct alerts | False alarms | Misses | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Walk across the frame (expect no alert) | | | | | |
| 2 | Sit down on a chair (expect no alert) | | | | | |
| 3 | Bend to pick something up (expect no alert) | | | | | |
| 4 | Sit on the floor slowly (expect no alert) | | | | | |
| 5 | Lie down slowly on the mat (expect ON_GROUND, no alert) | | | | | |
| 6 | Controlled quick drop, stay down 8 s (expect alert after ~5 s) | | | | | |
| 7 | Quick drop, get up within 2 s (expect no alert) | | | | | |
| 8 | Two people, one does #6 (expect alert for that ID only) | | | | | |
| 9 | Partly out of frame / behind a chair (expect no alert) | | | | | |

## Phone / IP camera stream

Status: not tested yet.

## Observations and known limits

-
