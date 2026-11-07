"""Tests for src/guide_bench.py (the Sentra Guide benchmark). Owner: iesa242."""
import pytest
from guide_helpers import CHAIR_AHEAD, FakeClock, FakeModel, FakeSource

import guide_bench
from guide_bench import BenchSink, Sample, markdown_table, printable_command, run_config, summarize
from sentra.guide.detector import ObjectDetector


def fake_detector(imgsz, track):
    return ObjectDetector(model=FakeModel(CHAIR_AHEAD), imgsz=imgsz, track=track)


def test_run_config_measures_frames_after_warmup():
    result = run_config(lambda: FakeSource(30), fake_detector, 320, True, frames=20, warmup=5)
    assert result.frames == 20 and result.imgsz == 320 and result.track
    assert result.objects_per_frame == 1.0  # the fake chair, every frame
    assert result.model_ms_median is not None and result.model_ms_p90 >= result.model_ms_median
    assert result.latency_ms_median is not None and result.latency_ms_p90 >= result.latency_ms_median
    assert result.fps > 0 and result.stale_frames == 0 and result.notes == []


def test_tracking_setting_reaches_the_model():
    detectors = []

    def make(imgsz, track):
        detectors.append(fake_detector(imgsz, track))
        return detectors[-1]

    run_config(lambda: FakeSource(5), make, 480, False, frames=3, warmup=1)
    calls = detectors[0].model.calls
    assert calls[0][0] == "predict"  # warm-up
    assert {name for name, _ in calls[1:]} == {"predict"} and calls[1][1]["imgsz"] == 480
    detectors.clear()
    run_config(lambda: FakeSource(5), make, 480, True, frames=3, warmup=1)
    assert {name for name, _ in detectors[0].model.calls[1:]} == {"track"}


def test_a_source_that_ends_early_is_reported():
    result = run_config(lambda: FakeSource(10), fake_detector, 640, True, frames=20, warmup=5)
    assert result.frames == 5 and result.notes == ["source ended early"]


def test_sink_gives_up_after_timeout():
    clock = FakeClock()
    sink = BenchSink(total=100, timeout_s=10.0, clock=clock)
    assert not sink.should_stop()
    clock.now = 10.5
    assert sink.should_stop() and sink.timed_out


def test_summary_skips_warmup_and_computes_percentiles_and_fps():
    samples = [Sample(at=99.0, model_ms=500.0, latency_ms=900.0, objects=0, stale=True)]  # warm-up frame
    samples += [Sample(at=i * 0.1, model_ms=float(i), latency_ms=float(i) + 10, objects=2, stale=False)
                for i in range(1, 11)]
    result = summarize(samples, 640, False, warmup=1)
    assert result.frames == 10
    assert result.model_ms_median == 5.5 and result.model_ms_p90 == 9.1
    assert result.latency_ms_median == 15.5
    assert result.fps == pytest.approx(10.0) and result.seconds == pytest.approx(0.9)
    assert result.objects_per_frame == 2.0 and result.stale_frames == 0


def test_summary_with_nothing_measured():
    result = summarize([], 320, True, warmup=5)
    assert result.frames == 0 and result.model_ms_median is None and result.notes == ["no frames measured"]
    assert "n/a" in markdown_table([result])


def test_printed_command_hides_stream_urls_and_folders(tmp_path):
    url = "http://user:secret@10.0.0.5:8080/video"
    shown = printable_command(["--source", url, "--frames", "50"])
    assert shown == "python src/guide_bench.py --source '<stream URL>' --frames 50"
    assert "10.0.0.5" not in printable_command([f"--source={url}"])
    clip = tmp_path / "private" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"x")
    assert printable_command(["--source", str(clip)]) == "python src/guide_bench.py --source clip.mp4"
    assert printable_command(["--source", "0"]) == "python src/guide_bench.py --source 0"


def test_markdown_table_has_one_row_per_configuration():
    rows = [summarize([Sample(i * 0.05, 30.0, 40.0, 1, False) for i in range(5)], size, True, 0)
            for size in (640, 320)]
    table = markdown_table(rows).splitlines()
    assert len(table) == 4 and table[0].startswith("| imgsz")
    assert table[2].startswith("| 640 | on | 5 / 0.2 | 20 | 30 / 30 | 40 / 40 | 1 | 0 |")


def test_main_runs_every_configuration_and_prints_the_table(monkeypatch, capsys):
    monkeypatch.setattr(guide_bench, "FrameSource", lambda source, rotate=0: FakeSource(15))
    monkeypatch.setattr(guide_bench, "ObjectDetector",
                        lambda model, conf, imgsz, track: fake_detector(imgsz, track))
    results = guide_bench.main(["--imgsz", "640", "320", "--frames", "10", "--warmup", "2"])
    assert [(r.imgsz, r.track) for r in results] == [(640, True), (640, False), (320, True), (320, False)]
    out = capsys.readouterr().out
    assert "Command: `python src/guide_bench.py --imgsz 640 320 --frames 10 --warmup 2`" in out
    assert out.count("\n| 640 |") == 2 and out.count("\n| 320 |") == 2


def test_bad_arguments_rejected():
    with pytest.raises(SystemExit):
        guide_bench.main(["--frames", "1"])
