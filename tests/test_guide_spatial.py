"""Direction zones, severity and depth aggregation for Sentra Guide."""
import numpy as np
import pytest

from guide_helpers import det, frame
from sentra.guide.depth import NoDepth, box_depth
from sentra.guide.schema import DepthEstimate, Region, Severity
from sentra.guide.spatial import SpatialConfig, SpatialEstimator


@pytest.mark.parametrize("cx, region", [
    (0.05, Region.LEFT), (0.34, Region.LEFT), (0.36, Region.CENTER), (0.5, Region.CENTER),
    (0.64, Region.CENTER), (0.66, Region.RIGHT), (0.95, Region.RIGHT),
])
def test_region_from_box_centre(cx, region):
    (obj,) = SpatialEstimator().locate(frame(det(cx=cx)))
    assert obj.region is region
    assert obj.center_x == pytest.approx(cx)


def test_severity_central_and_large_is_high_central_small_is_medium_side_is_low():
    big, small, side, side_overlapping = SpatialEstimator().locate(frame(
        det(cx=0.5, height=0.6), det(cx=0.5, height=0.2), det(cx=0.1, width=0.1), det(cx=0.3, width=0.2),
    ))
    assert big.severity is Severity.HIGH
    assert small.severity is Severity.MEDIUM
    assert (side.severity, side.in_corridor) == (Severity.LOW, False)
    # centre 0.3 is LEFT, but the box reaches 0.4, inside the central band
    assert (side_overlapping.region, side_overlapping.in_corridor) == (Region.LEFT, True)
    assert side_overlapping.severity is Severity.MEDIUM


def test_tracked_object_near_an_edge_does_not_flip_flop():
    spatial = SpatialEstimator(SpatialConfig(left_edge=0.35, hysteresis=0.04))
    regions = [spatial.locate(frame(det(cx=cx, track_id=7), t=100 + i))[0].region
               for i, cx in enumerate([0.40, 0.34, 0.36, 0.33, 0.30])]
    # stays CENTER until it is clearly past the edge (below 0.31)
    assert regions == [Region.CENTER] * 4 + [Region.LEFT]
    # an untracked box has no memory
    assert spatial.locate(frame(det(cx=0.34)))[0].region is Region.LEFT


def test_bad_region_edges_rejected():
    with pytest.raises(ValueError):
        SpatialEstimator(SpatialConfig(left_edge=0.7, right_edge=0.3))


def test_depths_line_up_with_detections_and_default_to_unknown():
    d = DepthEstimate(1.4, 0.9, "record3d", validated=True)
    a, b = SpatialEstimator().locate(frame(det(cx=0.2), det(cx=0.8)), [d, DepthEstimate()])
    assert a.depth == d and b.depth.meters is None
    (c,) = SpatialEstimator().locate(frame(det()))
    assert c.depth.meters is None and not c.depth.speakable


def test_no_depth_provider_never_gives_numbers():
    f = frame(det(), det(cx=0.1))
    provider = NoDepth()
    assert [e.meters for e in provider.estimate(np.zeros((10, 10, 3)), f)] == [None, None]
    assert "off" in provider.status()


def test_box_depth_uses_median_of_valid_pixels_and_scales_box_to_depth_resolution():
    depth = np.full((192, 256), 3.0, np.float32)  # Record3D-sized depth map, background at 3 m
    depth[60:130, 100:160] = 1.5                   # the object
    depth[90, 120] = np.nan                         # a few invalid pixels
    depth[95, 125] = 0.0
    # Same object in a 1920 x 1440 RGB frame: depth pixels * 7.5
    box = (100 * 7.5, 60 * 7.5, 160 * 7.5, 130 * 7.5)
    meters, valid = box_depth(depth, box, (1920, 1440))
    assert meters == pytest.approx(1.5)
    assert 0.95 < valid < 1.0


def test_box_depth_returns_none_when_too_few_pixels_are_valid_or_confident():
    depth = np.full((100, 100), 2.0, np.float32)
    confidence = np.zeros((100, 100), np.uint8)  # Record3D confidence: 0 low, 1 medium, 2 high
    confidence[:, :10] = 2
    meters, valid = box_depth(depth, (20, 20, 80, 80), (100, 100), confidence=confidence, min_confidence=2)
    assert meters is None and valid == 0.0
    holes = np.full((100, 100), np.nan, np.float32)
    assert box_depth(holes, (20, 20, 80, 80), (100, 100))[0] is None
