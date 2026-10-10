"""CenterNet-lite labels, loss masking, visibility parsing, decoding and ONNX parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fomo_servo.centernet.annotations import (
    VisibilityError,
    read_visibility,
)
from fomo_servo.centernet.decode import decode_centernet
from fomo_servo.centernet.loss import centernet_loss
from fomo_servo.centernet.model import CenterNetLiteNet
from fomo_servo.centernet.targets import Box, build_targets, gaussian_radius
from fomo_servo.geometry.letterbox import LetterboxTransform

S, STRIDE, G, N = 192, 8, 24, 7


def _targets(boxes):
    return build_targets(boxes, grid_size=G, stride=STRIDE, num_classes=N)


def test_single_target_peak_is_one_at_center_cell() -> None:
    t = _targets([Box(1, 60.0, 76.0, 100.0, 108.0, "full")])  # centre (80, 92) -> cell (10, 11)
    assert t.heat.shape == (N, G, G)
    assert t.heat[1].max() == 1.0
    assert np.argwhere(t.heat[1] == 1.0).tolist() == [[11, 10]]
    assert t.heat[1, 11, 11] < 1.0 and t.heat[1, 11, 11] > 0.0
    assert t.heat[0].max() == 0.0
    assert np.allclose(t.offset[:, 11, 10], [0.0, 0.5])
    assert np.allclose(t.size[:, 11, 10], [np.log(5.0), np.log(4.0)])
    assert t.reg_mask.sum() == 1 and t.size_mask.sum() == 1


def test_two_adjacent_targets_each_keep_a_peak() -> None:
    t = _targets(
        [
            Box(0, 40.0, 40.0, 72.0, 72.0, "full"),  # centre (56, 56) -> cell (7, 7)
            Box(0, 56.0, 40.0, 88.0, 72.0, "full"),  # centre (72, 56) -> cell (9, 7)
        ]
    )
    peaks = np.argwhere(t.heat[0] == 1.0).tolist()
    assert sorted(peaks) == [[7, 7], [7, 9]]
    assert 0.0 < t.heat[0, 7, 8] < 1.0


def test_gaussian_radius_matches_centernet_reference() -> None:
    assert gaussian_radius(8.0, 8.0, 0.7) == pytest.approx(2.1866, abs=1e-3)  # r3 root of the reference
    assert gaussian_radius(8.0, 8.0) < gaussian_radius(16.0, 16.0)


def test_ignore_box_masks_heat_loss_cells_and_has_no_regression() -> None:
    t = _targets([Box(0, 80.0, 80.0, 112.0, 112.0, "ignore")])
    assert t.heat.max() == 0.0
    assert t.reg_mask.sum() == 0 and t.size_mask.sum() == 0
    assert (t.heat_weight[10:14, 10:14] == 0).all()
    assert t.heat_weight[0, 0] == 1.0
    assert t.heat_weight.sum() == G * G - 16


def test_ignore_does_not_mask_another_objects_positive() -> None:
    t = _targets(
        [
            Box(0, 80.0, 80.0, 112.0, 112.0, "ignore"),
            Box(1, 84.0, 84.0, 100.0, 100.0, "full"),  # centre cell (11, 11)
        ]
    )
    assert t.heat[1, 11, 11] == 1.0
    assert t.heat_weight[11, 11] == 1.0


@pytest.mark.parametrize("vis", ["truncated", "occluded"])
def test_truncated_and_occluded_keep_heat_and_offset_but_not_size(vis: str) -> None:
    t = _targets([Box(0, 60.0, 60.0, 100.0, 100.0, vis)])
    assert t.heat[0].max() == 1.0
    assert t.reg_mask.sum() == 1
    assert t.size_mask.sum() == 0


def _batch(t):
    return {
        "heat": torch.from_numpy(t.heat)[None],
        "offset": torch.from_numpy(t.offset)[None],
        "size": torch.from_numpy(t.size)[None],
        "reg_mask": torch.from_numpy(t.reg_mask)[None],
        "size_mask": torch.from_numpy(t.size_mask)[None],
        "heat_weight": torch.from_numpy(t.heat_weight)[None],
    }


def _raw(batch, size_noise=0.0):
    heat = torch.logit(batch["heat"].clamp(1e-3, 1 - 1e-3))
    off = torch.logit(batch["offset"].clamp(1e-3, 1 - 1e-3))
    return torch.cat([heat, off, batch["size"] + size_noise], dim=1)


def test_ignored_cells_do_not_enter_heat_loss() -> None:
    t = _targets([Box(0, 80.0, 80.0, 112.0, 112.0, "ignore")])
    batch = _batch(t)
    raw = torch.full((1, 11, G, G), -5.0)
    base, _ = centernet_loss(raw, batch)
    bad = raw.clone()
    bad[:, 0, 10:14, 10:14] = 8.0  # confident detections inside the ignore region
    changed, _ = centernet_loss(bad, batch)
    assert torch.allclose(base, changed)
    outside = raw.clone()
    outside[:, 0, 0, 0] = 8.0
    assert centernet_loss(outside, batch)[0] > base


def test_size_loss_is_masked_for_truncated_and_active_for_full() -> None:
    for vis, expect_active in (("truncated", False), ("full", True)):
        t = _targets([Box(0, 60.0, 60.0, 100.0, 100.0, vis)])
        batch = _batch(t)
        _, parts_a = centernet_loss(_raw(batch), batch)
        _, parts_b = centernet_loss(_raw(batch, size_noise=2.0), batch)
        assert (parts_b["size"] > parts_a["size"]) == expect_active
        assert parts_a["size"] == pytest.approx(0.0, abs=1e-6)


def test_perfect_prediction_has_near_zero_loss() -> None:
    t = _targets([Box(1, 60.0, 76.0, 100.0, 108.0, "full")])
    batch = _batch(t)
    total, _ = centernet_loss(_raw(batch), batch)
    assert float(total) < 0.05


def _synthetic_output(peaks):
    out = np.zeros((N + 4, G, G), dtype=np.float32)
    for cls, gx, gy, score, ox, oy, lw, lh in peaks:
        out[cls, gy, gx] = score
        out[N, gy, gx] = ox
        out[N + 1, gy, gx] = oy
        out[N + 2, gy, gx] = lw
        out[N + 3, gy, gx] = lh
    return out


def test_decode_two_peaks_and_original_coordinates() -> None:
    transform = LetterboxTransform.from_image_size(640, 480, S)  # scale 0.3, pad_top 24
    out = _synthetic_output(
        [
            (0, 10, 10, 0.9, 0.25, 0.75, np.log(4.0), np.log(2.0)),
            (0, 12, 10, 0.8, 0.5, 0.5, np.log(3.0), np.log(3.0)),
        ]
    )
    dets = decode_centernet(out, transform, stride=STRIDE, threshold=0.3)
    assert len(dets) == 2
    first = max(dets, key=lambda d: d.score)
    lb_x, lb_y = (10 + 0.25) * STRIDE, (10 + 0.75) * STRIDE
    ox, oy = transform.inverse_point(lb_x, lb_y)
    assert first.x == pytest.approx(ox) and first.y == pytest.approx(oy)
    assert first.w == pytest.approx(4.0 * STRIDE / transform.scale)
    assert first.h == pytest.approx(2.0 * STRIDE / transform.scale)
    assert first.class_id == 0


def test_decode_resolves_two_adjacent_cell_peaks_of_equal_height() -> None:
    """Training targets give two adjacent centres equal height 1.0; both survive 3x3 NMS."""

    transform = LetterboxTransform.from_image_size(192, 192, S)
    t = _targets([Box(0, 40.0, 40.0, 72.0, 72.0, "full"), Box(0, 48.0, 40.0, 80.0, 72.0, "full")])
    out = np.concatenate([t.heat, t.offset, t.size], axis=0)
    dets = decode_centernet(out, transform, stride=STRIDE, threshold=0.5)
    assert len(dets) == 2
    assert sorted(round(d.x) for d in dets) == [56, 64]


def test_decode_threshold_and_plateau_suppression() -> None:
    transform = LetterboxTransform.from_image_size(192, 192, S)
    out = _synthetic_output([(2, 5, 5, 0.2, 0.5, 0.5, 0.0, 0.0)])
    assert decode_centernet(out, transform, stride=STRIDE, threshold=0.3) == ()
    out[2, 5, 6] = 0.5  # stronger neighbour suppresses the weaker local maximum
    out[2, 5, 5] = 0.45
    dets = decode_centernet(out, transform, stride=STRIDE, threshold=0.3)
    assert len(dets) == 1 and dets[0].score == pytest.approx(0.5)


@pytest.mark.parametrize("bad", ["Full", "visible", 3])
def test_unknown_visibility_is_an_error(bad) -> None:
    with pytest.raises(VisibilityError):
        read_visibility({"attributes": {"visibility": bad}}, "x.json")


def test_visibility_precedence_attributes_then_description_then_full() -> None:
    assert read_visibility({"attributes": {"visibility": "occluded"}, "description": "ignore"}, "x") == "occluded"
    assert read_visibility({"attributes": {}, "description": "truncated"}, "x") == "truncated"
    assert read_visibility({"attributes": {}, "description": ""}, "x") == "full"
    assert read_visibility({}, "x") == "full"
    with pytest.raises(VisibilityError):
        read_visibility({"attributes": {}, "description": "partly hidden"}, "x")


def test_model_output_shape_and_param_increment() -> None:
    model = CenterNetLiteNet(num_classes=N, input_size=S)
    out = model(torch.zeros(2, 3, S, S))
    assert out.shape == (2, N + 4, G, G)
    assert float(out[:, :N].max()) < 0.2  # bias init -2.19 -> ~0.1
    assert float(out[:, N : N + 2].min()) >= 0.0 and float(out[:, N : N + 2].max()) <= 1.0


def test_onnx_export_matches_pytorch(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")
    from fomo_servo.centernet.export import export_centernet_onnx

    torch.manual_seed(0)
    model = CenterNetLiteNet(num_classes=N, input_size=S).eval()
    path = tmp_path / "c.onnx"
    export_centernet_onnx(model, path, opset=17)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_outputs()[0].shape == [1, N + 4, G, G]
    x = torch.rand(1, 3, S, S)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_centroid_matching_adjacency_and_spearman() -> None:
    from fomo_servo.centernet.annotations import GroundTruthBox
    from fomo_servo.centernet.evaluation import Pred, adjacent_pairs, evaluate_threshold, spearman

    gts = [GroundTruthBox(0, 0, 0, 60, 60), GroundTruthBox(1, 50, 0, 110, 60), GroundTruthBox(0, 300, 300, 360, 360, "ignore")]
    preds = [Pred(0, 0.9, 30, 30), Pred(0, 0.8, 80, 30), Pred(0, 0.7, 330, 330), Pred(0, 0.6, 500, 500)]
    assert adjacent_pairs(gts, 81.0) == [(0, 1)]
    agnostic = evaluate_threshold([preds], [gts], class_agnostic=True, adjacent_distance=81.0)
    assert (agnostic["tp"], agnostic["fp"], agnostic["fn"]) == (2, 1, 0)  # ignore-box hit is not an FP
    assert agnostic["adjacent_separation_rate"] == 1.0
    aware = evaluate_threshold([preds], [gts], class_agnostic=False, adjacent_distance=81.0)
    assert (aware["tp"], aware["fn"]) == (1, 1)  # second pred has the wrong class
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
