"""Minimal centre-cell offset encoding contract."""
import numpy as np

from fomo_servo.centernet.keypoints import decode_offsets, encode_offsets


def test_signed_offsets_round_trip_from_cell_geometric_centre():
    points = ((0.0, 191.0), (160.0, 16.0))
    encoded, mask = encode_offsets(*points, cell_x=7, cell_y=9, stride=8)
    np.testing.assert_array_equal(mask, np.ones(4))
    assert encoded[0] < 0 and encoded[1] > 0
    np.testing.assert_allclose(
        decode_offsets(encoded, cell_x=7, cell_y=9, stride=8), points, atol=2e-5,
    )


def test_single_endpoint_masks_only_missing_coordinates():
    encoded, mask = encode_offsets(None, (76.0, 84.0), cell_x=9, cell_y=10, stride=8)
    np.testing.assert_array_equal(mask, [0, 0, 1, 1])
    np.testing.assert_array_equal(encoded, np.zeros(4))
    np.testing.assert_allclose(decode_offsets(encoded, cell_x=9, cell_y=10, stride=8)[1], [76, 84])
