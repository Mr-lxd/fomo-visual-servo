"""Tests for explicit TFLite/LiteRT tensor inspection helpers."""

from __future__ import annotations

import numpy as np

from fomo_servo.deployment.tflite_runtime import (
    inspect_tflite_interpreter,
    output_looks_like_probabilities,
    prepare_tflite_input,
)


class _FakeInterpreter:
    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self):
        return [
            {
                "name": "serving_default_input:0",
                "shape": np.asarray([1, 192, 192, 3], dtype=np.int32),
                "shape_signature": np.asarray([1, 192, 192, 3], dtype=np.int32),
                "dtype": np.float32,
                "quantization": (0.0, 0),
                "quantization_parameters": {"scales": np.asarray([]), "zero_points": np.asarray([])},
                "index": 0,
            }
        ]

    def get_output_details(self):
        return [
            {
                "name": "StatefulPartitionedCall:0",
                "shape": np.asarray([1, 24, 24, 8], dtype=np.int32),
                "shape_signature": np.asarray([1, 24, 24, 8], dtype=np.int32),
                "dtype": np.float32,
                "quantization": (0.0, 0),
                "quantization_parameters": {"scales": np.asarray([]), "zero_points": np.asarray([])},
                "index": 7,
            }
        ]

    def _get_ops_details(self):
        return [{"op_name": "CONV_2D"}, {"op_name": "SOFTMAX"}]


def test_tensor_inspection_keeps_real_layout_dtype_and_operator_names() -> None:
    info = inspect_tflite_interpreter(_FakeInterpreter(), model_sha256="a" * 64)

    assert info["input"]["shape"] == [1, 192, 192, 3]
    assert info["input"]["dtype"] == "float32"
    assert info["output"]["shape"] == [1, 24, 24, 8]
    assert info["operators"] == ["CONV_2D", "SOFTMAX"]
    assert info["graph_has_softmax"] is True


def test_float_input_is_raw_hwc_batch_without_unrequested_scaling() -> None:
    image = np.full((192, 192, 3), 255, dtype=np.uint8)
    detail = _FakeInterpreter().get_input_details()[0]

    prepared = prepare_tflite_input(image, detail)

    assert prepared.shape == (1, 192, 192, 3)
    assert prepared.dtype == np.float32
    assert float(prepared.max()) == 255.0


def test_float_input_accepts_an_explicit_evidence_backed_scale() -> None:
    image = np.full((192, 192, 3), 255, dtype=np.uint8)
    detail = _FakeInterpreter().get_input_details()[0]

    prepared = prepare_tflite_input(image, detail, float_scale=1.0 / 255.0)

    assert prepared.dtype == np.float32
    assert float(prepared.max()) == 1.0


def test_probability_detection_requires_nonnegative_normalized_last_axis() -> None:
    probabilities = np.full((1, 2, 2, 3), 1.0 / 3.0, dtype=np.float32)

    assert output_looks_like_probabilities(probabilities) is True
    assert output_looks_like_probabilities(probabilities * 2.0) is False
