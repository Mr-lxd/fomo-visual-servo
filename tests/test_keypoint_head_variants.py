"""Minimal round-2 keypoint-head contract: architecture and gradient isolation."""
from pathlib import Path

import pytest
import torch

from fomo_servo.centernet import keypoint_training

PARAMETERS = {"linear1x1": 132, "conv3x3_relu_conv1x1": 9380}


@pytest.fixture
def build(monkeypatch):
    """Build the keypoint model without touching the external initialisation file."""
    monkeypatch.setattr(keypoint_training, "initialize_model_weights",
                        lambda model, path, sha256: {"path": str(path), "sha256": sha256})

    def factory(arch: str, detach: bool = False):
        return keypoint_training.FOMOKeypointNet(
            init_weights=Path("unused.pt"), init_sha256="0" * 64, keypoint_head_arch=arch,
            detach_keypoint_input=detach, num_classes=7, input_size=192,
            width_multiplier=0.35, head_channels=32,
        )

    return factory


@pytest.mark.parametrize("arch", sorted(PARAMETERS))
def test_head_architectures_emit_four_offset_channels(build, arch):
    model = build(arch)
    assert sum(p.numel() for p in model.keypoint_head.parameters()) == PARAMETERS[arch]
    with torch.no_grad():
        logits, offsets = model(torch.rand(1, 3, 192, 192))
    assert logits.shape == (1, 8, 24, 24) and offsets.shape == (1, 4, 24, 24)


def test_detached_input_keeps_endpoint_gradient_off_the_shared_features(build):
    detached = build("conv3x3_relu_conv1x1", detach=True)
    detached.train()
    detached(torch.rand(1, 3, 192, 192))[1].sum().backward()
    assert all(p.grad is None for p in detached.backbone.parameters())
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in detached.head.parameters())
    assert all(p.grad is not None for p in detached.keypoint_head.parameters())

    shared = build("conv3x3_relu_conv1x1", detach=False)
    shared.train()
    shared(torch.rand(1, 3, 192, 192))[1].sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in shared.head.parameters())
