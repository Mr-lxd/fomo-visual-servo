"""Joint B-box50 detection and centre-cell endpoint training (offline only)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from fomo_servo.config import _parse_augmentation_config
from fomo_servo.datasets.augmentation import AugmentationPipeline
from fomo_servo.datasets.rng import make_sample_rng, stable_sample_seed
from fomo_servo.datasets.yolo import AbsoluteBox
from fomo_servo.geometry.letterbox import LetterboxTransform, letterbox_rgb
from fomo_servo.models.mobilenet_v2_fomo import MobileNetV2FOMONet
from fomo_servo.training.engine import initialize_model_weights
from .bbox_targets import bbox_classification_loss, build_bbox_targets
from .keypoints import KPSample, build_keypoint_targets
from .targets import Box, VISIBILITIES
from .training import VIS_STRIDE, _read_rgb, seed_everything

INDEX_STRIDE = VIS_STRIDE * len(VISIBILITIES)


class KeypointTrainDataset(Dataset):
    """Same image/RNG augmentation as B-box50, plus [4,G,G] endpoint supervision.

    Box indices travel in high class-id bits so clipping/dropping cannot shift
    endpoint associations. Points use returned hflip/affine metadata, consume
    no RNG draws, and are masked individually outside the augmented image.
    """
    def __init__(self, samples: Sequence[KPSample], *, augmentation: Mapping[str, Any],
                 input_size: int, stride: int, seed: int, object_weight: float):
        self.samples = list(samples)
        self.input_size, self.stride, self.seed = input_size, stride, seed
        self.object_weight = object_weight
        self.pipeline = AugmentationPipeline(_parse_augmentation_config(dict(augmentation)), is_train=True)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample = self.samples[index]
        image = _read_rgb(sample.image_path)
        height, width = image.shape[:2]
        boxes = tuple(AbsoluteBox(
            b.class_id + VIS_STRIDE * VISIBILITIES.index(b.visibility) + INDEX_STRIDE * i,
            b.x_min, b.y_min, b.x_max, b.y_max,
        ) for i, b in enumerate(sample.boxes))
        result = self.pipeline.apply(
            image, boxes, make_sample_rng(self.seed, self.current_epoch, index),
            epoch=self.current_epoch, sample_index=index,
            sample_seed=stable_sample_seed(self.seed, self.current_epoch, index),
        )
        letterboxed, transform = letterbox_rgb(result.image, self.input_size)
        metadata = result.metadata
        matrix = np.eye(3, dtype=np.float64)[:2]
        if metadata.affine_applied:
            matrix = cv2.getRotationMatrix2D((width / 2., height / 2.), metadata.affine_rotation, metadata.affine_scale)
            matrix[0, 2] += metadata.affine_translate_x
            matrix[1, 2] += metadata.affine_translate_y

        def transform_point(point):
            if point is None:
                return None
            x, y = point
            if metadata.horizontal_flip_applied:
                x = width - 1 - x  # endpoint coordinates are image pixel centres
            x, y = matrix @ np.asarray([x, y, 1.])
            if not (0 <= x < width and 0 <= y < height):
                return None
            return transform.forward_point(float(x), float(y))

        lb_boxes, lb_keypoints = [], []
        for box in result.boxes:
            code = box.foreground_class_id
            original_index = code // INDEX_STRIDE
            code %= INDEX_STRIDE
            lb_boxes.append(Box(code % VIS_STRIDE, *transform.forward_box(
                box.x_min, box.y_min, box.x_max, box.y_max,
            ), VISIBILITIES[code // VIS_STRIDE]))
            kp = sample.keypoints[original_index]
            lb_keypoints.append(replace(kp, head=transform_point(kp.head), tail=transform_point(kp.tail)))
        grid = self.input_size // self.stride
        return {
            "image": np.ascontiguousarray(letterboxed.transpose(2, 0, 1), dtype=np.float32) / 255.,
            **build_bbox_targets(lb_boxes, grid_size=grid, stride=self.stride,
                                 central_half=True, object_weight=self.object_weight),
            **build_keypoint_targets(lb_boxes, lb_keypoints, grid_size=grid, stride=self.stride),
        }


class FOMOKeypointNet(MobileNetV2FOMONet):
    """Original backbone/head state names plus random Conv2d(32,4,1).

    forward returns detection logits [B,8,G,G] and unbounded signed-log1p
    offsets [B,4,G,G]. Both branches share the original head's ReLU features.
    Detection initialisation happens before creating the random branch.
    """
    def __init__(self, *, init_weights: Path, init_sha256: str, **kwargs):
        super().__init__(**kwargs)
        self.init_report = initialize_model_weights(self, init_weights, init_sha256)
        self.keypoint_head = nn.Conv2d(self.head_channels, 4, kernel_size=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 RGB [B,3,S,S] → logits [B,8,G,G], offsets [B,4,G,G]."""
        self._validate_images(images)
        shared = self.head[1](self.head[0](self.backbone(images)))
        return self.head[2](shared), self.keypoint_head(shared)


def keypoint_loss(offsets: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked scalar Smooth-L1 for [B,4,G,G]; mean over annotated coordinates.

    Float32 evaluation is AMP-safe; an empty mask yields differentiable zero.
    """
    terms = F.smooth_l1_loss(offsets.float(), targets.float(), reduction="none")
    return (terms * mask.float()).sum() / mask.float().sum().clamp_min(1.)


def train_keypoint_model(
    epochs: int, train_samples: Sequence[KPSample], cfg: Mapping[str, Any],
    init_weights: Path, log: Any = print, snapshot_epochs: Sequence[int] = (), on_snapshot: Any = None,
) -> tuple[nn.Module, list[dict[str, float]], dict]:
    """Joint train every weight; return model, epoch history and init/AMP report."""
    t, m = cfg["training"], cfg["model"]
    weight = cfg["keypoint_loss"]["weight"]
    seed_everything(t["seed"])
    device = torch.device(t["device"])
    model = FOMOKeypointNet(
        init_weights=init_weights, init_sha256=m["init_sha256"],
        num_classes=m["num_classes"], input_size=m["input_size"],
        width_multiplier=m["width_multiplier"], head_channels=m["head_channels"],
    ).to(device)
    dataset = KeypointTrainDataset(
        train_samples, augmentation=cfg["augmentation"], input_size=m["input_size"],
        stride=m["output_stride"], seed=t["seed"], object_weight=cfg["fomo_loss"]["object_weight"],
    )
    loader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(t["seed"]), num_workers=t["num_workers"])
    opt_cfg = t["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=t["amp_initial_scale"], enabled=use_amp)
    history, skipped_steps = [], 0
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        totals, count = {}, 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, offsets = model(batch["image"])
                detection = bbox_classification_loss(logits, batch["target"], batch["positive_weight"], batch["loss_mask"])
                endpoint = keypoint_loss(offsets, batch["keypoint_target"], batch["keypoint_mask"])
                loss = detection + weight * endpoint
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at epoch {}".format(epoch + 1))
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped_steps += 1
            n = batch["image"].shape[0]
            for name, value in (("loss", loss), ("detection_loss", detection), ("keypoint_loss", endpoint)):
                totals[name] = totals.get(name, 0.) + float(value.detach()) * n
            count += n
        row = {"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}}
        history.append(row)
        if on_snapshot is not None and epoch + 1 in snapshot_epochs:
            on_snapshot(epoch + 1, model)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            log("  epoch {}/{}: {}".format(epoch + 1, epochs, row))
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError("non-finite weights in {}".format(name))
    return model.eval(), history, {**model.init_report, "amp_skipped_steps": skipped_steps,
                                 "keypoint_initialization": "pytorch_module_defaults", "keypoint_loss_weight": weight}


@torch.no_grad()
def predict_keypoint_raw(model: nn.Module, samples: Sequence[KPSample], input_size: int,
                         device: str = "cuda") -> tuple[np.ndarray, np.ndarray, list[LetterboxTransform]]:
    """Return float32 logits [N,8,G,G], offsets [N,4,G,G] and letterbox transforms."""
    model.eval().to(device)
    outputs, offsets, transforms = [], [], []
    for sample in samples:
        letterboxed, transform = letterbox_rgb(_read_rgb(sample.image_path), input_size)
        tensor = torch.from_numpy(np.ascontiguousarray(letterboxed.transpose(2, 0, 1), dtype=np.float32) / 255.)[None].to(device)
        logits, endpoint = model(tensor)
        outputs.append(logits[0].float().cpu().numpy())
        offsets.append(endpoint[0].float().cpu().numpy())
        transforms.append(transform)
    return np.stack(outputs), np.stack(offsets), transforms
