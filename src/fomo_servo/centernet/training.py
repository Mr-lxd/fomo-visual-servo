"""Shared dataset, training loop and raw-output prediction for B (FOMO) and C (CenterNet-lite)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fomo_servo.config import LossConfig, _parse_augmentation_config
from fomo_servo.datasets.augmentation import AugmentationPipeline
from fomo_servo.datasets.heatmap import generate_fomo_heatmap
from fomo_servo.datasets.rng import make_sample_rng, stable_sample_seed
from fomo_servo.datasets.yolo import AbsoluteBox
from fomo_servo.geometry.letterbox import LetterboxTransform, letterbox_rgb
from fomo_servo.losses.classification import FOMOClassificationLoss
from fomo_servo.models.mobilenet_v2_fomo import MobileNetV2FOMONet
from fomo_servo.training.engine import initialize_model_weights

from .annotations import PoolSample
from .loss import centernet_loss
from .model import CenterNetLiteNet, load_fomo_initialisation
from .targets import VISIBILITIES, Box, build_targets

VIS_STRIDE = 16  # visibility is carried through augmentation inside foreground_class_id


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("unable to read image: {}".format(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class CVTrainDataset(Dataset):
    """Augmented, letterboxed samples with either FOMO or CenterNet-lite targets.

    ``kind='fomo'`` uses every box as a target (the existing recipe is unchanged by
    visibility); ``kind='centernet'`` honours visibility. Both use the same
    ``AugmentationPipeline`` and per-(seed, epoch, index) RNG as the baseline.
    """

    def __init__(
        self,
        samples: Sequence[PoolSample],
        *,
        kind: str,
        augmentation: Mapping[str, Any],
        input_size: int,
        stride: int,
        num_classes: int,
        seed: int,
        min_overlap: float = 0.7,
    ) -> None:
        if kind not in {"fomo", "centernet"}:
            raise ValueError("kind must be 'fomo' or 'centernet'")
        self.samples = list(samples)
        self.kind = kind
        self.input_size, self.stride, self.num_classes = input_size, stride, num_classes
        self.seed, self.min_overlap = seed, min_overlap
        self.pipeline = AugmentationPipeline(_parse_augmentation_config(dict(augmentation)), is_train=True)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = _read_rgb(sample.image_path)
        boxes = [
            AbsoluteBox(
                b.class_id + VIS_STRIDE * VISIBILITIES.index(b.visibility),
                b.x_min, b.y_min, b.x_max, b.y_max,
            )
            for b in sample.boxes
        ]
        rng = make_sample_rng(self.seed, self.current_epoch, index)
        result = self.pipeline.apply(
            image, tuple(boxes), rng, epoch=self.current_epoch, sample_index=index,
            sample_seed=stable_sample_seed(self.seed, self.current_epoch, index),
        )
        letterboxed, transform = letterbox_rgb(result.image, self.input_size)
        lb_boxes = []
        for box in result.boxes:
            x0, y0, x1, y1 = transform.forward_box(box.x_min, box.y_min, box.x_max, box.y_max)
            lb_boxes.append(
                Box(
                    box.foreground_class_id % VIS_STRIDE, x0, y0, x1, y1,
                    VISIBILITIES[box.foreground_class_id // VIS_STRIDE],
                )
            )
        tensor = np.ascontiguousarray(letterboxed.transpose(2, 0, 1), dtype=np.float32) / 255.0
        grid = self.input_size // self.stride
        if self.kind == "fomo":
            heatmap = generate_fomo_heatmap(
                [((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2, b.class_id) for b in lb_boxes],
                self.input_size, self.stride, self.num_classes, collision_policy="keep_first",
            )
            return {"image": tensor, "target": heatmap.class_index}
        t = build_targets(
            lb_boxes, grid_size=grid, stride=self.stride, num_classes=self.num_classes,
            min_overlap=self.min_overlap,
        )
        return {
            "image": tensor, "heat": t.heat, "offset": t.offset, "size": t.size,
            "reg_mask": t.reg_mask, "size_mask": t.size_mask, "heat_weight": t.heat_weight,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(kind: str, cfg: Mapping[str, Any], init_weights: Path) -> tuple[nn.Module, dict]:
    m = cfg["model"]
    if kind == "fomo":
        model = MobileNetV2FOMONet(
            num_classes=m["num_classes"], input_size=m["input_size"],
            width_multiplier=m["width_multiplier"], head_channels=m["head_channels"],
        )
        report = initialize_model_weights(model, init_weights, m["init_sha256"])
    else:
        model = CenterNetLiteNet(
            num_classes=m["num_classes"], input_size=m["input_size"],
            width_multiplier=m["width_multiplier"], head_channels=m["head_channels"],
        )
        report = load_fomo_initialisation(model, init_weights, m["init_sha256"])
    return model, report


def train_model(
    kind: str,
    epochs: int,
    train_samples: Sequence[PoolSample],
    cfg: Mapping[str, Any],
    init_weights: Path,
    log: Any = print,
) -> tuple[nn.Module, list[dict[str, float]], dict]:
    """Train ``epochs`` fixed epochs; return the final-epoch model, history and init report."""

    t, m = cfg["training"], cfg["model"]
    seed_everything(t["seed"])
    device = torch.device(t["device"])
    model, report = build_model(kind, cfg, init_weights)
    model.to(device)
    dataset = CVTrainDataset(
        train_samples, kind=kind, augmentation=cfg["augmentation"],
        input_size=m["input_size"], stride=m["output_stride"], num_classes=m["num_classes"],
        seed=t["seed"], min_overlap=cfg["centernet_loss"]["gaussian_min_overlap"],
    )
    loader = DataLoader(
        dataset, batch_size=t["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(t["seed"]), num_workers=t["num_workers"],
    )
    opt_cfg = t["optimizer"]
    if kind == "centernet":
        new_ids = {id(p) for p in model.new_head_parameters()}
        groups = [
            {"params": [p for p in model.parameters() if id(p) not in new_ids], "lr": opt_cfg["learning_rate"]},
            {"params": model.new_head_parameters(), "lr": t["new_head_learning_rate"]},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": opt_cfg["learning_rate"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=opt_cfg["weight_decay"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=t["amp_initial_scale"], enabled=use_amp)
    fomo_loss = FOMOClassificationLoss(LossConfig(**cfg["fomo_loss"])).to(device) if kind == "fomo" else None
    cl = cfg["centernet_loss"]

    history = []
    skipped_steps = 0
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if kind == "fomo":
                    loss = fomo_loss(model(batch["image"]), batch["target"])
                    parts = {"loss": float(loss.detach())}
                else:
                    raw = model.forward_raw(batch["image"])
            if kind == "centernet":
                loss, parts = centernet_loss(
                    raw, batch, alpha=cl["heat"]["alpha"], beta=cl["heat"]["beta"],
                    offset_weight=cl["offset_weight"], size_weight=cl["size_weight"],
                )
                parts["loss"] = float(loss.detach())
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at epoch {}".format(epoch))
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)  # AMP skips the step on inf/NaN gradients and lowers the scale
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped_steps += 1
            n = batch["image"].shape[0]
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * n
            count += n
        row = {"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}}
        history.append(row)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            log("  epoch {:>3}/{}: {}".format(epoch + 1, epochs, ", ".join("{}={:.4f}".format(k, v) for k, v in row.items() if k != "epoch")))
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError("non-finite weights in {} after training".format(name))
    report = {**report, "amp_skipped_steps": skipped_steps}
    return model.eval(), history, report


@torch.no_grad()
def predict_raw(
    model: nn.Module, kind: str, samples: Sequence[PoolSample], input_size: int, device: str = "cuda"
) -> tuple[np.ndarray, list[LetterboxTransform]]:
    """Return stacked outputs ``[N,C,G,G]`` (FOMO logits / CenterNet activated) and transforms."""

    model.eval().to(device)
    outputs, transforms = [], []
    for sample in samples:
        letterboxed, transform = letterbox_rgb(_read_rgb(sample.image_path), input_size)
        tensor = torch.from_numpy(
            np.ascontiguousarray(letterboxed.transpose(2, 0, 1), dtype=np.float32) / 255.0
        )[None].to(device)
        outputs.append(model(tensor)[0].float().cpu().numpy())
        transforms.append(transform)
    return np.stack(outputs), transforms
