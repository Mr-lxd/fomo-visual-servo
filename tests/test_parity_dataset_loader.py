"""Dataset integration tests for the explicit parity-clean label reader."""

from __future__ import annotations

from pathlib import Path

from fomo_servo.datasets import YOLOv5FOMODataset


def test_dataset_uses_explicit_label_loader_for_audited_view(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "yolo_micro"
    root = tmp_path / "dataset"
    for source in fixture_root.rglob("*"):
        destination = root / source.relative_to(fixture_root)
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    calls: list[Path] = []

    def label_loader(path: Path, class_count: int):
        calls.append(path)
        assert class_count == 2
        return ()

    dataset = YOLOv5FOMODataset(
        root=root,
        split="train",
        input_size=96,
        stride=8,
        class_mode="preserve",
        label_loader=label_loader,
    )

    sample = dataset[0]

    assert calls == [sample.label_path]
    assert sample.original_boxes == ()
