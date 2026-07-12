from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _config_api():
    try:
        module = importlib.import_module("fomo_servo.config")
    except ModuleNotFoundError:
        return None, None
    return getattr(module, "ConfigurationError", None), getattr(module, "load_config", None)


def test_load_config_reads_yaml_and_derives_fomo_contract(tmp_path: Path) -> None:
    configuration_error, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"

    config_path = tmp_path / "aquarium.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  train_split: train
  validation_split: val
  classes: [creature]
  collision_policy: keep_first
model:
  input_size: 192
  output_stride: 8
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.root == Path("data/aquarium_creature")
    assert config.dataset.class_names == ("creature",)
    assert config.dataset.collision_policy == "keep_first"
    assert config.model.input_size == 192
    assert config.model.output_stride == 8
    assert config.grid_size == 24
    assert config.output_channels == 2


def test_load_config_expands_dataset_root_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"

    dataset_root = tmp_path / "aquarium_pretrain"
    monkeypatch.setenv("FOMO_DATASET_ROOT", str(dataset_root))
    config_path = tmp_path / "environment_path.yaml"
    config_path.write_text(
        """
dataset:
  root: ${FOMO_DATASET_ROOT}
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.root == dataset_root


def test_load_config_reads_model_and_training_runtime_fields(tmp_path: Path) -> None:
    configuration_error, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"

    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  backbone: mobilenet_v2_lite
  width_multiplier: 0.35
  head_channels: 32
  input_size: 192
  output_stride: 8
training:
  device: auto
  amp: true
  amp_initial_scale: 256.0
  num_workers: 4
  pin_memory: true
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert hasattr(config.model, "backbone"), "model.backbone must be available"
    assert hasattr(
        config.model, "width_multiplier"
    ), "model.width_multiplier must be available"
    assert hasattr(config.model, "head_channels"), "model.head_channels must be available"
    assert hasattr(config, "training"), "training configuration must be available"
    assert config.model.backbone == "mobilenet_v2_lite"
    assert config.model.width_multiplier == pytest.approx(0.35)
    assert config.model.head_channels == 32
    assert config.training.device == "auto"
    assert hasattr(config.training, "amp"), "training.amp must be available"
    assert hasattr(
        config.training, "amp_initial_scale"
    ), "training.amp_initial_scale must be available"
    assert hasattr(
        config.training, "num_workers"
    ), "training.num_workers must be available"
    assert hasattr(
        config.training, "pin_memory"
    ), "training.pin_memory must be available"
    assert config.training.amp is True
    assert config.training.amp_initial_scale == pytest.approx(256.0)
    assert config.training.num_workers == 4
    assert config.training.pin_memory is True
    assert hasattr(config.training, "epochs"), "training.epochs must be available"
    assert config.training.checkpoint_criterion == "grid_f1"
    assert hasattr(config, "loss"), "loss configuration must be available"
    assert config.postprocess.component_mode == "connected_components"
    assert config.evaluation.matching_mode == "centroid_in_bbox"
    assert config.loss.class_weight_mode == "manual"
    assert config.loss.class_weights == (1.0, 1.0)


def test_load_config_accepts_uppercase_train_alias(tmp_path: Path) -> None:
    configuration_error, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"

    config_path = tmp_path / "uppercase_train.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
TRAIN:
  DEVICE: cpu
  AMP: false
  NUM_WORKERS: 0
  PIN_MEMORY: false
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.device == "cpu"
    assert config.training.amp is False
    assert config.training.num_workers == 0
    assert config.training.pin_memory is False


def test_load_config_reads_automatic_loss_class_weight_settings(tmp_path: Path) -> None:
    """Auto weights are configured in YAML and resolved only from the train split."""

    _, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"
    config_path = tmp_path / "auto_weights.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  classes: [fish, crab]
model:
  input_size: 96
  output_stride: 8
loss:
  name: focal_cross_entropy
  gamma: 2.0
  class_weights:
    mode: auto
    background_weight: 1.0
    foreground_base_weight: 25.0
    class_balance: sqrt_inverse_frequency
    min_foreground_weight: 12.5
    max_foreground_weight: 75.0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.loss.class_weight_mode == "auto"
    assert config.loss.class_weights is None
    assert config.loss.background_weight == pytest.approx(1.0)
    assert config.loss.foreground_base_weight == pytest.approx(25.0)
    assert config.loss.class_balance == "sqrt_inverse_frequency"
    assert config.loss.min_foreground_weight == pytest.approx(12.5)
    assert config.loss.max_foreground_weight == pytest.approx(75.0)


def test_aug00_none_config_is_fixed_no_augmentation_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline config fixes all training controls and uses manual weight 4."""

    _, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"
    monkeypatch.setenv("FOMO_DATASET_ROOT", "data/aquarium_pretrain")

    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "experiments" / "aug00_none.yaml"
    )

    assert config.experiment.name == "aug00_none"
    assert config.training.epochs == 60
    assert config.training.early_stopping_patience == 0
    assert config.loss.class_weight_mode == "manual"
    assert config.loss.class_weights == (1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    assert config.model.input_size == 192
    assert config.model.output_stride == 8
    assert config.dataset.train_split == "train"
    assert config.dataset.validation_split == "val"
    assert config.postprocess.inference_threshold == pytest.approx(0.5)
    assert config.evaluation.checkpoint_threshold == pytest.approx(0.5)
    assert config.evaluation.threshold_sweep_enabled is True


def test_aug00_none_locked_config_keeps_the_same_fixed_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The locked baseline has no augmentation and separates both thresholds."""

    _, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"
    monkeypatch.setenv("FOMO_DATASET_ROOT", "data/aquarium_pretrain")

    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "aug00_none_locked.yaml"
    )

    assert config.experiment.name == "aug00_none_locked"
    assert config.training.output_dir.as_posix().endswith(
        "outputs/experiments/aug00_none_locked"
    )
    assert config.training.epochs == 60
    assert config.training.early_stopping_patience == 0
    assert config.loss.class_weights == (1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    assert config.postprocess.inference_threshold == pytest.approx(0.5)
    assert config.evaluation.checkpoint_threshold == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 190
  output_stride: 8
""".lstrip(),
            "model.input_size must be divisible by model.output_stride",
        ),
        (
            """
dataset:
  root: data/aquarium_creature
  classes: []
model:
  input_size: 192
  output_stride: 8
""".lstrip(),
            "dataset.classes must contain at least one class name",
        ),
        (
            """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
training:
  device: auto
TRAIN:
  DEVICE: cpu
""".lstrip(),
            "not both",
        ),
        (
            """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
training:
  num_workers: -1
""".lstrip(),
            "training.num_workers must be a non-negative integer",
        ),
        (
            """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
loss:
  name: weighted_cross_entropy
  class_weights: [1.0]
""".lstrip(),
            "loss.class_weights must contain 2 values",
        ),
    ],
)
def test_load_config_reports_validation_errors(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    configuration_error, load_config = _config_api()
    assert callable(load_config), "fomo_servo.config.load_config must be available"
    assert isinstance(configuration_error, type), "ConfigurationError must be available"

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(configuration_error, match=message):
        load_config(config_path)
