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


def test_load_config_reads_inert_checkpoint_selection_v2_defaults(tmp_path: Path) -> None:
    """Existing YAML must receive disabled snapshots and deterministic v2 defaults."""

    _, load_config = _config_api()
    assert callable(load_config)
    config_path = tmp_path / "v2_defaults.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.epoch_snapshots.enabled is False
    assert config.training.epoch_snapshots.format == "weights_only"
    assert config.training.epoch_snapshots.interval == 1
    assert config.training.epoch_snapshots.keep_last is None
    assert config.evaluation.checkpoint_selection.metric == "centroid_pr_auc_macro"
    assert config.evaluation.checkpoint_selection.split == config.dataset.validation_split
    assert config.evaluation.checkpoint_selection.threshold_grid == config.evaluation.threshold_sweep
    assert config.evaluation.threshold_calibration.enabled is False
    assert config.evaluation.threshold_calibration.fallback_threshold == pytest.approx(0.5)


def test_load_config_accepts_explicit_train_only_protocol(tmp_path: Path) -> None:
    _, load_config = _config_api()
    config_path = tmp_path / "train_only.yaml"
    config_path.write_text(
        """
dataset:
  root: data/lab_pool
  validation_split: null
  classes: [fish, jellyfish]
model:
  input_size: 192
  output_stride: 8
training:
  checkpoint_policy: fixed_final_epoch
  early_stopping_patience: 0
evaluation:
  threshold_sweep:
    enabled: false
  threshold_calibration:
    enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.validation_split is None
    assert config.training.checkpoint_policy == "fixed_final_epoch"
    assert config.evaluation.threshold_sweep_enabled is False


def test_load_config_reads_strict_weights_initialization(tmp_path: Path) -> None:
    _, load_config = _config_api()
    config_path = tmp_path / "initialize.yaml"
    config_path.write_text(
        """
dataset:
  root: data/lab_pool
  classes: [fish]
model:
  input_size: 96
  output_stride: 8
training:
  initialize_from: weights/epoch_040_weights.pt
  initialize_sha256: e8c242f4af2b87b70fea2a516352f28e70bf438161eeb7d092231ed46c976a1d
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.initialize_from == Path("weights/epoch_040_weights.pt")
    assert config.training.initialize_sha256 == (
        "e8c242f4af2b87b70fea2a516352f28e70bf438161eeb7d092231ed46c976a1d"
    )


@pytest.mark.parametrize(
    ("training_yaml", "message"),
    [
        (
            "initialize_from: weights/init.pt\n  initialize_sha256: not-a-sha",
            "training.initialize_sha256",
        ),
        (
            "initialize_from: weights/init.pt\n  initialize_sha256: " + "a" * 64 + "\n  resume: outputs/last.pt",
            "mutually exclusive",
        ),
        (
            "checkpoint_policy: validation_best\n  early_stopping_patience: 0",
            "fixed_final_epoch",
        ),
        (
            "checkpoint_policy: fixed_final_epoch\n  early_stopping_patience: 2",
            "early stopping",
        ),
    ],
)
def test_load_config_rejects_invalid_train_only_or_initialization_protocol(
    tmp_path: Path, training_yaml: str, message: str
) -> None:
    configuration_error, load_config = _config_api()
    validation = "null" if "checkpoint_policy" in training_yaml else "val"
    config_path = tmp_path / "invalid_protocol.yaml"
    config_path.write_text(
        """
dataset:
  root: data/lab_pool
  validation_split: {validation}
  classes: [fish]
model:
  input_size: 96
  output_stride: 8
training:
  {training_yaml}
evaluation:
  threshold_sweep:
    enabled: false
""".format(validation=validation, training_yaml=training_yaml).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(configuration_error, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("yaml_fragment", "message"),
    [
        ("format: full", "training.epoch_snapshots.format"),
        ("interval: 0", "training.epoch_snapshots.interval"),
        ("keep_last: 0", "training.epoch_snapshots.keep_last"),
    ],
)
def test_load_config_rejects_invalid_epoch_snapshot_settings(
    tmp_path: Path, yaml_fragment: str, message: str
) -> None:
    """Weights-only snapshots have a deliberately small, strict schema."""

    configuration_error, load_config = _config_api()
    assert callable(load_config)
    config_path = tmp_path / "invalid_snapshots.yaml"
    config_path.write_text(
        """
dataset:
  root: data/aquarium_creature
  classes: [creature]
model:
  input_size: 96
  output_stride: 8
training:
  epoch_snapshots:
    enabled: true
    {yaml_fragment}
""".format(yaml_fragment=yaml_fragment).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(configuration_error, match=message):
        load_config(config_path)


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


def test_lab_pool_engineering_config_locks_approved_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, load_config = _config_api()
    dataset_root = tmp_path / "lab_pool_v1_d2_trainonly"
    checkpoint = tmp_path / "epoch_040_weights.pt"
    monkeypatch.setenv("FOMO_LAB_POOL_TRAIN_ROOT", str(dataset_root))
    monkeypatch.setenv("FOMO_D2_EPOCH40_WEIGHTS", str(checkpoint))
    root = Path(__file__).resolve().parents[1]

    config = load_config(
        root / "configs/engineering/lab_pool_adaptation_seed42_e20.yaml"
    )

    assert config.dataset.root == dataset_root
    assert config.dataset.validation_split is None
    assert config.dataset.class_names == (
        "fish",
        "jellyfish",
        "penguin",
        "puffin",
        "shark",
        "starfish",
        "stingray",
    )
    assert config.dataset.class_mode == "preserve"
    assert config.model.backbone == "mobilenet_v2_fomo"
    assert config.model.width_multiplier == pytest.approx(0.35)
    assert config.model.head_channels == 32
    assert config.model.input_size == 192
    assert config.model.output_stride == 8
    assert config.model.pretrained is False
    assert config.training.initialize_from == checkpoint
    assert config.training.initialize_sha256 == (
        "e8c242f4af2b87b70fea2a516352f28e70bf438161eeb7d092231ed46c976a1d"
    )
    assert config.training.checkpoint_policy == "fixed_final_epoch"
    assert config.training.batch_size == 8
    assert config.training.epochs == 20
    assert config.training.seed == 42
    assert config.training.early_stopping_patience == 0
    assert config.training.epoch_snapshots.enabled is True
    assert config.training.epoch_snapshots.interval == 20
    assert config.training.epoch_snapshots.keep_last == 1
    assert config.training.optimizer.name == "adamw"
    assert config.training.optimizer.learning_rate == pytest.approx(0.0001)
    assert config.training.optimizer.weight_decay == pytest.approx(0.0001)
    assert config.training.scheduler.name == "none"
    assert config.augmentation.preset == "underwater_conservative"
    assert config.loss.name == "ei_weighted_xent_legacy"
    assert config.loss.background_weight == pytest.approx(1.0)
    assert config.loss.object_weight == pytest.approx(100.0)
    assert config.evaluation.threshold_sweep_enabled is False
    assert config.evaluation.threshold_calibration.enabled is False
    assert config.postprocess.inference_threshold == pytest.approx(0.40)
    assert config.evaluation.checkpoint_threshold == pytest.approx(0.40)


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
