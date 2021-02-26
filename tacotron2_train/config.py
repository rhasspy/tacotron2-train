"""Configuration classes"""
import collections
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import DataClassJsonMixin


@dataclass
class AudioConfig(DataClassJsonMixin):
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    sampling_rate: int = 22050
    sample_bytes: int = 2
    channels: int = 1
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    normalized: bool = True


@dataclass
class ModelConfig(DataClassJsonMixin):
    # Symbols
    n_symbols: int = 0
    symbols_embedding_dim: int = 512
    mask_padding: bool = False

    # Encoding
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512

    # Decoder
    n_frames_per_step: int = 5  # 1
    decoder_rnn_dim: int = 256  # 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 2000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1
    decoder_no_early_stopping: bool = False

    # Attention
    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    # Guided attention
    guided_attention_alpha: float = 5.0
    guided_attention_sigma: float = 0.4

    # Postnet
    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5


@dataclass
class TrainingConfig(DataClassJsonMixin):
    seed: int = 1234
    epochs: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_threshold: float = 1.0
    grad_clip: float = 5.0
    anneal_steps: typing.Optional[typing.Tuple[int, ...]] = None
    anneal_factor: float = 0.1  # choices: 0.1, 0.3
    dynamic_loss_scaling: bool = True
    disable_uniform_initialize_bn_weight: bool = False
    batch_size: int = 32
    fp16_run: bool = True
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    version: int = 1

    def save(self, config_file: typing.TextIO):
        """Save config as JSON to a file"""
        json.dump(self.to_dict(), config_file, indent=4)

    @staticmethod
    def load(config_file: typing.TextIO) -> "TrainingConfig":
        """Load config from a JSON file"""
        return TrainingConfig.from_json(config_file.read())

    @staticmethod
    def load_and_merge(
        config: "TrainingConfig",
        config_files: typing.Iterable[typing.Union[str, Path, typing.TextIO]],
    ) -> "TrainingConfig":
        """Loads one or more JSON configuration files and overlays them on top of an existing config"""
        base_dict = config.to_dict()
        for maybe_config_file in config_files:
            if isinstance(maybe_config_file, (str, Path)):
                # File path
                config_file = open(maybe_config_file, "r")
            else:
                # File object
                config_file = maybe_config_file

            with config_file:
                # Load new config and overlay on existing config
                new_dict = json.load(config_file)
                TrainingConfig.recursive_update(base_dict, new_dict)

        return TrainingConfig.from_dict(base_dict)

    @staticmethod
    def recursive_update(
        base_dict: typing.Dict[typing.Any, typing.Any],
        new_dict: typing.Mapping[typing.Any, typing.Any],
    ) -> None:
        """Recursively overwrites values in base dictionary with values from new dictionary"""
        for k, v in new_dict.items():
            if isinstance(v, collections.Mapping) and (base_dict.get(k) is not None):
                TrainingConfig.recursive_update(base_dict[k], v)
            else:
                base_dict[k] = v
