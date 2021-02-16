"""Configuration classes"""
import json
import typing
from dataclasses import dataclass, field

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
    n_frames_per_step: int = 1
    decoder_rnn_dim: int = 1024
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
