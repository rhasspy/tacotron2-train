import logging
import time
import typing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint, save_checkpoint
from .config import TrainingConfig
from .loss_function import LossType
from .models import ModelType, OptimizerType, setup_model
from .utils import to_gpu

try:
    from apex import amp
except ImportError:
    # apex not available (no fp16)
    amp = None


_LOGGER = logging.getLogger("tacotron2_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[OptimizerType] = None,
    global_step: int = 1,
    checkpoint_epochs: int = 1,
    learning_rate: float = 1e-3,
):
    """Run training for the specified number of epochs"""
    torch.manual_seed(config.seed)

    model, optimizer = setup_model(config, model=model, optimizer=optimizer)

    amp_run = False
    if config.fp16_run and amp:
        # Use AMP for FP16 run
        amp_run = True
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    assert model is not None
    assert optimizer is not None

    criterion = LossType(
        ga_alpha=config.model.guided_attention_alpha,
        ga_sigma=config.model.guided_attention_sigma,
    )
    criterion.cuda()

    # Begin training
    for epoch in range(1, config.epochs + 1):
        _LOGGER.debug(
            "Begin epoch %s/%s (global step=%s)", epoch, config.epochs, global_step
        )
        epoch_start_time = time.perf_counter()
        global_step, learning_rate = train_step(
            global_step,
            epoch,
            learning_rate,
            config,
            model,
            optimizer,
            criterion,
            train_loader,
            amp_run,
        )

        if (epoch % checkpoint_epochs) == 0:
            # Save checkpoint
            checkpoint_path = model_dir / f"checkpoint_{global_step}.pth"
            _LOGGER.debug("Saving checkpoint to %s", checkpoint_path)
            save_checkpoint(
                Checkpoint(
                    model=model,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    version=config.version,
                ),
                checkpoint_path,
            )
            _LOGGER.info("Saved checkpoint to %s", checkpoint_path)

        epoch_end_time = time.perf_counter()
        _LOGGER.debug(
            "Epoch %s complete in %s second(s) (global step=%s)",
            epoch,
            epoch_end_time - epoch_start_time,
            global_step,
        )


def train_step(
    global_step: int,
    epoch: int,
    learning_rate: float,
    config: TrainingConfig,
    model: ModelType,
    optimizer: OptimizerType,
    criterion: LossType,
    train_loader: DataLoader,
    amp_run: bool,
) -> typing.Tuple[int, float]:
    steps_per_epoch = len(train_loader)

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        learning_rate = adjust_learning_rate(global_step, epoch, optimizer, config)

        model.zero_grad()
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            _len_x,
        ) = batch

        # Put on GPU
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(
            text_padded, input_lengths, mel_padded, output_lengths
        )

        loss = criterion(
            mel_outputs,
            mel_outputs_postnet,
            gate_outputs,
            mel_padded,
            gate_padded,
            alignments,
            input_lengths,
            output_lengths,
        )

        reduced_loss = loss.item()

        if np.isnan(reduced_loss):
            raise RuntimeError(f"loss is NaN at global step {global_step}")

        if amp_run:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), config.grad_clip_threshold
            )
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_threshold
            )

        optimizer.step()

        _LOGGER.debug(
            "Loss: %s (step=%s/%s)", reduced_loss, batch_idx + 1, steps_per_epoch
        )
        global_step += 1

    return global_step, learning_rate


# -----------------------------------------------------------------------------


def adjust_learning_rate(
    iteration: int, epoch: int, optimizer: OptimizerType, config: TrainingConfig
) -> float:

    p = 0
    if config.anneal_steps is not None:
        for a_step in config.anneal_steps:
            if epoch >= int(a_step):
                p = p + 1

    if config.anneal_factor == 0.3:
        lr = config.learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = config.learning_rate * (config.anneal_factor ** p)

    # if optimizer.param_groups[0]["lr"] != lr:
    #     DLLogger.log(
    #         step=(epoch, iteration),
    #         data={
    #             "learning_rate changed": str(optimizer.param_groups[0]["lr"])
    #             + " -> "
    #             + str(lr)
    #         },
    #     )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
