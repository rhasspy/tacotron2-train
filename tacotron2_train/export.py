#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("tacotron2_train.export")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="tacotron2-train.export")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("output", help="Path to output model (.pt)")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load checkpoint
    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, config, use_cuda=False)
    model = checkpoint.model

    _LOGGER.info(
        "Loaded checkpoint from %s (global step=%s)",
        args.checkpoint,
        checkpoint.global_step,
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Inference only
    model.forward = model.infer

    jitted_model = torch.jit.script(model)
    torch.jit.save(jitted_model, str(args.output))

    _LOGGER.info("Saved TorchScript model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
