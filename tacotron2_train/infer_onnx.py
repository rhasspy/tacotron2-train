#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import jsonlines
import numpy as np
import onnxruntime

from .config import TrainingConfig

_LOGGER = logging.getLogger("tacotron2_train.infer_onnx")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="tacotron2-train.infer_onnx")
    parser.add_argument("model_dir", help="Path to directory with Onnx models")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument(
        "--no-optimizations", action="store_true", help="Disable Onnx optimizations"
    )
    parser.add_argument("--gate-threshold", type=float, default=0.6)
    parser.add_argument("--max-decoder-steps", type=int, default=1000)
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.model_dir = Path(args.model_dir)

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load models
    encoder_path = args.model_dir / "encoder.onnx"
    decoder_path = args.model_dir / "decoder_iter.onnx"
    postnet_path = args.model_dir / "postnet.onnx"

    sess_options = onnxruntime.SessionOptions()
    if args.no_optimizations:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    _LOGGER.debug("Loading encoder from %s", encoder_path)
    encoder = onnxruntime.InferenceSession(str(encoder_path), sess_options=sess_options)

    _LOGGER.debug("Loading decoder from %s", decoder_path)
    decoder_iter = onnxruntime.InferenceSession(
        str(decoder_path), sess_options=sess_options
    )

    _LOGGER.debug("Loading postnet from %s", postnet_path)
    postnet = onnxruntime.InferenceSession(str(postnet_path), sess_options=sess_options)

    _LOGGER.info("Loaded models from %s", args.model_dir)

    # Process input phonemes
    output_obj = {
        "id": "",
        "audio": {
            "filter_length": config.audio.filter_length,
            "hop_length": config.audio.hop_length,
            "win_length": config.audio.win_length,
            "mel_channels": config.audio.n_mel_channels,
            "sample_rate": config.audio.sampling_rate,
            "sample_bytes": config.audio.sample_bytes,
            "channels": config.audio.channels,
            "mel_fmin": config.audio.mel_fmin,
            "mel_fmax": config.audio.mel_fmax,
            "normalized": config.audio.normalized,
        },
        "mel": [],
    }

    start_time = time.perf_counter()

    if os.isatty(sys.stdin.fileno()):
        print("Reading whitespace-separated phoneme ids from stdin...", file=sys.stderr)

    # Read phoneme ids from standard input.
    # Phoneme ids are separated by whitespace (<p1> <p2> ...)
    writer = jsonlines.Writer(sys.stdout, flush=True)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            utt_id = ""
            if args.csv:
                # Input format is id | p1 p2 p3...
                utt_id, line = line.split("|", maxsplit=1)

            # Phoneme ids as p1 p2 p3...
            phoneme_ids = [int(p) for p in line.split()]
            _LOGGER.debug("%s (id=%s)", phoneme_ids, utt_id)

            # Convert to tensors
            # TODO: Allow batches
            text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            text_lengths = np.array([text.shape[1]], dtype=np.int64)

            # Infer mel spectrograms
            start_time = time.perf_counter()
            mel = infer(
                text,
                text_lengths,
                encoder,
                decoder_iter,
                postnet,
                gate_threshold=args.gate_threshold,
                max_decoder_steps=args.max_decoder_steps,
            )
            end_time = time.perf_counter()

            # Write mel spectrogram and settings as a JSON object on one line
            mel_list = mel.squeeze(0).tolist()
            output_obj["id"] = utt_id
            output_obj["mel"] = mel_list

            writer.write(output_obj)

            _LOGGER.debug(
                "Generated mel in %s second(s) (%s, shape=%s)",
                end_time - start_time,
                utt_id,
                list(mel.shape),
            )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


def infer(
    text,
    text_lengths,
    encoder,
    decoder_iter,
    postnet,
    gate_threshold=0.6,
    max_decoder_steps=1000,
):
    # Encoder
    memory, processed_memory, _ = encoder.run(
        None, {"sequences": text, "sequence_lengths": text_lengths}
    )

    # Decoder
    mel_lengths = np.zeros([memory.shape[0]], dtype=np.int32)
    not_finished = np.ones([memory.shape[0]], dtype=np.int32)
    mel_outputs, gate_outputs, alignments = (np.zeros(1), np.zeros(1), np.zeros(1))
    first_iter = True

    (
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    ) = init_decoder_inputs(memory, processed_memory, text_lengths)

    while True:
        (
            mel_output,
            gate_output,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        ) = decoder_iter.run(
            None,
            {
                "decoder_input": decoder_input,
                "attention_hidden": attention_hidden,
                "attention_cell": attention_cell,
                "decoder_hidden": decoder_hidden,
                "decoder_cell": decoder_cell,
                "attention_weights": attention_weights,
                "attention_weights_cum": attention_weights_cum,
                "attention_context": attention_context,
                "memory": memory,
                "processed_memory": processed_memory,
                "mask": mask,
            },
        )

        if first_iter:
            mel_outputs = np.expand_dims(mel_output, 2)
            gate_outputs = np.expand_dims(gate_output, 2)
            alignments = np.expand_dims(attention_weights, 2)
            first_iter = False
        else:
            mel_outputs = np.concatenate(
                (mel_outputs, np.expand_dims(mel_output, 2)), 2
            )
            gate_outputs = np.concatenate(
                (gate_outputs, np.expand_dims(gate_output, 2)), 2
            )
            alignments = np.concatenate(
                (alignments, np.expand_dims(attention_weights, 2)), 2
            )

        dec = (
            np.less_equal(sigmoid(gate_output), gate_threshold)
            .astype(np.int32)
            .squeeze(1)
        )
        not_finished = not_finished * dec
        mel_lengths += not_finished

        if np.sum(not_finished) == 0:
            _LOGGER.debug("Stopping after %s decoder steps(s)", mel_outputs.shape[2])
            break

        if mel_outputs.shape[2] >= max_decoder_steps:
            _LOGGER.warning("Reached max decoder steps (%s)", max_decoder_steps)
            break

        decoder_input = mel_output

    # Postnet
    mel_outputs_postnet = postnet.run(None, {"mel_outputs": mel_outputs})[0]

    return mel_outputs_postnet


# -----------------------------------------------------------------------------


def get_mask_from_lengths(lengths):
    max_len = np.max(lengths)
    ids = np.arange(0, max_len, dtype=lengths.dtype)
    return ids > np.expand_dims(lengths, 1)


def init_decoder_inputs(memory, processed_memory, memory_lengths):

    dtype = memory.dtype
    bs = memory.shape[0]
    seq_len = memory.shape[1]
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = np.zeros((bs, attention_rnn_dim), dtype=dtype)
    attention_cell = np.zeros((bs, attention_rnn_dim), dtype=dtype)
    decoder_hidden = np.zeros((bs, decoder_rnn_dim), dtype=dtype)
    decoder_cell = np.zeros((bs, decoder_rnn_dim), dtype=dtype)
    attention_weights = np.zeros((bs, seq_len), dtype=dtype)
    attention_weights_cum = np.zeros((bs, seq_len), dtype=dtype)
    attention_context = np.zeros((bs, encoder_embedding_dim), dtype=dtype)
    mask = get_mask_from_lengths(memory_lengths)
    decoder_input = np.zeros((bs, n_mel_channels), dtype=dtype)

    return (
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
