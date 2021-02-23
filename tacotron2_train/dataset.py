"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import json
import logging
import random
import typing
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

_LOGGER = logging.getLogger("tacotron2_train.dataset")

# -----------------------------------------------------------------------------


class PhonemeMelLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        id_phonemes: typing.Dict[str, torch.IntTensor],
        id_mels: typing.Dict[str, torch.FloatTensor],
        mels_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        self.id_phonemes = id_phonemes
        self.id_mels = id_mels
        self.mels_dir = Path(mels_dir) if mels_dir else None

        if self.id_mels:
            self.ids = list(
                set.intersection(set(id_phonemes.keys()), set(id_mels.keys()))
            )
            assert self.ids, "No shared utterance ids between phonemes and mels"
        else:
            # Assume all ids will be present in mels_dir
            self.ids = list(id_phonemes.keys())

        random.shuffle(self.ids)

    def __getitem__(self, index):
        utt_id = self.ids[index]
        text = self.id_phonemes[utt_id]
        mel = self.id_mels.get(utt_id)

        if mel is None:
            assert self.mels_dir, f"Missing mel for id {utt_id}, but no mels_dir"
            mel_path = self.mels_dir / (utt_id + ".npy")

            # TODO: Verify shape
            mel = torch.FloatTensor(np.load(mel_path, allow_pickle=True))

            # Cache mel
            self.id_mels[utt_id] = mel

        # phonemes, mels, lengths
        return (text, mel, len(text))

    def __len__(self):
        return len(self.ids)


class PhonemeMelCollate:
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
        )


# -----------------------------------------------------------------------------


def load_phonemes(csv_file: typing.TextIO) -> typing.Dict[str, torch.IntTensor]:
    phonemes = {}
    reader = csv.reader(csv_file, delimiter="|")
    for row in reader:
        utt_id, phoneme_str = row[0], row[1]
        phoneme_ids = [int(p) for p in phoneme_str.strip().split()]

        if not phoneme_ids:
            _LOGGER.warning("No phonemes for %s, dropping utterance", utt_id)
            continue

        phonemes[utt_id] = torch.IntTensor(phoneme_ids)

    return phonemes


def load_mels(jsonl_file: typing.TextIO) -> typing.Dict[str, torch.FloatTensor]:
    mels = {}
    for line in jsonl_file:
        line = line.strip()
        if not line:
            continue

        mel_obj = json.loads(line)
        utt_id = mel_obj["id"]
        mels[utt_id] = torch.FloatTensor(mel_obj["mel"])

    return mels
