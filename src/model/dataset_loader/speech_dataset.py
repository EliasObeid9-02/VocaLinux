import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import librosa
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from config import CFG, EOS_TOKEN, SOS_TOKEN
from spec_augment.method import PolicyType, SpecAugment
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """
    LibriSpeech dataset
    """

    dataset: torchaudio.datasets.LIBRISPEECH
    char2id: Dict[str, int]
    max_len: int
    augmentation: bool
    spec_augment: SpecAugment
    pkwargs: Dict[str, Any]
    current: Dict[str, Union[torch.Tensor, str]]
    mel_converter: torchaudio.transforms.MelSpectrogram
    db_converter: torchaudio.transforms.AmplitudeToDB

    def __init__(
        self,
        char2id: Dict[str, int],
        split: str = CFG.dataset_list[4],
        max_len: int = 0,
        augmentation: bool = False,
        pkwargs: Optional[dict] = None,
    ) -> None:
        """
        char2id:        mapping from charcters to ID
        split:          which dataset split to use
        max_len:        maximum padding length
        augmentation:   whether to apply padding using SpecAugment
        pkwargs:        optional parameters for preprocessing
        """
        super(SpeechDataset, self).__init__()
        self.dataset: torchaudio.datasets.LIBRISPEECH = torchaudio.datasets.LIBRISPEECH(
            root=CFG.dataset_root,
            url=split,
        )
        self.char2id = char2id
        self.max_len = max_len
        self.augmentation = augmentation
        self.spec_augment = SpecAugment("LB")
        self.pkwargs = pkwargs or {}
        self.current = {}

        self.mel_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.sr, n_fft=CFG.n_fft, hop_length=CFG.hop_length, n_mels=CFG.n_mels
        )
        self.db_converter = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns (spectrogram, label) tuple at index 'idx' in dataset
        """
        if torch.is_tensor(idx):
            idx_list: List[int] = idx.tolist()
            idx = idx_list[0] if isinstance(idx_list, list) else idx_list

        waveform: torch.Tensor
        utterance: str
        waveform, _, utterance, _, _, _ = self.dataset[idx]

        mel_spec: torch.Tensor = self.mel_converter(waveform)
        log_mel: torch.Tensor = self.db_converter(mel_spec)

        if self.augmentation:
            rand: int = random.randint(1, 10)
            if rand > 3:
                log_mel = self.spec_augment.freq_masking(log_mel)
                if rand < 7:
                    log_mel = self.spec_augment.time_masking(log_mel)
            else:
                log_mel = self.spec_augment.time_masking(log_mel)

        self.current = {"input": log_mel, "label": utterance}

        processed_spec: torch.Tensor = log_mel[0, :, :].squeeze(1).t()
        if self.max_len > 0:
            pad_amount: int = max(0, self.max_len - processed_spec.shape[1])
            processed_spec = torch.nn.functional.pad(processed_spec, (0, pad_amount), "constant", 0)

        labels: List[int] = [SOS_TOKEN]
        for char in utterance:
            labels.append(self.char2id.get(char, self.char2id["<unk>"]))
        labels.append(EOS_TOKEN)
        return (processed_spec, np.array(labels))

    def check_log_mel_spec(self, itemIndex: int) -> None:
        self.__getitem__(itemIndex)
        input: torch.Tensor = cast(torch.Tensor, self.current.get("input"))
        print(f"Spectrogram shape: {input.shape}")
        self.plot_spectrogram(
            input[0],
            title="LogMelSpectrogram",
            ylabel="mel freq",
        )
        label: str = cast(str, self.current.get("label"))
        print(f"Sample transcript: {label}")

    @staticmethod
    def plot_spectrogram(
        specgram: torch.Tensor, title: Optional[str] = None, ylabel: str = "freq_bin"
    ) -> None:
        fig: matplotlib.figure.FigureBase
        ax: matplotlib.axes.Axes

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))  # type: ignore
        ax = cast(matplotlib.axes.Axes, ax)  # Explicit cast

        im = ax.imshow(librosa.power_to_db(specgram.numpy()), origin="lower", aspect="auto")

        ax.set_title(title or "Spectrogram")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("frame")
        fig.colorbar(im, ax=ax)
        plt.show(block=False)


def _collate_fn(batch: List[Tuple[torch.Tensor, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that pads variable-length sequences to the maximum length in the batch.

    Args:
        batch: A list of tuples where each tuple contains:
            - [0]: Input sequence tensor of shape (seq_len, feat_dim)
            - [1]: Target sequence as a list of integers

    Returns:
        A tuple containing:
        - Padded input sequences tensor of shape (batch_size, max_seq_len, feat_dim)
        - Padded target sequences tensor of shape (batch_size, max_target_len)

    Example:
        >>> batch = [(torch.rand(10, 40), [1,2,3]),
        ...          (torch.rand(15, 40), [4,5])]
        >>> inputs, targets = _collate_fn(batch)
        >>> inputs.shape
        torch.Size([2, 15, 40])
        >>> targets.shape
        torch.Size([2, 3])
    """

    def _seq_length(p: Tuple[torch.Tensor, List[int]]) -> int:
        """
        Return the sequence length (time dimension) of a sample
        """
        return p[0].size(0)

    def _target_length(p: Tuple[torch.Tensor, List[int]]) -> int:
        """
        Return the target sequence length of a sample
        """
        return len(p[1])

    # Find samples with maximum length in batch
    max_seq_sample: torch.Tensor = max(batch, key=_seq_length)[0]
    max_target_sample: List[int] = max(batch, key=_target_length)[1]

    max_seq_size: int = max_seq_sample.size(0)
    max_target_size: int = len(max_target_sample)
    feat_size: int = max_seq_sample.size(1)
    batch_size: int = len(batch)

    seqs: torch.Tensor = torch.zeros((batch_size, max_seq_size, feat_size))
    targets: torch.Tensor = torch.zeros(batch_size, max_target_size, dtype=torch.long)

    for x in range(batch_size):
        tensor, target = batch[x]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets
