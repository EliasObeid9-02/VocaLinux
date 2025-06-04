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
from spec_augment.method import SpecAugment
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
