import random
from typing import Literal

import numpy as np
import torch

PolicyType = Literal["LB", "LD", "SM", "SS"]


class SpecAugment:
    """
    A simple data augmentation method applied to directly to feature inputs
    of a neural network
    it is used in speech recognition tasks to add noise to voice recordings

    Reference: https://arxiv.org/abs/1904.08779
    """

    def __init__(self, policy: PolicyType, zero_mean_normalized: bool = True) -> None:
        """
        Initialize SpecAugment with a given policy.

        Args:
            policy: Augmentation policy, one of:
                - 'LB' (LibriSpeech basic)
                - 'LD' (LibriSpeech double)
                - 'SM' (Switchboard mild)
                - 'SS' (Switchboard strong)
            zero_mean_normalized: Whether input features are zero-mean normalized
        """
        self.policy: PolicyType = policy
        self.zero_mean_normalized: bool = zero_mean_normalized

        self.W: int  # Time warp parameter
        self.F: int  # Frequency mask max size
        self.m_F: int  # Number of frequency masks
        self.T: int  # Time mask max size
        self.p: float  # Probability of applying masking
        self.m_T: int  # Number of time masks
        if self.policy == "LB":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == "LD":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == "SM":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == "SS":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        else:
            raise ValueError(f"Invalid policy: {policy}. Must be one of 'LB', 'LD', 'SM', 'SS'")

    def time_masking(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to spectrogram.

        Args:
            feature: Input spectrogram of shape (time, freq)

        Returns:
            Time-masked spectrogram
        """
        tau = feature.shape[2]
        masked = feature.clone()
        for _ in range(self.m_T):
            t = int(np.random.uniform(0, self.T))
            t0 = random.randint(0, tau - min(tau, t))
            masked[:, :, t0 : t0 + t] = 0
        return masked

    def freq_masking(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.

        Args:
            feature: Input spectrogram of shape (time, freq)

        Returns:
            Frequency-masked spectrogram
        """
        size = feature.shape[1]
        masked = feature.clone()
        for _ in range(self.m_F):
            random.uniform
            f = int(np.random.uniform(0, self.F))
            f0 = random.randint(0, max(0, size - f))
            masked[:, f0 : f0 + f] = 0
        return masked
