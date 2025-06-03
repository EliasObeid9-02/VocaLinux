import random

import numpy as np
from spec_augment.image_wrap import sparse_image_warp


class SpecAugment:
    """
    A simple data aumentation method applied to directly to feature inputs
    of a neural network
    it is used in speech recognition tasks to add noise to voice recordings
    """

    def __init__(self, policy, zero_mean_normalized=True):
        """
        policy: one of the following
        'LB' (LibriSpeech basic)
        'LD' (LibriSpeech double)
        'SM' (Switchboard mild)
        'SS' (Switchboard strong)
        zero_mean_normalized: optional parameter to normalize features
        """
        self.policy = policy
        self.zero_mean_normalized = zero_mean_normalized

        """
        Policy Specific Parameters
        W:   time warp
        F:   frequency mask max size
        m_F: number of frequency masks
        T:   time mask max size
        p:   probability of applying masking
        m_T: number of time masks
        """
        if self.policy == "LB":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == "LD":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == "SM":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == "SS":
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2

    def time_warping(self, feature):
        # Reshape to [Batch_size, time, freq, 1] for sparse_image_warp func.
        feature = np.reshape(feature, (-1, feature.shape[0], feature.shape[1], 1))
        v, tau = feature.shape[1], feature.shape[2]
        horiz_line_thru_ctr = feature[0][v // 2]

        # Random point along the horizontal/time axis
        random_pt = horiz_line_thru_ctr[random.randrange(self.W, tau - self.W)]
        w = np.random.uniform((-self.W), self.W)  # distance

        src_points = [[[v // 2, random_pt[0]]]]
        dest_points = [[[v // 2, random_pt[0] + w]]]

        feature, _ = sparse_image_warp(feature, src_points, dest_points, num_boundaries_points=2)
        return feature

    def time_masking(self, feature):
        tau = feature.shape[2]  # Time frames
        # Apply m_T time masks to the Mel Spectrogram
        for _ in range(self.m_T):
            t = int(np.random.uniform(0, self.T))  # [0, T)
            upper = tau if t > tau else t  # make limitation
            t0 = random.randint(0, tau - upper)  # [0, tau - t)
            feature[:, :, t0 : t0 + t] = 0
        return feature

    def freq_masking(self, feature):
        size = feature.shape[1]
        for _ in range(self.m_F):
            f = int(np.random.uniform(0, self.F))  # [0, F)
            f0 = random.randint(0, size - f)  # [0, v - f)
            feature[:, f0 : f0 + f] = 0
        return feature
