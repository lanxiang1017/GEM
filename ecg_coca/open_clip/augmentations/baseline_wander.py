import numpy as np
import torch


class BaselineWander(torch.nn.Module):
    def __init__(
            self,
            max_amplitude=0.5,
            min_amplitude=0,
            p=1.0,
            max_freq=0.2,
            min_freq=0.01,
            k=3,
            fs=500,
            **kwargs,
    ):
        super(BaselineWander, self).__init__()
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.k = k
        self.freq = fs
        self.p = p

    def forward(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0, 1):
            batch, csz, tsz = new_sample.shape
            amp_channel = np.random.normal(1, 0.5, size=(csz, 1))
            c = np.array([i for i in range(12)])
            amp_general = np.random.uniform(self.min_amplitude, self.max_amplitude, size=self.k)
            noise = np.zeros(shape=(1, tsz))
            for k in range(self.k):
                noise += self._apply_baseline_wander(tsz) * amp_general[k]
            noise = (noise * amp_channel).astype(np.float32)
            new_sample[:, c, :] = new_sample[:, c, :] + noise[c, :]
        return new_sample.float()

    def _apply_baseline_wander(self, tsz):
        f = np.random.uniform(self.min_freq, self.max_freq)
        t = np.linspace(0, tsz - 1, tsz)
        r = np.random.uniform(0, 2 * np.pi)
        noise = np.cos(2 * np.pi * f * (t / self.freq) + r)
        return noise