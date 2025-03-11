import numbers
import random, torch.nn as nn
import warnings

import torch
import numpy as np
# from julius.filters import highpass_filter, lowpass_filter
from scipy.interpolate import interp1d
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, asdict
from .constants import MIMIC_IV_MEAN, MIMIC_IV_STD
from .augmentations import BaselineWander, RandomMasking, CutMix


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        for i in range(len(self.mean)):
            x[:, i, :] = (x[:, i, :] - self.mean[i]) / self.std[i]
        return x


class Resize(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, length = x.shape
        if length < self.seq_length:
            new_x = torch.zeros((b, c, self.seq_length))
            new_x[:, :, 0:length] = x
        elif length > self.seq_length:
            new_x = x[:, :, 0:self.seq_length]
        else:
            new_x = x
        return new_x


class Compose:
    """
    Data augmentation module that transforms any
    given data example with a chain of augmentations.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    Note:
        In order to script the transformation, please use ``torch.nn.ModuleList``
        as input instead of list/tuple of transforms as shown below:

        Make sure to use only scriptable transformations,
        i.e. that work with ``torch.Tensor``.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, ecg):
        if self.p < torch.rand(1):
            return ecg
        for t in self.transforms:
            ecg = t(ecg)
        return ecg

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@dataclass
class PreprocessCfg:
    seq_length: int = 5000
    duration: int = 10
    sampling_rate: int = 500
    dataset: str = None
    mean: Tuple[float, ...] = None
    std: Tuple[float, ...] = None
    resize_mode: str = 'shortest'

    @property
    def num_channels(self):
        return 12

    @property
    def input_size(self):
        return self.num_channels, self.seq_length


_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())


def merge_preprocess_dict(
        base: Union[PreprocessCfg, Dict],
        overlay: Dict,
):
    """ Merge overlay key-value pairs on top of base preprocess cfg or dict.
    Input dicts are filtered based on PreprocessCfg fields.
    """
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    return base_clean


def merge_preprocess_kwargs(base: PreprocessCfg, **kwargs):
    return merge_preprocess_dict(base, kwargs)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    dur: Optional[Tuple[float, float]] = 10
    sr: Optional[int] = 500


def _setup_size(size, error_msg):
    return (12, size)


def ecg_transform(
        ecg_size: Tuple[int, int],
        is_train: bool,
        resize_mode: Optional[str] = None,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    if mean is not None:
        normalize = Normalize(mean=mean, std=std)
    else:
        normalize = Normalize(mean=MIMIC_IV_MEAN, std=MIMIC_IV_STD)
    resize = Resize(seq_length=ecg_size[1])
    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    dur = aug_cfg.dur
    sr = aug_cfg.sr
    if is_train:
        train_transform = [
            RandomApply([BaselineWander(fs=sr), ], p=0.5),
            RandomApply([CutMix(fs=sr)], p=0.5),
            RandomApply([RandomMasking(fs=sr)], p=0.3)
        ]
        train_transform.extend([
            normalize,
            resize
        ])
        train_transform = Compose(train_transform)
        return train_transform
    else:
        transforms = []

        transforms.extend([
            normalize,
            resize
        ])
        return Compose(transforms)


def ecg_transform_v2(
        cfg: PreprocessCfg,
        is_train: bool,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    return ecg_transform(
        ecg_size=(cfg.num_channels, cfg.seq_length),
        is_train=is_train,
        mean=cfg.mean,
        std=cfg.std,
        resize_mode=cfg.resize_mode,
        aug_cfg=aug_cfg,
    )
