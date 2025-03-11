"""
"""

from numbers import Real
from random import randint
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def get_indices(prob: float, pop_size: int, scale_ratio: float = 0.1) -> List[int]:
    """Get a list of indices to be selected.

    A random list of indices in the range ``[0, pop_size-1]``
    is generated, with the probability of each index to be selected.

    Parameters
    ----------
    prob : float
        The probability of each index to be selected.
    pop_size : int
        The size of the population.
    scale_ratio : float, default 0.1
        Scale ratio of std of the normal distribution to the population size.

    Returns
    -------
    indices : List[int],
        A list of indices.

    TODO
    ----
    Add parameter `min_dist` so that
    any 2 selected indices are at least `min_dist` apart.

    """
    rng = np.random.default_rng()
    k = rng.normal(pop_size * prob, scale_ratio * pop_size)
    # print(pop_size * prob, scale_ratio*pop_size)
    k = int(round(np.clip(k, 0, pop_size)))
    indices = rng.choice(list(range(pop_size)), k).tolist()
    return indices


class RandomMasking(torch.nn.Module):
    """Randomly mask ECGs with a probability.

    Parameters
    ----------
    fs : int
        Sampling frequency of the ECGs to be augmented.
    mask_value : numbers.Real, default 0.0
        Value to mask with.
    mask_width : Sequence[numbers.Real], default ``[0.08, 0.18]``
        Width range of the masking window, with units in seconds
    kwargs : dict, optional
        Additional keyword arguments.

    Examples
    --------
    .. code-block:: python

        rm = RandomMasking(fs=500, prob=0.7)
        sig = torch.randn(32, 12, 5000)
        critical_points = [np.arange(250, 5000 - 250, step=400) for _ in range(32)]
        sig, _ = rm(sig, None, critical_points=critical_points)

    """

    __name__ = "RandomMasking"

    def __init__(
        self,
        fs: int,
        mask_value: Real = 0.0,
        mask_width: Sequence[Real] = [0.08, 0.18],
        prob: Union[Sequence[Real], Real] = [0.3, 0.15],
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.fs = fs
        self.prob = prob
        if isinstance(self.prob, Real):
            self.prob = np.array([self.prob, self.prob])
        else:
            self.prob = np.array(self.prob)
        assert (self.prob >= 0).all() and (self.prob <= 1).all(), "Probability must be between 0 and 1"
        self.mask_value = mask_value
        self.mask_width = (np.array(mask_width) * self.fs).round().astype(int)

    def forward(
        self,
        sig: Tensor,
        critical_points: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """Forward method of the RandomMasking augmenter.

        Parameters
        ----------
        sig : torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
        critical_points : Sequence[Sequence[int]], optional
            If given, random masking will be performed in windows centered at these points.
            This is useful for example when one wants to randomly mask QRS complexes.
        kwargs : dict, optional
            Not used, but kept for consistency with other augmenters.

        Returns
        -------
        sig : torch.Tensor
            The augmented ECGs, of shape ``(batch, lead, siglen)``.
        label : torch.Tensor
            Label tensor of the augmented ECGs, unchanged.
        extra_tensors : Sequence[torch.Tensor], optional
            Unchanged extra tensors.

        """
        batch, lead, siglen = sig.shape

        sig_mask_prob = 0.5 / self.mask_width[1]
        sig_mask_scale_ratio = min(self.prob[1] / 4, 0.1) / self.mask_width[1]
        mask = torch.full_like(sig, 1, dtype=sig.dtype, device=sig.device)
        for batch_idx in get_indices(prob=self.prob[0], pop_size=batch):
            if critical_points is not None:
                indices = get_indices(prob=self.prob[1], pop_size=len(critical_points[batch_idx]))
                indices = np.arange(siglen)[indices]
            else:
                indices = np.array(
                    get_indices(
                        prob=sig_mask_prob,
                        pop_size=siglen - self.mask_width[1],
                        scale_ratio=sig_mask_scale_ratio,
                    )
                )
                indices += self.mask_width[1] // 2
            for j in indices:
                masked_radius = randint(self.mask_width[0], self.mask_width[1]) // 2
                mask[batch_idx, :, j - masked_radius : j + masked_radius] = self.mask_value
            # print(f"batch_idx = {batch_idx}, len(indices) = {len(indices)}")
        sig = sig.mul_(mask)
        return sig

