"""
This file constrains different similarity measures used to compare blocks 
between two images.

Note: I've modified the functions to accept keywords that control array
broadcasting (Typically via `dim` and `keepdims` in the final reduction to a
scalar). This is essential for performant reusable code; otherwise a python
for loop might be necessary to broadcast the kernel operation across each image
patch. Maybe this loop could be vectorized with `torch.vmap` but the
torch->python->torch transitions or JIT optimizations will still cost
unacceptable overhead.
"""

import torch


def ssd_similarity_measure(
        patch1: torch.Tensor,
        patch2: torch.Tensor,
        **kwargs,
) -> torch.Tensor:
    """Test if two patches are similar by the SSD distance measure.
    
    SSD measure is sum of squared difference of pixel values in two patches.
    It is a good measure when the system has Gaussian noise.

    Args:
    -   patch1: one of the patch to compare (tensor of any shape/dimensions)
    -   patch2: the other patch to compare (tensor of the same shape as patch1)
    Returns:
    -   ssd_value: a single ssd value of the patch
    """
    return ((patch2 - patch1)**2).sum(**kwargs)


def sad_similarity_measure(
        patch1: torch.Tensor,
        patch2: torch.Tensor,
        **kwargs,
) -> torch.Tensor:
    """Test if two patches are similar by the SAD distance measure.

    SAD is the sum of absolute difference. In general, absolute differences
    are more robust to large noise/outliers than squared differences.
    Ref: https://en.wikipedia.org/wiki/Sum_of_absolute_differences

    Args:
    -   patch1: one of the patch to compare (tensor of any shape/dimensions)
    -   patch2: the other patch to compare (tensor of the same shape as patch1)
    Returns:
    -   sad_value: the scalar sad value of the patch
    """    
    return (patch2 - patch1).abs().sum(**kwargs)