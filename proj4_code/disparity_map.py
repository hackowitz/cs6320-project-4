"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def calculate_disparity_map(
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    block_size: int,
    sim_measure_function: Callable,
    max_search_bound: int = 50,
) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small 
    patch around a pixel from the left image in the right image
    
    Note: 
    1.  It is important for this project to follow the convention of search
      input in left image and search target in right image
    2.  While searching for disparity value for a patch, it may happen that there
      are multiple disparity values with the minimum value of the similarity
      measure. In that case we need to pick the smallest disparity value.
      Please check the numpy's argmin and pytorch's argmin carefully.
      Example:
      -- diparity_val -- | -- similarity error --
      -- 0               | 5 
      -- 1               | 4
      -- 2               | 7
      -- 3               | 4
      -- 4               | 12
    
      In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.
    
    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be >= 1.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   block_size: the size of the block to be used for searching between
                  left and right image
    -   sim_measure_function: a function to measure similarity measure between
                            two tensors of the same shape; returns the error value
    -   max_search_bound: the maximum horizontal distance (in terms of pixels) 
                        to use for searching
    Returns:
    -   disparity_map: The map of disparity values at each pixel. 
                     Tensor of shape (H-2*(block_size//2),W-2*(block_size//2))
    """
    assert left_img.shape == right_img.shape

    # tacky, but lets us vectorize
    H, W, C = left_img.shape
    padding = torch.zeros(H, max_search_bound - 1, C)
    left_img = torch.cat([left_img, padding], dim=1)

    return sim_measure_function(
        left_img
            .unfold(1, max_search_bound, 1)        # for each search offset
            .unfold(0, block_size, 1)              # patch y
            .unfold(1, block_size, 1),             # patch x
        right_img[..., torch.newaxis]
            .expand(-1, -1, -1, max_search_bound)  # no search offsets on rhs
            .unfold(0, block_size, 1)              # patch y
            .unfold(1, block_size, 1),             # patch x
        dim=(2, -2, -1),                           # W*H*C*O*X*Y -> W*H*O
    ).min(dim=-1).indices                          # W*H*O -> W*H
    

def calculate_cost_volume(left_img: torch.Tensor,
                          right_img: torch.Tensor,
                          max_disparity: int,
                          sim_measure_function: Callable,
                          block_size: int = 9):
  """
  Calculate the cost volume. Each pixel will have D=max_disparity cost values
  associated with it. Basically for each pixel, we compute the cost of
  different disparities and put them all into a tensor.

  Note: 
  1.  It is important for this project to follow the convention of search
      input in left image and search target in right image
  2.  If the shifted patch in the right image will go out of bounds, it is
      good to set the default cost for that pixel and disparity to be something
      high(we recommend 255), so that when we consider costs, valid disparities will have a lower
      cost. 

  Args:
  -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be 1 or 3.
  -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
  -   max_disparity:  represents the number of disparity values we will consider.
                  0 to max_disparity-1
  -   sim_measure_function: a function to measure similarity measure between
                  two tensors of the same shape; returns the error value
  -   block_size: the size of the block to be used for searching between
                  left and right image
  Returns:
  -   cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
                dimensions, and D is max_disparity. cost_volume[x,y,d] 
                represents the similarity or cost between a patch around left[x,y]  
                and a patch shifted by disparity d in the right image. 
  """
  #placeholder
  H = left_img.shape[0]
  W = right_img.shape[1]
  cost_volume = torch.zeros(H, W, max_disparity)
  ############################################################################
  # Student code begin
  ############################################################################

  raise NotImplementedError('calculate_cost_volume not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return cost_volume
def confidence_from_cost_volume(cost_volume: torch.Tensor,
                                method: str = 'peak_ratio',
                                exclusion: int = 2,
                                invalid_cost: float = 255.0) -> torch.Tensor:
  """
  Score how much we should trust the disparity chosen at each pixel.

  calculate_disparity_map() keeps only the winning disparity and throws the rest of
  the error profile away. The cost volume still has all of it, and the shape of that
  profile tells us whether the winner was a clear winner or a coin flip. That score
  is what this function returns.

  Two ways to measure it:

  'peak_ratio'
      Take the lowest cost, then the lowest cost that is at least `exclusion`
      disparities away from it, and divide the second by the first. If the winner is
      far better than anything else, the ratio is large. If something just as good
      sits elsewhere in the profile, the ratio is close to 1. Skipping the few
      disparities next to the winner matters: the points either side of a minimum
      are always low, and without the gap every pixel would score about 1.

  'curvature'
      How sharply the cost rises on either side of the winner:
      cost[d-1] + cost[d+1] - 2 * cost[d]. A narrow, deep notch gives a big number,
      a shallow dip gives a small one. This only looks at the three values around
      the minimum, so unlike peak ratio it cannot see a rival minimum further away.

  Entries equal to `invalid_cost` are disparities that were never evaluated, because
  the shifted patch would have fallen off the left edge of the image. They are not
  real costs and must be left out of both calculations.

  Return 0 for a pixel when there is no sensible answer: fewer than three disparities
  were evaluated, or the winner sits at the edge of the evaluated range, so one of
  the two neighbours that curvature needs does not exist.

  Args:
  -   cost_volume: tensor of shape (H, W, D) from calculate_cost_volume()
  -   method: 'peak_ratio' or 'curvature'
  -   exclusion: how many disparities either side of the winner to skip when
                 looking for the runner-up (peak_ratio only)
  -   invalid_cost: the filler value calculate_cost_volume() stores for disparities
                    it could not evaluate
  Returns:
  -   confidence: tensor of shape (H, W). Larger means more trustworthy. The two
                  methods are on different scales, so compare a map against itself,
                  not against the other method.
  """
  #placeholder
  confidence = torch.zeros(cost_volume.shape[0], cost_volume.shape[1])
  ############################################################################
  # Student code begin
  ############################################################################

  raise NotImplementedError('confidence_from_cost_volume not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return confidence
