"""
Unit tests for disparity_map function
"""

import torch
import numpy as np

from proj4_code.disparity_map import (calculate_disparity_map, calculate_cost_volume,
                                      confidence_from_cost_volume)
from proj4_code.similarity_measures import ssd_similarity_measure, sad_similarity_measure
from proj4_code.utils import generate_random_stereogram, generate_delta_fn_images


def test_disparity_deltafn_success():
  """
  Tests the disparity map giving inputs which just have a single pixel value
  """
  im_dim = 51
  block_size = 1

  im_left, im_right = generate_delta_fn_images((im_dim, im_dim))

  # calculate the disparity manually
  left_idx = torch.argmax(im_left[:, :, 1]).item()
  left_r = left_idx//im_dim
  left_c = left_idx - left_r*im_dim

  right_idx = torch.argmax(im_right[:, :, 1]).item()
  right_r = right_idx // im_dim
  right_c = right_idx - right_r*im_dim

  disparity_expected = left_c - right_c

  # get the disparity map from the function
  disp_map = calculate_disparity_map(im_left,
                                     im_right,
                                     block_size,
                                     ssd_similarity_measure,
                                     max_search_bound=disparity_expected+3
                                     )

  # we should get two non-zero values in the disparity map
  nonzero_disp = torch.nonzero(disp_map).data

  # check the size
  assert nonzero_disp.size() == (2, 2)

  # check that the ows are same
  assert nonzero_disp[0, 0].item() == nonzero_disp[1, 0].item()
  assert nonzero_disp[0, 1].item() + nonzero_disp[1,
                                                  1].item() == left_c + right_c
  assert abs(nonzero_disp[0, 1].item() -
             nonzero_disp[1, 1].item()) == abs(left_c - right_c)

  val1 = disp_map[left_r, left_c].item()
  val2 = disp_map[right_r, right_c].item()

  assert val1 == disparity_expected
  # At the right image's bright pixel the left image is dark, so the best match is one
  # column over: the expected disparity there is 1. (This used to be written as
  # `== block_size`, which passes only because block_size happens to be 1.)
  assert val2 == 1


def test_disparity_deltafn_failure():
  """
  Tests the disparity map giving inputs which just have a single pixel value
  The bounds for search will be smaller and result in a failure
  """
  im_dim = 51
  block_size = 1

  im_left, im_right = generate_delta_fn_images((im_dim, im_dim))

  # calculate the disparity manually
  left_idx = torch.argmax(im_left[:, :, 1]).item()
  left_r = left_idx//im_dim
  left_c = left_idx - left_r*im_dim

  right_idx = torch.argmax(im_right[:, :, 1]).item()
  right_r = right_idx // im_dim
  right_c = right_idx - right_r*im_dim

  disparity_expected = left_c - right_c

  # get the disparity map from the function
  disp_map = calculate_disparity_map(im_left,
                                     im_right,
                                     block_size,
                                     ssd_similarity_measure,
                                     max_search_bound=disparity_expected-1
                                     )
 
  # we should get two non-zero values in the disparity map
  nonzero_disp = torch.nonzero(disp_map).data

  # check the size
  assert nonzero_disp.size() != (2, 2)

  val1 = disp_map[left_r, left_c].item()
  val2 = disp_map[right_r, right_c].item()

  assert val1 == 0


def test_disparity_map_size():
  """
  Checks the size of the disparity map
  """
  im_left = torch.randn((15, 15, 1))
  im_right = torch.randn_like(im_left)

  for block_size in (3, 5, 7, 9, 13):
    disp_map = calculate_disparity_map(
        im_left, im_right, block_size, ssd_similarity_measure)
    assert disp_map.size() == (max(0, 15-block_size+1), max(0, 15-block_size+1))


def test_disparity_random_stereogram():
  """
  Checks the disparity map for random stereogram 
  """
  H = 51
  W = 51
  disparity = 4
  im_left, im_right = generate_random_stereogram(
      im_size=(H, W, 3), disparity=4)

  block_size = 5
  disp_map = calculate_disparity_map(
      im_left, im_right, block_size, ssd_similarity_measure)

  # define the limits where we will encounter non-zero disparity maps
  x_lims = torch.tensor([H//2 - H//4 - block_size//2-1, H//2 + H//4 + block_size//2-1],
                        dtype=torch.long)
  y_lims = torch.tensor([W//2 - W//4 - block_size//2 - disparity-1, W//2 + W//4 + block_size//2-1],
                        dtype=torch.long)

  # get the points where disparity map is greater than zero
  nonzero_idx = torch.nonzero(disp_map)

  # we will see the effect of stereo outside this region too, depending on the block size
  falsevals = torch.nonzero(~(
      (nonzero_idx[:, 0] >= x_lims[0]) &
      (nonzero_idx[:, 0] <= x_lims[1]) &
      (nonzero_idx[:, 1] >= y_lims[0]) &
      (nonzero_idx[:, 1] <= y_lims[1])
  ))

  assert ~falsevals.shape[0]


def test_disparity_translation_shift():
  """
  Test where we generate the 2nd image by just horizonataly shifting the 1st image
  """
  H, W = (21, 21)
  shift_val = 4
  im1 = torch.randn(H, W, 2)
  W = W - shift_val
  im2 = im1[:, shift_val:, :].clone()
  im1 = im1[:, :-shift_val, :].clone()

  block_size = 3
  disp_map = calculate_disparity_map(
      im1, im2, block_size, ssd_similarity_measure)

  assert disp_map.shape[0] == H-2*(block_size//2)
  assert disp_map.shape[1] == W-2*(block_size//2)

  assert torch.nonzero(disp_map[:, shift_val+1:] != shift_val).shape[0] == 0
    
def test_calculate_cost_volume():
    """
    Test calculate cost volume with simple dot 
    """
    left_image = torch.zeros((10,10,3))
    left_image[8,6,:] = 1
    right_image = torch.zeros((10,10,3))
    right_image[8,5,:] = 1

    cost_volume = calculate_cost_volume(left_image, right_image, 4, sad_similarity_measure, 1)

    assert np.all(np.isclose(cost_volume[8,6,:].cpu().numpy(),[3,0,3,3]))

    left_image = torch.zeros((10,10,3))
    left_image[8,7,:] = 1
    right_image = torch.zeros((10,10,3))
    right_image[8,1,:] = 1

    cost_volume = calculate_cost_volume(left_image, right_image, 7, sad_similarity_measure, 1)

    assert np.all(np.isclose(cost_volume[8,7,:].cpu().numpy(),[3, 3, 3, 3, 3, 3, 0]))

    left_image = torch.zeros((10,10,3))
    left_image[5,6,:] = 1
    right_image = torch.zeros((10,10,3))
    right_image[5,3,:] = 1

    cost_volume = calculate_cost_volume(left_image, right_image, 7, sad_similarity_measure, 2)

    assert np.argmin(cost_volume[5,6,:].cpu().numpy()) == 3


def test_confidence_peak_ratio():
  """
  Two hand-made error profiles with known answers.

  Pixel (0,0) has one clear minimum: cost 2 at d=4, everything else 10. The runner-up
  outside the +/-2 window is 10, so the ratio is 10/2 = 5.
  Pixel (0,1) has two minima of equal depth, at d=2 and d=6. The second is outside the
  window around the first, so the ratio is 2/2 = 1 -- correctly reporting that the
  match is a coin flip.
  """
  cost_volume = torch.full((1, 2, 10), 10.0)
  cost_volume[0, 0, 4] = 2.0
  cost_volume[0, 1, 2] = 2.0
  cost_volume[0, 1, 6] = 2.0

  confidence = confidence_from_cost_volume(cost_volume, method='peak_ratio')

  assert confidence.shape == (1, 2), \
      'expected one score per pixel, shape (H, W); got {}'.format(tuple(confidence.shape))
  assert abs(float(confidence[0, 0]) - 5.0) < 1e-4, \
      'one deep minimum at d=4 should score 10/2 = 5, got {:.3f}'.format(float(confidence[0, 0]))
  assert abs(float(confidence[0, 1]) - 1.0) < 1e-4, \
      'two equally deep minima should score 1, got {:.3f}. Make sure the runner-up is ' \
      'taken from outside the exclusion window.'.format(float(confidence[0, 1]))


def test_confidence_curvature():
  """
  Curvature should separate a sharp notch from a shallow dip, and should refuse to
  answer when the winner sits next to a disparity that was never evaluated.

  Pixel (0,0): sharp,   10 + 10 - 2*2 = 16.
  Pixel (0,1): shallow,  6 +  6 - 2*5 = 2.
  Pixel (0,2): the costs fall steadily to the last evaluated disparity, d=4, and
               d=5 onward are the 255 filler. The winner has no right-hand
               neighbour, so the answer is 0 and NOT 6 + 255 - 2*5 = 251.
  """
  cost_volume = torch.full((1, 3, 8), 10.0)
  cost_volume[0, 0, 4] = 2.0
  cost_volume[0, 1, :] = 6.0
  cost_volume[0, 1, 4] = 5.0
  cost_volume[0, 2, :] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 255.0, 255.0, 255.0])

  confidence = confidence_from_cost_volume(cost_volume, method='curvature')

  assert abs(float(confidence[0, 0]) - 16.0) < 1e-4, \
      'sharp minimum should score 10 + 10 - 2*2 = 16, got {:.3f}'.format(float(confidence[0, 0]))
  assert abs(float(confidence[0, 1]) - 2.0) < 1e-4, \
      'shallow minimum should score 6 + 6 - 2*5 = 2, got {:.3f}'.format(float(confidence[0, 1]))
  assert float(confidence[0, 2]) == 0.0, \
      'the winner sits at the last evaluated disparity, so curvature is undefined and ' \
      'the score should be 0; got {:.3f}. The 255 filler is not a real cost.'.format(
          float(confidence[0, 2]))
