# pylint: disable=bad-indentation
# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cow mask generation. 
https://github.com/google-research/google-research/blob/master/milking_cowmask/
Adapted from LAX implementation to NumPy due to PyTorch dataloader 
being incompatible with JAX
Author: Shihao Shen
Date: 29th Aug 2022
"""
import math
import numpy as np
from scipy import special
from scipy.signal import convolve

_ROOT_2 = math.sqrt(2.0)
_ROOT_2_PI = math.sqrt(2.0 * math.pi)


def gaussian_kernels(sigma, max_sigma):
	"""Make Gaussian kernels for Gaussian blur.
	Args:
		sigma: kernel sigma
		max_sigma: sigma upper limit as a float (this is used to determine
			the size of kernel required to fit all kernels)
  	Returns:
	  	a (1, kernel_width) numpy array
  	"""
	size = round(max_sigma * 3) * 2 + 1
	x = np.arange(-size, size + 1)[None, :].astype(np.float32)
	y = np.exp(-0.5 * x ** 2 / sigma ** 2)
	return y / (sigma * _ROOT_2_PI)


def cow_masks(mask_size, log_sigma_range, max_sigma, prop_range):
	"""Generate Cow Mask.
	Args:
      n_masks: number of masks to generate as an int
      mask_size: image size as a `(height, width)` tuple
      log_sigma_range: the range of the sigma (smoothing kernel)
          parameter in log-space`(log(sigma_min), log(sigma_max))`
      max_sigma: smoothing sigma upper limit
      prop_range: range from which to draw the proportion `p` that
        controls the proportion of pixel in a mask that are 1 vs 0
  Returns:
      Cow Masks as a [v, height, width, 1] numpy array
	"""

	# Draw the per-mask proportion p
	p = np.random.uniform(prop_range[0], prop_range[1])
	# Compute threshold factors
	threshold_factor = special.erfinv(2 * p - 1) * _ROOT_2

	sigma = np.exp(np.random.uniform(log_sigma_range[0], log_sigma_range[1]))

	noise = np.random.normal(size=mask_size)

	# Generate a kernel for each sigma
	kernel = gaussian_kernels(sigma, max_sigma)
	kernel = kernel.squeeze()
	# kernels in y and x
	krn_y = kernel[None, :]
	krn_x = kernel[:, None]

	# Apply kernels in y and x separately
	smooth_noise = convolve(noise, krn_y, mode='same')
	smooth_noise = convolve(smooth_noise, krn_x, mode='same')

	# Compute mean and std-dev
	noise_mu = smooth_noise.mean(axis=(0,1))
	noise_sigma = smooth_noise.std(axis=(0,1))
	# Compute thresholds
	threshold = threshold_factor * noise_sigma + noise_mu
	# Apply threshold
	mask = (smooth_noise <= threshold).astype(bool)

	return mask


if __name__=="__main__":
    import time
    import matplotlib.pyplot as plt

    cow_sigma_range = (20, 60)
    log_sigma_range = (math.log(cow_sigma_range[0]), math.log(cow_sigma_range[1]))
    cow_prop_range = (0.1, 0.5)
    s = time.time()
    max_iou = 0
    # for _ in range(1000):
    #     mask = cow_masks((240, 360), log_sigma_range, cow_sigma_range[1], cow_prop_range)
    #     max_iou = max(max_iou, np.sum(mask) / (240*360))
    # print(time.time() - s)
    # print(max_iou)

    mask = cow_masks((240, 360), log_sigma_range, cow_sigma_range[1], cow_prop_range)
    print(np.sum(mask) / (240*360))
    plt.imshow(mask * 255)
    plt.savefig('mask.png')