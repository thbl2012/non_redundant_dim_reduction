import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import *


def gaussian_kernel(x, y, bandwidth=1):
  return np.exp(- (np.sum((x - y) ** 2, axis=-1) / (2 * bandwidth ** 2)))


def loo_nadaraya_watson_predict(X, y, bandwidth=1, leave_one_out=True):
  y_pred = np.empty(y.shape[0], dtype=y.dtype)
  for i in range(y.shape[0]):
    weights = gaussian_kernel(X[i], X, bandwidth=bandwidth)
    dom = np.sum(weights * y) - leave_one_out * weights[i] * y[i]
    denom = np.sum(weights) - leave_one_out * weights[i]
    y_pred[i] = dom / denom
  return y_pred


def patches_reconstruction(patches, components, bandwidth=1, leave_one_out=True):
  height, width, patch_height, patch_width = patches.shape[:4]
  components_flat = components.reshape((height * width, ) + components.shape[2:])
  patches_pred = np.empty(patches.shape)

  kernel_matrix = squareform(np.exp(- pdist(components_flat, metric='sqeuclidean') / (2 * bandwidth**2)))
  kernel_matrix.flat[::kernel_matrix.shape[0] + 1] = (1 - leave_one_out)
  kernel_matrix /= np.sum(kernel_matrix, axis=1)[:, np.newaxis]

  return np.dot(kernel_matrix,
                patches.reshape((height * width, patch_height * patch_width) + patches.shape[4:])
                ).reshape(patches.shape)

  # for i in range(patch_height):
  #   for j in range(patch_width):
  #     patches_pred.flat[i * patch_width + j::patch_height * patch_width] \
  #         = np.dot(kernel_matrix, patches.flat[i * patch_width + j::patch_height * patch_width])
  # return patches_pred


def image_reconstruction(patches, slide_height, slide_width=0):
  if slide_width == 0:
    slide_width = slide_height
  height, width, patch_height, patch_width = patches.shape[:4]
  img_height = slide_height * (height - 1) + patch_height
  img_width = slide_width * (width - 1) + patch_width
  img = np.zeros((img_height, img_width), dtype=np.float)
  overlap_count = np.zeros((img_height, img_width), dtype=np.int32)
  for i in range(height):
    for j in range(width):
      actual_i = i * slide_height
      actual_j = j * slide_width
      img[actual_i:actual_i + patch_height, actual_j: actual_j + patch_width] += patches[i, j]
      overlap_count[actual_i:actual_i + patch_height, actual_j: actual_j + patch_width] += 1
  img /= overlap_count
  return img
