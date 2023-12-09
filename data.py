import os
import numpy as np
import math
import cv2
import re
from sklearn.feature_extraction.image import extract_patches_2d
from utils import *


def get_swiss_roll(n_samples, length=10, width=5, noise=0.5, save=True):
  std_end = 4 * math.pi
  std_length = 0.5 * (std_end * math.sqrt(std_end ** 2 + 1)
                      + math.log(std_end + math.sqrt(std_end ** 2 + 1)))
  ratio = length / std_length
  # create dataset
  phi = np.random.uniform(0 + math.pi/2, std_end + math.pi/2, size=n_samples)
  Z = width * np.random.rand(n_samples)
  X = ratio * phi * np.sin(phi)
  Y = ratio * phi * np.cos(phi)
  err = np.random.normal(0, scale=noise, size=(n_samples, 3))

  swiss_roll = np.array([X, Y, Z]).transpose() + err

  # check that we have the right shape
  print(swiss_roll.shape)
  if save:
    np.save(TRAIN + '/swiss_roll_{}.npy'.format(n_samples), swiss_roll)
    np.save(TRAIN + '/swiss_roll_{}_positions.npy'.format(n_samples), phi)
  return swiss_roll, phi


def get_noisy_swiss_roll(n_samples, length=10, width=5, noise=0.5, save=True):
  std_end = 4 * math.pi
  std_length = 0.5 * (std_end * math.sqrt(std_end ** 2 + 1)
                      + math.log(std_end + math.sqrt(std_end ** 2 + 1)))
  ratio = length / std_length
  # create dataset
  phi = np.random.uniform(0 + math.pi/2, std_end + math.pi/2, size=n_samples)
  Z = width * np.random.rand(n_samples)
  X = ratio * phi * np.sin(phi)
  Y = ratio * phi * np.cos(phi)
  err = np.random.normal(0, scale=noise, size=(n_samples, 3))

  swiss_roll = np.array([X, Y, Z]).transpose() + err

  # check that we have the right shape
  print(swiss_roll.shape)
  if save:
    np.save(TRAIN + '/noisy_swiss_roll_{}.npy'.format(n_samples), swiss_roll)
    np.save(TRAIN + '/noisy_swiss_roll_{}_positions.npy'.format(n_samples), phi)
  return swiss_roll, phi


def get_swiss_roll_holes(n_samples, len_phi=10, len_z=5, hole_radius=5, save=True):
  swiss_roll, positions = get_swiss_roll(n_samples=n_samples, length=len_phi, width=len_z, save=False)
  centers = np.array([[-10., 10., 0.], [6., 10., 0.], [13., 10., 0.]])
  swiss_roll_holes = swiss_roll
  positions_holes = positions
  for p in centers:
    choices = np.sum((swiss_roll_holes - p) ** 2, axis=1) > hole_radius ** 2
    swiss_roll_holes = swiss_roll_holes[choices]
    positions_holes = positions_holes[choices]
  np.save(TRAIN + '/swiss_roll_{}_holes.npy'.format(n_samples), swiss_roll_holes)
  np.save(TRAIN + '/swiss_roll_{}_holes_positions.npy'.format(n_samples), positions_holes)


def get_image_data(imgname, grayscale=False):
  if grayscale:
    imdata = cv2.imread(imgname + '.png', cv2.IMREAD_GRAYSCALE)
  else:
    imdata = cv2.imread(imgname + '.png', cv2.IMREAD_COLOR)
  if imdata is None:
    if grayscale:
      imdata = cv2.imread(imgname + '.jpg', cv2.IMREAD_GRAYSCALE)
    else:
      imdata = cv2.imread(imgname + '.jpg', cv2.IMREAD_COLOR)
  np.save('{}/{}_{}'.format(TRAIN, 'g' if grayscale else 'c', imgname), imdata)
  return imdata


def get_image_patches(imgname, height, width=0, slide=None):
  if width == 0:
    width = height
  img = np.load('{}/{}.npy'.format(TRAIN, imgname))
  img_height, img_width = img.shape[:2]
  patches = extract_patches_2d(img, (height, width))
  patches = patches.reshape((img_height - height + 1, img_width - width + 1) + patches.shape[1:])
  if slide:
    patches = patches[0::slide[0], 0::slide[1]]
  else:
    slide = (1, 1)
  np.save('{}/{}_patches_{}_{}_slide_{}_{}'.format(TRAIN, imgname, height, width, *slide), patches)
  return patches


def get_obj14_data():
  data = np.empty((OBJECT14_IMG_COUNT, OBJECT14_IMG_HEIGHT, OBJECT14_IMG_WIDTH))
  for i in range(72):
    data[i] = cv2.imread('{}/obj14_raw/obj14__{}.png'.format(TRAIN, i * 5), cv2.IMREAD_GRAYSCALE)
  np.save('{}/obj14'.format(TRAIN), data)


def get_person1_data():
  data = np.empty((PERSON01_IMG_COUNT // 2, PERSON01_IMG_HEIGHT, PERSON01_IMG_WIDTH), dtype=np.int)
  angles = np.empty((PERSON01_IMG_COUNT // 2, 2))
  dirname = '{}/head_pose_1/Person01/series1/'.format(TRAIN)
  for i, filename in enumerate([dirname + name for name
                                in os.listdir(dirname)
                                if name[-4:] == '.jpg']):
    data[i] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    starts = [m.start(0) for m in re.finditer('[-|+]', filename)]
    angles[i, 0] = int(filename[starts[0]:starts[1]])
    angles[i, 1] = int(filename[starts[1]:-4])
    print(i)
  np.save('{}/person'.format(TRAIN), data)
  np.save('{}/person_angle'.format(TRAIN), angles)
  return data, angles


def get_person1_unique_data():
  data = np.empty((PERSON01_IMG_COUNT, PERSON01_IMG_HEIGHT, PERSON01_IMG_WIDTH), dtype=np.int)
  angles = np.empty((PERSON01_IMG_COUNT, 2))
  dirname = '{}/head_pose_1/Person01/'.format(TRAIN)
  imgdict = dict()
  for i, filename in enumerate([dirname + name for name
                                in os.listdir(dirname)
                                if name[-4:] == '.jpg']):
    starts = [m.start(0) for m in re.finditer('[-|+]', filename)]
    tilt = int(filename[starts[0]:starts[1]])# / 90
    pan = int(filename[starts[1]:-4])# / 90
    if (tilt, pan) not in imgdict:
      imgdict[(tilt, pan)] = []
    imgdict[(tilt, pan)].append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

  import itertools
  for x, y in itertools.product([-90, -60, -30, -15, 0, +15, +30, +60, +90],
                                [-90, -75, -60, -45, -30, -15, 0, +15, +30, +45, +60, +75, +90]):
    if (x, y) not in imgdict:
      print(x, y)

  # for t, l in imgdict.items():
  #   print(t, len(l), sep=': ')

  # np.save('{}/person'.format(TRAIN), data)
  # np.save('{}/person_angle'.format(TRAIN), angles)
  # return data, angles

