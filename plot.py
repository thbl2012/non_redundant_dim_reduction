from utils import *
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


TRAIN = 'train'

plt.rcParams.update(
  {'legend.fontsize': 'x-large',
   'axes.labelsize': 'xx-large',
   'axes.titlesize': 'xx-large',
   'xtick.labelsize': 'xx-large',
   'ytick.labelsize': 'xx-large',
   'figure.figsize': (8, 6),
   'figure.subplot.left': 0.1,
   'figure.subplot.right': 0.9,
   'figure.subplot.bottom': 0.1,
   'figure.subplot.top': 0.9}
)


def plot_train_data(dataname, save=False):
  name = '{}/{}'.format(TRAIN, dataname)
  data = np.load(name + '.npy')
  positions = np.load(name + '_positions.npy')

  imgdir = 'plot_{}'.format(TRAIN)
  if save:
    os.makedirs(imgdir, exist_ok=True)

  ax1 = plt.figure(figsize=(9, 6)).add_subplot(111, projection='3d')
  ax1.scatter(data[:, 0], data[:, 1], data[:, 2], s=9, c=positions)
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  ax1.set_zlabel('z')
  if save:
    plt.savefig('{}/{}_3d.png'.format(imgdir, dataname), bbox_inches='tight', dpi=200)
  plt.show()
  plt.close()

  plt.scatter(data[:, 0], data[:, 1], s=5, c=positions)
  plt.xlabel('x')
  plt.ylabel('y')
  if save:
    plt.savefig('{}/{}_xy.png'.format(imgdir, dataname), bbox_inches='tight', dpi=200)
  # plt.show()
  plt.close()

  plt.scatter(data[:, 0], data[:, 2], s=5, c=positions)
  plt.xlabel('x')
  plt.ylabel('z')
  if save:
    plt.savefig('{}/{}_xz.png'.format(imgdir, dataname), bbox_inches='tight', dpi=200)
  # plt.show()
  plt.close()

  plt.scatter(data[:, 1], data[:, 2], s=5, c=positions)
  plt.xlabel('y')
  plt.ylabel('z')
  if save:
    plt.savefig('{}/{}_yz.png'.format(imgdir, dataname), bbox_inches='tight', dpi=200)
  # plt.show()
  plt.close()


def plot_transformed_data_3d(method_name, dataname, color_axis=1, save=False):
  if method_name == TRAIN:
    raise ValueError('This function is not intended to plot training data')
  name = '{}/{}'.format(method_name, dataname)
  train_data = np.load('{}/{}.npy'.format(TRAIN, dataname))
  positions = np.load(name + '.npy')[:, color_axis - 1]

  imgdir = 'plot_{}'.format(method_name)

  ax1 = plt.figure(figsize=(9, 6)).add_subplot(111, projection='3d')
  ax1.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], s=9, c=positions)
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  ax1.set_zlabel('z')
  # set_axes_equal(ax1)
  if save:
    os.makedirs(imgdir, exist_ok=True)
    plt.savefig('{}/{}_3d_ax{}.png'.format(imgdir, dataname, color_axis), bbox_inches='tight', dpi=200)
  plt.show()
  plt.close()


def plot_transformed_data_2d(method_name, dataname, axes=(1, 2), save=False, same_color=False):
  if method_name == TRAIN:
    raise ValueError('This function is not intended to plot training data')
  name = '{}/{}'.format(method_name, dataname)
  transformed_data = np.load(name + '.npy')
  if same_color:
    positions = 'k'
  else:
    positions = np.load('{}/{}_positions.npy'.format(TRAIN, dataname))

  imgdir = 'plot_{}'.format(method_name)
  plt.scatter(transformed_data[:, axes[0] - 1], transformed_data[:, axes[1] - 1], s=5, c=positions)
  plt.xlabel('x{}'.format(axes[0]))
  plt.ylabel('x{}'.format(axes[1]))
  if save:
    os.makedirs(imgdir, exist_ok=True)
    plt.savefig('{}/{}_x{}x{}.png'.format(imgdir, dataname, axes[0], axes[1]),
                bbox_inches='tight', dpi=200)
  plt.show()
  plt.close()


def plot_obj_views(method_name, save=False):
  dataname = 'person'
  if method_name == TRAIN:
    raise ValueError('This function is not intended to plot training data')
  name = '{}/{}'.format(method_name, dataname)
  transformed_data = np.load(name + '.npy')
  angles = np.load('{}/{}_angle.npy'.format(TRAIN, dataname))

  print(transformed_data.shape)
  print(angles.shape)

  imgdir = 'plot_{}'.format(method_name)

  for i in range(2):
    plt.scatter(angles[:, 0], angles[:, 1], s=36, c=transformed_data[:, i])
    plt.show()
    plt.close()

  plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=16, c='r')
  plt.show()
  plt.close()

  angle_names = ['tilt', 'pan']
  fig, axes = plt.subplots(2, 2, figsize=(10, 6))
  for i in range(2):
    for j in range(2):
      axes[i, j].scatter(angles[:, i], transformed_data[:, j], s=10, c='r')
  fig.subplots_adjust(top=0.82)
  if save:
    os.makedirs(imgdir, exist_ok=True)
    plt.savefig('{}/{}_views.png'.format(imgdir, dataname), bbox_inches='tight', dpi=200)
  plt.show()
  plt.close()


def plot_knn_graph(method_name, save=False):
  dataname = 'person'
  if method_name == TRAIN:
    raise ValueError('This function is not intended to plot training data')
  name = '{}/{}'.format(method_name, dataname)
  transformed_data = np.load(name + '.npy')
  angles = np.load('{}/{}_angle.npy'.format(TRAIN, dataname))

  print(transformed_data.shape)
  print(angles.shape)

  imgdir = 'plot_{}'.format(method_name)


def patches_to_image(method_name, dataname, component=1, save=False):
  if method_name == TRAIN:
    raise ValueError('This function is not intended to plot training data')
  patches = np.load('{}/{}.npy'.format(method_name, dataname))[:, :, component-1]
  sign = 2 * (patches[0] > 0) - 1
  plt.imshow(sign * patches, cmap='jet')
  plt.colorbar()
  imgdir = 'plot_{}'.format(method_name)
  if save:
    os.makedirs(imgdir, exist_ok=True)
    plt.savefig('{}/{}_x{}.png'.format(imgdir, dataname, component), bbox_inches='tight', format='png')
  plt.show()
  plt.close()


def show_image(imgname, method_name, save=False):
  img = np.load('{}/{}.npy'.format(method_name, imgname))
  plt.imshow(img, cmap='gray')
  plt.colorbar()
  imgdir = 'plot_{}'.format(method_name)
  if save:
    os.makedirs(imgdir, exist_ok=True)
    plt.savefig('{}/{}.png'.format(imgdir, imgname), bbox_inches='tight', format='png')
  plt.show()
  plt.close()
  # import cv2
  # cv2.imwrite('{}/{}_recon_img.jpg'.format(imgdir, imgname), img)
