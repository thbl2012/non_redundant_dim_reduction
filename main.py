from data import *
from plot import *
from itertools import combinations
import original_methods
import nonredundant
import blau
from image_reconstruction import *
from utils import *


def dim_reduction(dataname, method_name, n_components, n_neighbors, n_jobs=1, alpha=0.6):
  name = '{}/{}.npy'.format(TRAIN, dataname)
  data = np.load(name)
  print('Data "{}" loaded. Shape: {}'.format(dataname, data.shape))
  if method_name[0] == 'n':
    module = nonredundant
  elif method_name[0] == 'b':
    module = blau
  else:
    module = original_methods
  result, err = getattr(module, method_name)(
    data, n_components=n_components, n_neighbors=n_neighbors, n_jobs=n_jobs, alpha=alpha
  )
  print('Reconstruction error: {}'.format(err))
  os.makedirs(method_name, exist_ok=True)
  np.save('{}/{}'.format(method_name, dataname), result)


def generate_image_patches(imgname, patch_size, slide_size):
  imdata = get_image_data(imgname, grayscale=True)
  print(imdata.shape)
  for i in range(5, 25, 5):
    if i != 10:
      continue
    patches = get_image_patches('g_' + imgname, i, slide=(slide_size, slide_size))
    print(i, patches.shape, sep=': ')


def image_components(dataname, method_name, n_components, n_neighbors, n_jobs=1, alpha=0.6):
  name = '{}/{}.npy'.format(TRAIN, dataname)
  data = np.load(name)
  print('Data "{}" loaded. Shape: {}'.format(name, data.shape))
  height, width, patch_height, patch_width = data.shape[:4]
  if method_name[0] == 'n':
    module = nonredundant
  elif method_name[0] == 'b':
    module = blau
  else:
    module = original_methods
  result, err = getattr(module, method_name)(
    data.reshape((height * width, patch_height * patch_width) + data.shape[4:]),
    n_components=n_components, n_neighbors=n_neighbors, n_jobs=n_jobs, alpha=alpha
  )
  print('Reconstruction error: {}'.format(err))
  result /= np.sqrt(np.sum(result ** 2, axis=0))
  os.makedirs(method_name, exist_ok=True)
  np.save('{}/{}'.format(method_name, dataname),
          result.reshape(height, width, n_components))
  return method_name


def reconstruct_grayscale_image(imgname, method_name, patch_size,
                                slide_size, bandwidth=1., leave_one_out=True):
  dataname = 'g_{0:}_patches_{1:}_{1:}_slide_{2:}_{2:}'.format(
    imgname, patch_size, slide_size
  )
  patches = np.load('{}/{}.npy'.format(TRAIN, dataname))
  components = np.load('{}/{}.npy'.format(method_name, dataname))
  recon_patches = patches_reconstruction(patches, components,
                                         bandwidth=bandwidth, leave_one_out=leave_one_out)
  recon_img = image_reconstruction(recon_patches, slide_size)
  np.save('{}/{}_recon_img'.format(method_name, dataname), recon_img)
  cv2.imwrite('plot_{}/{}_recon_img_cv2.png'.format(method_name, dataname), recon_img)
  # Calculate PSNR
  img = np.load('{}/g_{}.npy'.format(TRAIN, imgname))
  rmse = np.sqrt(np.mean((img - recon_img) ** 2))
  PIXEL_MAX = 255.0
  print('PSNR: {}'.format(20 * math.log10(PIXEL_MAX / rmse)))


def obj_views_detection(dataname, method_name, n_components, n_neighbors, n_jobs=1, alpha=0.6):
  name = '{}/{}.npy'.format(TRAIN, dataname)
  data = np.load(name)
  print('Data "{}" loaded. Shape: {}'.format(dataname, data.shape))
  if method_name[0] == 'n':
    module = nonredundant
  elif method_name[0] == 'b':
    module = blau
  else:
    module = original_methods
  result, err = getattr(module, method_name)(
    data.reshape(data.shape[0], -1), n_components=n_components,
    n_neighbors=n_neighbors, n_jobs=n_jobs, alpha=alpha
  )
  print('Reconstruction error: {}'.format(err))
  os.makedirs(method_name, exist_ok=True)
  np.save('{}/{}'.format(method_name, dataname), result)
  # for i in range(1, n_components+1):
  #   plot_transformed_data_3d(method_name, dataname, color_axis=i, save=True)
  # for i, j in combinations(range(1, n_components+1), 2):
  #   plot_transformed_data_2d(method_name, dataname, axes=(i, j), save=True, same_color=True)


def test_neighbors(n_components, n_neighbors, n_jobs=1, alpha=0.6):
  dataname = 'person'
  method_name = 'nltsa'
  name = '{}/{}.npy'.format(TRAIN, dataname)
  data = np.load(name)
  data = data.reshape(data.shape[0], -1)
  print('Data "{}" loaded. Shape: {}'.format(dataname, data.shape))
  from nred_lle import NonRedundantLocallyLinearEmbedding as NLLE
  import time
  model = NLLE(n_neighbors=n_neighbors, n_components=n_components,
               method='ltsa', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  _, indices = model.nbrs_.kneighbors(data)
  angles = np.load('{}/{}_angle.npy'.format(TRAIN, dataname))
  for pair in indices:
    plt.plot(angles[pair][:, 0], angles[pair][:, 1], linestyle='-', marker='.', markersize=25, c='r')
  plt.show()

  # os.makedirs(method_name, exist_ok=True)
  # np.save('{}/{}'.format(method_name, dataname), result)


def extract_squares():
  height = 120
  width = 120
  topleft = [(140, 7), (110, 435), (280, 310), (483, 431), (280, 764), (85, 770)]
  names = ['isomap', 'nisomap', '']
  for name in names:
    img = cv2.imread('compare/whitehouse{}.png'.format(name), cv2.IMREAD_GRAYSCALE)
    for index, (i, j) in enumerate(topleft):
      cv2.imwrite('compare/whitehouse{}_{}.png'.format(name, index + 1),
                  img[i:i + height, j:j + width])


def mark_photo():
  thick = 5
  height = 120
  width = 120
  topleft = [(140, 7), (110, 435), (280, 764), (85, 770)]
  img = cv2.imread('compare/whitehouse.png', cv2.IMREAD_COLOR)
  red = np.array([0, 0, 255])
  for index, (i, j) in enumerate(topleft):
    img[i - thick:i, j:j + width + thick] = red
    img[i:i + height + thick, j + width:j + width + thick] = red
    img[i + height:i + height + thick, j - thick:j + width] = red
    img[i - thick:i + height, j - thick:j] = red
  cv2.imwrite('compare/whitehouse_marked.png', img)


def main():
  imgname = 'white_house'
  patch_size = 20
  slide = 10
  n_components = 2
  dataname = 'g_{0:}_patches_{1:}_{1:}_slide_{2:}_{2:}'.format(imgname, patch_size, slide)
  method_name = 'nltsa'
  n_samples = 2500
  swiss_roll_name = 'swiss_roll_{}'.format(n_samples)
  n_jobs = -1

  # get_swiss_roll(n_samples, length=60, width=10, noise=0.5)
  # plot_train_data(swiss_roll_name)
  #
  dim_reduction(swiss_roll_name, method_name, n_components=n_components, n_neighbors=20, n_jobs=n_jobs, alpha=0.6)
  for i in range(1, n_components+1):
    plot_transformed_data_3d(method_name, swiss_roll_name, color_axis=i, save=True)
  for i, j in combinations(range(1, n_components+1), 2):
    plot_transformed_data_2d(method_name, swiss_roll_name, axes=(i, j), save=True)

  # obj_views_detection('person', method_name, n_components=2, n_neighbors=12, alpha=0.1)
  # plot_obj_views(method_name, save=True)
  # test_neighbors(n_components=n_components, n_neighbors=2, n_jobs=n_jobs)

  # get_image_data(imgname, grayscale=True)
  # patches = get_image_patches('g_' + imgname, patch_size, slide=(slide, slide))
  # print(patches.shape, sep=': ')
  # image_components(dataname, method_name, n_components=n_components, n_neighbors=100, alpha=0.6)
  # for i in range(1, n_components+1):
  #   patches_to_image(method_name, dataname, i, save=True)
  # reconstruct_grayscale_image(imgname, method_name, patch_size, slide, bandwidth=0.001, leave_one_out=True)
  # show_image('{}_recon_img'.format(dataname), method_name, save=True)

  # extract_squares()
  # mark_photo()


if __name__ == '__main__':
  main()
