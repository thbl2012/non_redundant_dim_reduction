import time
from sklearn.manifold import Isomap, LocallyLinearEmbedding as LLE, SpectralEmbedding as LEM
import numpy as np


def isomap(data, n_components=2, n_neighbors=10, n_jobs=1, **kwargs):
  model = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs, path_method='auto')
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error()


def lle(data, n_components=2, n_neighbors=10, n_jobs=1, **kwargs):
  model = LLE(n_neighbors=n_neighbors, n_components=n_components,
              method='standard', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def hlle(data, n_components=2, n_neighbors=10, n_jobs=1, **kwargs):
  model = LLE(n_neighbors=n_neighbors, n_components=n_components,
              method='hessian', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def ltsa(data, n_components=2, n_neighbors=10, n_jobs=1, **kwargs):
  model = LLE(n_neighbors=n_neighbors, n_components=n_components,
              method='ltsa', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def lem(data, n_components=2, n_neighbors=10, n_jobs=1, **kwargs):
  model = LEM(n_neighbors=n_neighbors, n_components=n_components,
              affinity='nearest_neighbors', n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, None
