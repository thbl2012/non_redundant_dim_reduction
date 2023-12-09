import time
from nred_lle import NonRedundantLocallyLinearEmbedding as NLLE
from nred_lem import NonRedundantSpectralEmbedding as NLEM
from nred_isomap import NonRedundantIsomap as NIsomap


def nlle(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.3):
  model = NLLE(n_neighbors=n_neighbors, n_components=n_components,
               reg=1e-3, n_jobs=n_jobs, max_iter=100, tol=1e-7)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def nhlle(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.3):
  model = NLLE(n_neighbors=n_neighbors, n_components=n_components,
               method='hessian', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def nltsa(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.3):
  model = NLLE(n_neighbors=n_neighbors, n_components=n_components,
               method='ltsa', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def nlem(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.5):
  model = NLEM(n_neighbors=n_neighbors, n_components=n_components,
               affinity='nearest_neighbors', n_jobs=n_jobs, alpha=alpha)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, None


def nisomap(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.5):
  model = NIsomap(n_neighbors=n_neighbors, n_components=n_components,
                  n_jobs=n_jobs, path_method='auto', alpha=alpha)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error()

