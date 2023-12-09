import time
from blau_lle import BlauLocallyLinearEmbedding as BLLE
# from nred_lem import NonRedundantSpectralEmbedding as NLEM
# from nred_isomap import NonRedundantIsomap as NIsomap


def blle(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.3):
  model = BLLE(n_neighbors=n_neighbors, n_components=n_components,
               reg=1e-3, n_jobs=n_jobs, max_iter=100, tol=1e-7)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def bhlle(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.3):
  model = BLLE(n_neighbors=n_neighbors, n_components=n_components,
               method='hessian', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_


def bltsa(data, n_components=2, n_neighbors=10, n_jobs=1, alpha=0.3):
  model = BLLE(n_neighbors=n_neighbors, n_components=n_components,
               method='ltsa', reg=1e-3, n_jobs=n_jobs)
  print('Fitting model with data...', end=' ')
  start = time.time()
  result = model.fit_transform(data, alpha=alpha)
  end = time.time()
  print('Fitting done in {:.2f} seconds'.format(end - start))
  return result, model.reconstruction_error_
