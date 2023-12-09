from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from scipy.linalg.decomp_svd import svd
from scipy.linalg.decomp import eigh
from scipy.sparse.linalg import eigsh
from utils import *


def nred_pca(M, k, alpha=0.6, eigen_solver='arpack', tol=1E-6,
             max_iter=100, random_state=None):
  if eigen_solver == 'auto':
    if M.shape[0] > 200 and k < 10:
      eigen_solver = 'arpack'
    else:
      eigen_solver = 'dense'

  print(eigen_solver)

  eigen_vectors = np.empty((M.shape[0], k), dtype=np.float)
  eigen_values = np.empty(k, dtype=np.float)

  if eigen_solver == 'arpack':
    random_state = check_random_state(random_state)
    # initialize with [-1,1] as in ARPACK
    v0 = random_state.uniform(-1, 1, M.shape[0])
    try:
      # First iteration
      eigen_values[0], eigvect_temp = eigsh(M, 1, tol=tol, maxiter=max_iter, v0=v0)
      eigen_vectors[:, 0] = eigvect_temp[:, 0]
      if hasattr(M, 'toarray'):
        M = M.toarray()
      for i in range(1, k):
        P = smoothing_matrix(alpha, eigen_vectors[:, :i], i)
        _, s, V = svd(np.concatenate((np.ones((1, P.shape[0])), P), axis=0), full_matrices=True)
        V = V[s < 1e-4 * s[0]]
        eigen_values[i], eigvect_temp = eigsh(np.dot(V, np.dot(M, V.T)), 1, tol=tol)
        eigen_vectors[:, i] = np.dot(V.T, eigvect_temp[:, 0])
        eigen_vectors[:, i] /= np.sqrt(np.sum(eigen_vectors[:, i] ** 2))

    except RuntimeError as msg:

      raise ValueError("Error in determining null-space with ARPACK. "
                       "Error message: '%s'. "
                       "Note that method='arpack' can fail when the "
                       "weight matrix is singular or otherwise "
                       "ill-behaved.  method='dense' is recommended. "
                       "See online documentation for more information."
                       % msg)

    return eigen_values, eigen_vectors

  elif eigen_solver == 'dense':
    if hasattr(M, 'toarray'):
      M = M.toarray()
    # First iteration
    _, eigvect_temp = eigh(M, eigvals=(M.shape[0] - 1, M.shape[0] - 1))
    eigen_vectors[:, 0] = eigvect_temp[:, 0]
    # Subsequent iterations
    for i in range(1, k):
      P = smoothing_matrix(alpha, eigen_vectors[:, :i], i)
      _, s, V = svd(np.concatenate((np.ones((1, P.shape[0])) / P.shape[0], P), axis=0), full_matrices=True)
      V = V[s < 1e-4 * s[0]]
      _, eigvect_temp = eigh(np.dot(V, np.dot(M, V.T)), eigvals=(V.shape[0] - 1, V.shape[0] - 1))
      eigen_vectors[:, i] = np.dot(V.T, eigvect_temp[:, 0])
      eigen_vectors[:, i] /= np.sqrt(np.sum(eigen_vectors[:, i] ** 2))

    return eigen_values, eigen_vectors
  else:
    raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


class NonRedundantIsomap(TransformerMixin, BaseEstimator):

  def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
               tol=0, max_iter=None, path_method='auto',
               neighbors_algorithm='auto', n_jobs=None, metric='minkowski',
               p=2, metric_params=None, alpha=0.6):
    self.n_neighbors = n_neighbors
    self.n_components = n_components
    self.eigen_solver = eigen_solver
    self.tol = tol
    self.max_iter = max_iter
    self.path_method = path_method
    self.neighbors_algorithm = neighbors_algorithm
    self.n_jobs = n_jobs
    self.metric = metric
    self.p = p
    self.metric_params = metric_params
    self.alpha = alpha

  def _fit_transform(self, X):

    self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                  algorithm=self.neighbors_algorithm,
                                  metric=self.metric, p=self.p,
                                  metric_params=self.metric_params,
                                  n_jobs=self.n_jobs)
    self.nbrs_.fit(X)

    # self.kernel_pca_ = KernelPCA(n_components=1,
    #                              kernel="precomputed",
    #                              eigen_solver=self.eigen_solver,
    #                              tol=self.tol, max_iter=self.max_iter,
    #                              n_jobs=self.n_jobs)

    kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                           metric=self.metric, p=self.p,
                           metric_params=self.metric_params,
                           mode='distance', n_jobs=self.n_jobs)

    self.dist_matrix_ = graph_shortest_path(kng,
                                            method=self.path_method,
                                            directed=False)
    G = self.dist_matrix_ ** 2
    G -= (G.mean(axis=1)[:, np.newaxis] + G.mean(axis=0) - G.mean())
    G *= -0.5

    self.lambdas_, self.embedding_ = nred_pca(
      G, self.n_components, alpha=self.alpha, tol=self.tol,
      max_iter=self.max_iter, eigen_solver=self.eigen_solver)

  @deprecated("Attribute `training_data_` was deprecated in version 0.22 and"
              " will be removed in 0.24.")
  @property
  def training_data_(self):
    check_is_fitted(self)
    return self.nbrs_._fit_X

  def reconstruction_error(self):
    """Compute the reconstruction error for the embedding.

    Returns
    -------
    reconstruction_error : float

    Notes
    -----
    The cost function of an isomap embedding is

    ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

    Where D is the matrix of distances for the input data X,
    D_fit is the matrix of distances for the output embedding X_fit,
    and K is the isomap kernel:

    ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
    """
    G = -0.5 * self.dist_matrix_ ** 2
    G_center = KernelCenterer().fit_transform(G)
    evals = self.lambdas_
    return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]

  def fit(self, X, y=None):
    """Compute the embedding vectors for data X

    Parameters
    ----------
    X : {array-like, sparse graph, BallTree, KDTree, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array, sparse graph, precomputed tree, or NearestNeighbors
        object.

    y : Ignored

    Returns
    -------
    self : returns an instance of self.
    """
    self._fit_transform(X)
    return self

  def fit_transform(self, X, y=None):
    """Fit the model from data in X and transform X.

    Parameters
    ----------
    X : {array-like, sparse graph, BallTree, KDTree}
        Training vector, where n_samples in the number of samples
        and n_features is the number of features.

    y : Ignored

    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
    """
    self._fit_transform(X)
    return self.embedding_

  # def transform(self, X):
  #   """Transform X.
  #
  #   This is implemented by linking the points X into the graph of geodesic
  #   distances of the training data. First the `n_neighbors` nearest
  #   neighbors of X are found in the training data, and from these the
  #   shortest geodesic distances from each point in X to each point in
  #   the training data are computed in order to construct the kernel.
  #   The embedding of X is the projection of this kernel onto the
  #   embedding vectors of the training set.
  #
  #   Parameters
  #   ----------
  #   X : array-like, shape (n_queries, n_features)
  #       If neighbors_algorithm='precomputed', X is assumed to be a
  #       distance matrix or a sparse graph of shape
  #       (n_queries, n_samples_fit).
  #
  #   Returns
  #   -------
  #   X_new : array-like, shape (n_queries, n_components)
  #   """
  #   check_is_fitted(self)
  #   distances, indices = self.nbrs_.kneighbors(X, return_distance=True)
  #
  #   # Create the graph of shortest distances from X to
  #   # training data via the nearest neighbors of X.
  #   # This can be done as a single array operation, but it potentially
  #   # takes a lot of memory.  To avoid that, use a loop:
  #
  #   n_samples_fit = self.nbrs_.n_samples_fit_
  #   n_queries = distances.shape[0]
  #   G_X = np.zeros((n_queries, n_samples_fit))
  #   for i in range(n_queries):
  #     G_X[i] = np.min(self.dist_matrix_[indices[i]] +
  #                     distances[i][:, None], 0)
  #
  #   G_X **= 2
  #   G_X *= -0.5
  #
  #   return self.kernel_pca_.transform(G_X)