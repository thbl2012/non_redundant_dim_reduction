from sklearn.manifold._spectral_embedding import _graph_connected_component, _set_diag, _graph_is_connected
from sklearn.manifold._spectral_embedding import *
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from scipy.linalg.decomp_svd import svd
from nred_lle import smoothing_matrix

"""Spectral Embedding"""


# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause


def nred_spectral_embedding(laplacian, dd, n_components=8, eigen_solver=None,
                            random_state=None, eigen_tol=0.0,
                            norm_laplacian=True, drop_first=True):
  try:
    from pyamg import smoothed_aggregation_solver
  except ImportError:
    if eigen_solver == "amg":
      raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                       "not available.")

  if eigen_solver is None:
    eigen_solver = 'arpack'
  elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
    raise ValueError("Unknown value for eigen_solver: '%s'."
                     "Should be 'amg', 'arpack', or 'lobpcg'"
                     % eigen_solver)

  random_state = check_random_state(random_state)

  n_nodes = laplacian.shape[0]
  # Whether to drop the first eigenvector
  if drop_first:
    n_components = n_components + 1

  if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
    (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
    # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
    # for details see the source code in scipy:
    # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
    # /lobpcg/lobpcg.py#L237
    # or matlab:
    # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
    laplacian = _set_diag(laplacian, 1, norm_laplacian)

    # Here we'll use shift-invert mode for fast eigenvalues
    # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    #  for a short explanation of what this means)
    # Because the normalized Laplacian has eigenvalues between 0 and 2,
    # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
    # when finding eigenvalues of largest magnitude (keyword which='LM')
    # and when these eigenvalues are very large compared to the rest.
    # For very large, very sparse graphs, I - L can have many, many
    # eigenvalues very near 1.0.  This leads to slow convergence.  So
    # instead, we'll use ARPACK's shift-invert mode, asking for the
    # eigenvalues near 1.0.  This effectively spreads-out the spectrum
    # near 1.0 and leads to much faster convergence: potentially an
    # orders-of-magnitude speedup over simply using keyword which='LA'
    # in standard mode.
    try:
      # We are computing the opposite of the laplacian inplace so as
      # to spare a memory allocation of a possibly very large array
      laplacian *= -1

      v0 = random_state.uniform(-1, 1, laplacian.shape[0])
      eigvals, diffusion_map = eigsh(
        laplacian, k=n_components, sigma=1.0, which='LM',
        tol=eigen_tol, v0=v0)
      embedding = diffusion_map.T[n_components::-1]
      if norm_laplacian:
        embedding = embedding / dd
    except RuntimeError:
      # When submatrices are exactly singular, an LU decomposition
      # in arpack fails. We fallback to lobpcg
      eigen_solver = "lobpcg"
      # Revert the laplacian to its opposite to have lobpcg work
      laplacian *= -1

  elif eigen_solver == 'amg':
    # Use AMG to get a preconditioner and speed up the eigenvalue
    # problem.
    if not sparse.issparse(laplacian):
      warnings.warn("AMG works better for sparse matrices")
    # lobpcg needs double precision floats
    laplacian = check_array(laplacian, dtype=np.float64,
                            accept_sparse=True)
    laplacian = _set_diag(laplacian, 1, norm_laplacian)

    # The Laplacian matrix is always singular, having at least one zero
    # eigenvalue, corresponding to the trivial eigenvector, which is a
    # constant. Using a singular matrix for preconditioning may result in
    # random failures in LOBPCG and is not supported by the existing
    # theory:
    #     see https://doi.org/10.1007/s10208-015-9297-1
    # Shift the Laplacian so its diagononal is not all ones. The shift
    # does change the eigenpairs however, so we'll feed the shifted
    # matrix to the solver and afterward set it back to the original.
    diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
    laplacian += diag_shift
    ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
    laplacian -= diag_shift

    M = ml.aspreconditioner()
    X = random_state.rand(laplacian.shape[0], n_components + 1)
    X[:, 0] = dd.ravel()
    _, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-5,
                              largest=False)
    embedding = diffusion_map.T
    if norm_laplacian:
      embedding = embedding / dd
    if embedding.shape[0] == 1:
      raise ValueError

  if eigen_solver == "lobpcg":
    # lobpcg needs double precision floats
    laplacian = check_array(laplacian, dtype=np.float64,
                            accept_sparse=True)
    if n_nodes < 5 * n_components + 1:
      # see note above under arpack why lobpcg has problems with small
      # number of nodes
      # lobpcg will fallback to eigh, so we short circuit it
      if sparse.isspmatrix(laplacian):
        laplacian = laplacian.toarray()
      _, diffusion_map = eigh(laplacian)
      embedding = diffusion_map.T[:n_components]
      if norm_laplacian:
        embedding = embedding / dd
    else:
      laplacian = _set_diag(laplacian, 1, norm_laplacian)
      # We increase the number of eigenvectors requested, as lobpcg
      # doesn't behave well in low dimension
      X = random_state.rand(laplacian.shape[0], n_components + 1)
      X[:, 0] = dd.ravel()
      _, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                largest=False, maxiter=2000)
      embedding = diffusion_map.T[:n_components]
      if norm_laplacian:
        embedding = embedding / dd
      if embedding.shape[0] == 1:
        raise ValueError

  embedding = _deterministic_vector_sign_flip(embedding)
  if drop_first:
    return embedding[1:n_components].T
  else:
    return embedding[:n_components].T


class NonRedundantSpectralEmbedding(BaseEstimator):
  """Spectral embedding for non-linear dimensionality reduction.

  Forms an affinity matrix given by the specified function and
  applies spectral decomposition to the corresponding graph laplacian.
  The resulting transformation is given by the value of the
  eigenvectors for each data point.

  Note : Laplacian Eigenmaps is the actual algorithm implemented here.

  Read more in the :ref:`User Guide <spectral_embedding>`.

  Parameters
  ----------
  n_components : integer, default: 2
      The dimension of the projected subspace.

  affinity : string or callable, default : "nearest_neighbors"
      How to construct the affinity matrix.
       - 'nearest_neighbors' : construct the affinity matrix by computing a
         graph of nearest neighbors.
       - 'rbf' : construct the affinity matrix by computing a radial basis
         function (RBF) kernel.
       - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
       - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
         of precomputed nearest neighbors, and constructs the affinity matrix
         by selecting the ``n_neighbors`` nearest neighbors.
       - callable : use passed in function as affinity
         the function takes in data matrix (n_samples, n_features)
         and return affinity matrix (n_samples, n_samples).

  gamma : float, optional, default : 1/n_features
      Kernel coefficient for rbf kernel.

  random_state : int, RandomState instance or None, optional, default: None
      A pseudo random number generator used for the initialization of the
      lobpcg eigenvectors.  If int, random_state is the seed used by the
      random number generator; If RandomState instance, random_state is the
      random number generator; If None, the random number generator is the
      RandomState instance used by `np.random`. Used when ``solver`` ==
      'amg'.

  eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
      The eigenvalue decomposition strategy to use. AMG requires pyamg
      to be installed. It can be faster on very large, sparse problems.

  n_neighbors : int, default : max(n_samples/10 , 1)
      Number of nearest neighbors for nearest_neighbors graph building.

  n_jobs : int or None, optional (default=None)
      The number of parallel jobs to run.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
      for more details.

  Attributes
  ----------

  embedding_ : array, shape = (n_samples, n_components)
      Spectral embedding of the training matrix.

  affinity_matrix_ : array, shape = (n_samples, n_samples)
      Affinity_matrix constructed from samples or precomputed.

  n_neighbors_ : int
      Number of nearest neighbors effectively used.

  Examples
  --------
  >>> from sklearn.datasets import load_digits
  >>> from sklearn.manifold import SpectralEmbedding
  >>> X, _ = load_digits(return_X_y=True)
  >>> X.shape
  (1797, 64)
  >>> embedding = SpectralEmbedding(n_components=2)
  >>> X_transformed = embedding.fit_transform(X[:100])
  >>> X_transformed.shape
  (100, 2)

  References
  ----------

  - A Tutorial on Spectral Clustering, 2007
    Ulrike von Luxburg
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

  - On Spectral Clustering: Analysis and an algorithm, 2001
    Andrew Y. Ng, Michael I. Jordan, Yair Weiss
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

  - Normalized cuts and image segmentation, 2000
    Jianbo Shi, Jitendra Malik
    http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
  """

  def __init__(self, n_components=2, affinity="nearest_neighbors",
               gamma=None, random_state=None, eigen_solver=None,
               n_neighbors=None, n_jobs=None, alpha=0.5):
    self.n_components = n_components
    self.affinity = affinity
    self.gamma = gamma
    self.random_state = random_state
    self.eigen_solver = eigen_solver
    self.n_neighbors = n_neighbors
    self.n_jobs = n_jobs
    self.alpha = alpha

  @property
  def _pairwise(self):
    return self.affinity in ["precomputed",
                             "precomputed_nearest_neighbors"]

  def _get_affinity_matrix(self, X, Y=None):
    """Calculate the affinity matrix from data
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        and n_features is the number of features.

        If affinity is "precomputed"
        X : array-like, shape (n_samples, n_samples),
        Interpret X as precomputed adjacency graph computed from
        samples.

    Y: Ignored

    Returns
    -------
    affinity_matrix, shape (n_samples, n_samples)
    """
    if self.affinity == 'precomputed':
      self.affinity_matrix_ = X
      return self.affinity_matrix_
    if self.affinity == 'precomputed_nearest_neighbors':
      estimator = NearestNeighbors(n_neighbors=self.n_neighbors,
                                   n_jobs=self.n_jobs,
                                   metric="precomputed").fit(X)
      connectivity = estimator.kneighbors_graph(X=X, mode='connectivity')
      self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
      return self.affinity_matrix_
    if self.affinity == 'nearest_neighbors':
      if sparse.issparse(X):
        warnings.warn("Nearest neighbors affinity currently does "
                      "not support sparse input, falling back to "
                      "rbf affinity")
        self.affinity = "rbf"
      else:
        self.n_neighbors_ = (self.n_neighbors
                             if self.n_neighbors is not None
                             else max(int(X.shape[0] / 10), 1))
        self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_,
                                                 include_self=True,
                                                 n_jobs=self.n_jobs)
        # currently only symmetric affinity_matrix supported
        self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ +
                                       self.affinity_matrix_.T)
        return self.affinity_matrix_
    if self.affinity == 'rbf':
      self.gamma_ = (self.gamma
                     if self.gamma is not None else 1.0 / X.shape[1])
      self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
      return self.affinity_matrix_
    self.affinity_matrix_ = self.affinity(X)
    return self.affinity_matrix_

  def fit(self, X, y=None):
    """Fit the model from data in X.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        and n_features is the number of features.

        If affinity is "precomputed"
        X : {array-like, sparse matrix}, shape (n_samples, n_samples),
        Interpret X as precomputed adjacency graph computed from
        samples.

    Returns
    -------
    self : object
        Returns the instance itself.
    """

    X = check_array(X, accept_sparse='csr', ensure_min_samples=2,
                    estimator=self)

    random_state = check_random_state(self.random_state)
    if isinstance(self.affinity, str):
      if self.affinity not in {"nearest_neighbors", "rbf", "precomputed",
                               "precomputed_nearest_neighbors"}:
        raise ValueError(("%s is not a valid affinity. Expected "
                          "'precomputed', 'rbf', 'nearest_neighbors' "
                          "or a callable.") % self.affinity)
    elif not callable(self.affinity):
      raise ValueError(("'affinity' is expected to be an affinity "
                        "name or a callable. Got: %s") % self.affinity)

    affinity_matrix = check_symmetric(self._get_affinity_matrix(X))

    if not _graph_is_connected(affinity_matrix):
      warnings.warn("Graph is not fully connected, spectral embedding"
                    " may not work as expected.")

    laplacian, dd = csgraph_laplacian(affinity_matrix, normed=True,
                                      return_diag=True)
    eigen_vectors = np.empty((affinity_matrix.shape[0], self.n_components))
    # First iteration
    eigen_vectors[:, :1] = nred_spectral_embedding(
      laplacian, dd, n_components=1,
      eigen_solver=self.eigen_solver, random_state=random_state
    )
    # Subsequence iterations
    if hasattr(laplacian, 'toarray'):
      laplacian = laplacian.toarray()
    for i in range(1, self.n_components):
      P = smoothing_matrix(self.alpha, eigen_vectors[:, :i], i)
      _, s, V = svd(np.concatenate((np.ones((1, P.shape[0])), P), axis=0), full_matrices=True)
      V = V[s < 1e-4 * s[0]]
      eigval_temp, eigvect_temp = eigsh(np.dot(V, np.dot(laplacian, V.T)), 1, sigma=0.0)
      # eigen_values[i] = eigval_temp[0]
      eigen_vectors[:, i] = np.dot(V.T, eigvect_temp[:, 0]) / np.sqrt(eigval_temp)

    self.embedding_ = eigen_vectors

    return self

  def fit_transform(self, X, y=None):
    """Fit the model from data in X and transform X.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        and n_features is the number of features.

        If affinity is "precomputed"
        X : {array-like, sparse matrix}, shape (n_samples, n_samples),
        Interpret X as precomputed adjacency graph computed from
        samples.

    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
    """
    self.fit(X)
    return self.embedding_
