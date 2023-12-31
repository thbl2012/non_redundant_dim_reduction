from sklearn.manifold._locally_linear import *
from sklearn.base import BaseEstimator, TransformerMixin, _UnstableArchMixin
from utils import *


def nred_null_space(M, k, alpha=0.6, eigen_solver='arpack', tol=1E-6, max_iter=100,
                    random_state=None):
  if eigen_solver == 'auto':
    if M.shape[0] > 200 and k < 10:
      eigen_solver = 'arpack'
    else:
      eigen_solver = 'dense'

  print(eigen_solver)

  eigen_values = np.empty(k, dtype=np.float)
  eigen_vectors = np.empty((M.shape[0], k), dtype=np.float)
  eigmax, _ = eigsh(M, 1, tol=tol, maxiter=max_iter)

  if eigen_solver == 'arpack':
    random_state = check_random_state(random_state)
    # initialize with [-1,1] as in ARPACK
    v0 = random_state.uniform(-1, 1, M.shape[0])
    try:
      # First iteration
      eigvals_temp, eigvect_temp = eigsh(M, 2, sigma=0.0, tol=tol, maxiter=max_iter, v0=v0)
      eigen_vectors[:, 0] = eigvect_temp[:, 1]
      eigen_values[0] = eigvals_temp[1]

      # Subsequent iterations
      # Set M = eigmax * I - M
      # M = identity(M.shape[0], dtype=float, format='csr').multiply(eigmax[0]) - M

      if hasattr(M, 'toarray'):
        M = M.toarray()
      for i in range(1, k):
        P = smoothing_matrix(alpha, eigen_vectors[:, :i], i)
        _, s, V = svd(np.concatenate(
                      (np.ones((1, P.shape[0])) / P.shape[0], P), axis=0),
                      full_matrices=True)
        V = V[s < 1e-4 * s[0]]
        eigen_values[i], eigvect_temp = eigsh(np.dot(V, np.dot(M, V.T)), 1, sigma=0.0, tol=tol)
        eigen_vectors[:, i] = np.dot(V.T, eigvect_temp[:, 0])
        eigen_vectors[:, i] /= np.sqrt(np.sum(eigen_vectors[:, i] ** 2))

        # _, s, V = svd(P, full_matrices=False)
        # V = V[s > 1e-2 * s[0]]
        # temp = - np.dot(V.T, V)
        # temp[::temp.shape[0] + 1] += 1
        # eigval_temp, eigvect_temp = eigsh(np.dot(temp, np.dot(M, temp)), 1, tol=tol)
        # eigen_values[i] = eigmax - eigval_temp[0]
        # eigen_vectors[:, i] = eigvect_temp[:, 0]

    except RuntimeError as msg:

      raise ValueError("Error in determining null-space with ARPACK. "
                       "Error message: '%s'. "
                       "Note that method='arpack' can fail when the "
                       "weight matrix is singular or otherwise "
                       "ill-behaved.  method='dense' is recommended. "
                       "See online documentation for more information."
                       % msg)

    return eigen_vectors, np.sum(eigen_values)

  elif eigen_solver == 'dense':
    if hasattr(M, 'toarray'):
      M = M.toarray()
    # First iteration
    eigvals_temp, eigvect_temp = eigh(M, eigvals=(0, 1))
    eigen_vectors[:, 0] = eigvect_temp[:, 1]
    eigen_values[0] = eigvals_temp[1]
    # Subsequent iterations
    for i in range(1, k):
      P = smoothing_matrix(alpha, eigen_vectors[:, :i], i)
      _, s, V = svd(np.concatenate((np.ones((1, P.shape[0])), P), axis=0), full_matrices=True)
      V = V[s < 1e-4 * s[0]]
      eigen_values[i], eigvect_temp = eigh(np.dot(V, np.dot(M, V.T)), eigvals=(0, 0))
      eigen_vectors[:, i] = np.dot(V.T, eigvect_temp[:, 0])
      eigen_vectors[:, i] /= np.sqrt(np.sum(eigen_vectors[:, i] ** 2))

    return eigen_vectors, np.sum(eigen_values)
  else:
    raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


def nred_locally_linear_embedding(
  X, n_neighbors, n_components, alpha=0.3, reg=1e-3, eigen_solver='auto', tol=1e-6,
  max_iter=100, method='standard', hessian_tol=1E-4, modified_tol=1E-12,
  random_state=None, n_jobs=None):

  if eigen_solver not in ('auto', 'arpack', 'dense'):
    raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

  if method not in ('standard', 'hessian', 'modified', 'ltsa'):
    raise ValueError("unrecognized method '%s'" % method)

  nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
  nbrs.fit(X)
  X = nbrs._fit_X
  N, d_in = X.shape

  if n_components > d_in:
    raise ValueError("output dimension must be less than or equal "
                     "to input dimension")
  if n_neighbors >= N:
    raise ValueError(
      "Expected n_neighbors <= n_samples, "
      " but n_samples = %d, n_neighbors = %d" %
      (N, n_neighbors)
    )
  if n_neighbors <= 0:
    raise ValueError("n_neighbors must be positive")

  M_sparse = (eigen_solver != 'dense')
  if method == 'standard':
    W = barycenter_kneighbors_graph(
      nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
    # we'll compute M = (I-W)'(I-W)
    # depending on the solver, we'll do this differently
    if M_sparse:
      M = eye(*W.shape, format=W.format) - W
      M = (M.T * M).tocsr()
    else:
      M = (W.T * W - W.T - W).toarray()
      M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

  elif method == 'hessian':
    dp = n_components * (n_components + 1) // 2

    if n_neighbors <= n_components + dp:
      raise ValueError("for method='hessian', n_neighbors must be "
                       "greater than "
                       "[n_components * (n_components + 3) / 2]")

    neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1,
                                return_distance=False)
    neighbors = neighbors[:, 1:]
    Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
    Yi[:, 0] = 1
    M = np.zeros((N, N), dtype=np.float64)
    use_svd = (n_neighbors > d_in)

    for i in range(N):
      Gi = X[neighbors[i]]
      Gi -= Gi.mean(0)
      # build Hessian estimator
      if use_svd:
        U = svd(Gi, full_matrices=0)[0]
      else:
        Ci = np.dot(Gi, Gi.T)
        U = eigh(Ci)[1][:, ::-1]

      Yi[:, 1:1 + n_components] = U[:, :n_components]

      j = 1 + n_components
      for k in range(n_components):
        Yi[:, j:j + n_components - k] = (U[:, k:k + 1] *
                                         U[:, k:n_components])
        j += n_components - k

      Q, R = qr(Yi)

      w = Q[:, n_components + 1:]
      S = w.sum(0)

      S[np.where(abs(S) < hessian_tol)] = 1
      w /= S

      nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
      M[nbrs_x, nbrs_y] += np.dot(w, w.T)

    if M_sparse:
      M = csr_matrix(M)

  elif method == 'modified':
    if n_neighbors < n_components:
      raise ValueError("modified LLE requires "
                       "n_neighbors >= n_components")

    neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1,
                                return_distance=False)
    neighbors = neighbors[:, 1:]

    # find the eigenvectors and eigenvalues of each local covariance
    # matrix. We want V[i] to be a [n_neighbors x n_neighbors] matrix,
    # where the columns are eigenvectors
    V = np.zeros((N, n_neighbors, n_neighbors))
    nev = min(d_in, n_neighbors)
    evals = np.zeros([N, nev])

    # choose the most efficient way to find the eigenvectors
    use_svd = (n_neighbors > d_in)

    if use_svd:
      for i in range(N):
        X_nbrs = X[neighbors[i]] - X[i]
        V[i], evals[i], _ = svd(X_nbrs, full_matrices=True)
      evals **= 2
    else:
      for i in range(N):
        X_nbrs = X[neighbors[i]] - X[i]
        C_nbrs = np.dot(X_nbrs, X_nbrs.T)
        evi, vi = eigh(C_nbrs)
        evals[i] = evi[::-1]
        V[i] = vi[:, ::-1]

    # find regularized weights: this is like normal LLE.
    # because we've already computed the SVD of each covariance matrix,
    # it's faster to use this rather than np.linalg.solve
    reg = 1E-3 * evals.sum(1)

    tmp = np.dot(V.transpose(0, 2, 1), np.ones(n_neighbors))
    tmp[:, :nev] /= evals + reg[:, None]
    tmp[:, nev:] /= reg[:, None]

    w_reg = np.zeros((N, n_neighbors))
    for i in range(N):
      w_reg[i] = np.dot(V[i], tmp[i])
    w_reg /= w_reg.sum(1)[:, None]

    # calculate eta: the median of the ratio of small to large eigenvalues
    # across the points.  This is used to determine s_i, below
    rho = evals[:, n_components:].sum(1) / evals[:, :n_components].sum(1)
    eta = np.median(rho)

    # find s_i, the size of the "almost null space" for each point:
    # this is the size of the largest set of eigenvalues
    # such that Sum[v; v in set]/Sum[v; v not in set] < eta
    s_range = np.zeros(N, dtype=int)
    evals_cumsum = stable_cumsum(evals, 1)
    eta_range = evals_cumsum[:, -1:] / evals_cumsum[:, :-1] - 1
    for i in range(N):
      s_range[i] = np.searchsorted(eta_range[i, ::-1], eta)
    s_range += n_neighbors - nev  # number of zero eigenvalues

    # Now calculate M.
    # This is the [N x N] matrix whose null space is the desired embedding
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
      s_i = s_range[i]

      # select bottom s_i eigenvectors and calculate alpha
      Vi = V[i, :, n_neighbors - s_i:]
      alpha_i = np.linalg.norm(Vi.sum(0)) / np.sqrt(s_i)

      # compute Householder matrix which satisfies
      #  Hi*Vi.T*ones(n_neighbors) = alpha_i*ones(s)
      # using prescription from paper
      h = np.full(s_i, alpha_i) - np.dot(Vi.T, np.ones(n_neighbors))

      norm_h = np.linalg.norm(h)
      if norm_h < modified_tol:
        h *= 0
      else:
        h /= norm_h

      # Householder matrix is
      #  >> Hi = np.identity(s_i) - 2*np.outer(h,h)
      # Then the weight matrix is
      #  >> Wi = np.dot(Vi,Hi) + (1-alpha_i) * w_reg[i,:,None]
      # We do this much more efficiently:
      Wi = (Vi - 2 * np.outer(np.dot(Vi, h), h) +
            (1 - alpha_i) * w_reg[i, :, None])

      # Update M as follows:
      # >> W_hat = np.zeros( (N,s_i) )
      # >> W_hat[neighbors[i],:] = Wi
      # >> W_hat[i] -= 1
      # >> M += np.dot(W_hat,W_hat.T)
      # We can do this much more efficiently:
      nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
      M[nbrs_x, nbrs_y] += np.dot(Wi, Wi.T)
      Wi_sum1 = Wi.sum(1)
      M[i, neighbors[i]] -= Wi_sum1
      M[neighbors[i], i] -= Wi_sum1
      M[i, i] += s_i

    if M_sparse:
      M = csr_matrix(M)

  elif method == 'ltsa':
    neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1,
                                return_distance=False)
    neighbors = neighbors[:, 1:]

    M = np.zeros((N, N))

    use_svd = (n_neighbors > d_in)

    for i in range(N):
      Xi = X[neighbors[i]]
      Xi -= Xi.mean(0)

      # compute n_components largest eigenvalues of Xi * Xi^T
      if use_svd:
        v = svd(Xi, full_matrices=True)[0]
      else:
        Ci = np.dot(Xi, Xi.T)
        v = eigh(Ci)[1][:, ::-1]

      Gi = np.zeros((n_neighbors, n_components + 1))
      Gi[:, 1:] = v[:, :n_components]
      Gi[:, 0] = 1. / np.sqrt(n_neighbors)

      GiGiT = np.dot(Gi, Gi.T)

      nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
      M[nbrs_x, nbrs_y] -= GiGiT
      M[neighbors[i], neighbors[i]] += 1

  return nred_null_space(M, n_components, alpha=alpha, eigen_solver=eigen_solver,
                         tol=tol, max_iter=max_iter, random_state=random_state)


class NonRedundantLocallyLinearEmbedding(
  TransformerMixin, _UnstableArchMixin, BaseEstimator):
  """Locally Linear Embedding

  Read more in the :ref:`User Guide <locally_linear_embedding>`.

  Parameters
  ----------
  n_neighbors : integer
      number of neighbors to consider for each point.

  n_components : integer
      number of coordinates for the manifold

  reg : float
      regularization constant, multiplies the trace of the local covariance
      matrix of the distances.

  eigen_solver : string, {'auto', 'arpack', 'dense'}
      auto : algorithm will attempt to choose the best method for input data

      arpack : use arnoldi iteration in shift-invert mode.
                  For this method, M may be a dense matrix, sparse matrix,
                  or general linear operator.
                  Warning: ARPACK can be unstable for some problems.  It is
                  best to try several random seeds in order to check results.

      dense  : use standard dense matrix operations for the eigenvalue
                  decomposition.  For this method, M must be an array
                  or matrix type.  This method should be avoided for
                  large problems.

  tol : float, optional
      Tolerance for 'arpack' method
      Not used if eigen_solver=='dense'.

  max_iter : integer
      maximum number of iterations for the arpack solver.
      Not used if eigen_solver=='dense'.

  method : string ('standard', 'hessian', 'modified' or 'ltsa')
      standard : use the standard locally linear embedding algorithm.  see
                 reference [1]
      hessian  : use the Hessian eigenmap method. This method requires
                 ``n_neighbors > n_components * (1 + (n_components + 1) / 2``
                 see reference [2]
      modified : use the modified locally linear embedding algorithm.
                 see reference [3]
      ltsa     : use local tangent space alignment algorithm
                 see reference [4]

  hessian_tol : float, optional
      Tolerance for Hessian eigenmapping method.
      Only used if ``method == 'hessian'``

  modified_tol : float, optional
      Tolerance for modified LLE method.
      Only used if ``method == 'modified'``

  neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
      algorithm to use for nearest neighbors search,
      passed to neighbors.NearestNeighbors instance

  random_state : int, RandomState instance or None, optional (default=None)
      If int, random_state is the seed used by the random number generator;
      If RandomState instance, random_state is the random number generator;
      If None, the random number generator is the RandomState instance used
      by `np.random`. Used when ``eigen_solver`` == 'arpack'.

  n_jobs : int or None, optional (default=None)
      The number of parallel jobs to run.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
      for more details.

  Attributes
  ----------
  embedding_ : array-like, shape [n_samples, n_components]
      Stores the embedding vectors

  reconstruction_error_ : float
      Reconstruction error associated with `embedding_`

  nbrs_ : NearestNeighbors object
      Stores nearest neighbors instance, including BallTree or KDtree
      if applicable.

  Examples
  --------
  >>> from sklearn.datasets import load_digits
  >>> from sklearn.manifold import LocallyLinearEmbedding
  >>> X, _ = load_digits(return_X_y=True)
  >>> X.shape
  (1797, 64)
  >>> embedding = LocallyLinearEmbedding(n_components=2)
  >>> X_transformed = embedding.fit_transform(X[:100])
  >>> X_transformed.shape
  (100, 2)

  References
  ----------

  .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
      by locally linear embedding.  Science 290:2323 (2000).
  .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
      linear embedding techniques for high-dimensional data.
      Proc Natl Acad Sci U S A.  100:5591 (2003).
  .. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
      Embedding Using Multiple Weights.
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
  .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
      dimensionality reduction via tangent space alignment.
      Journal of Shanghai Univ.  8:406 (2004)
  """

  def __init__(self, n_neighbors=5, n_components=2, reg=1E-3,
               eigen_solver='auto', tol=1E-6, max_iter=100,
               method='standard', hessian_tol=1E-4, modified_tol=1E-12,
               neighbors_algorithm='auto', random_state=None, n_jobs=None):
    self.n_neighbors = n_neighbors
    self.n_components = n_components
    self.reg = reg
    self.eigen_solver = eigen_solver
    self.tol = tol
    self.max_iter = max_iter
    self.method = method
    self.hessian_tol = hessian_tol
    self.modified_tol = modified_tol
    self.random_state = random_state
    self.neighbors_algorithm = neighbors_algorithm
    self.n_jobs = n_jobs

  def _fit_transform(self, X, alpha=0.3):
    self.nbrs_ = NearestNeighbors(self.n_neighbors,
                                  algorithm=self.neighbors_algorithm,
                                  n_jobs=self.n_jobs)

    random_state = check_random_state(self.random_state)
    X = check_array(X, dtype=float)
    self.nbrs_.fit(X)
    self.embedding_, self.reconstruction_error_ = (
      nred_locally_linear_embedding(
        self.nbrs_, self.n_neighbors, self.n_components, alpha=alpha,
        eigen_solver=self.eigen_solver, tol=self.tol,
        max_iter=self.max_iter, method=self.method,
        hessian_tol=self.hessian_tol, modified_tol=self.modified_tol,
        random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)
    )

  def fit(self, X, y=None, alpha=0.3):
    """Compute the embedding vectors for data X

    Parameters
    ----------
    X : array-like of shape [n_samples, n_features]
        training set.

    y : Ignored

    Returns
    -------
    self : returns an instance of self.
    """
    self._fit_transform(X, alpha=alpha)
    return self

  def fit_transform(self, X, y=None, alpha=0.3):
    """Compute the embedding vectors for data X and transform X.

    Parameters
    ----------
    X : array-like of shape [n_samples, n_features]
        training set.

    y : Ignored

    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
    """
    self._fit_transform(X, alpha=alpha)
    return self.embedding_

  def transform(self, X):
    """
    Transform new points into embedding space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    X_new : array, shape = [n_samples, n_components]

    Notes
    -----
    Because of scaling performed by this method, it is discouraged to use
    it together with methods that are not scale-invariant (like SVMs)
    """
    check_is_fitted(self)

    X = check_array(X)
    ind = self.nbrs_.kneighbors(X, n_neighbors=self.n_neighbors,
                                return_distance=False)
    weights = barycenter_weights(X, self.nbrs_._fit_X[ind],
                                 reg=self.reg)
    X_new = np.empty((X.shape[0], self.n_components))
    for i in range(X.shape[0]):
      X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
    return X_new