# combine minibatch kmeans and agglomerative clustering
# it first clusters data into a maximum number of clusters and then perform agglomerative clustering
# Because agglomerative clustering is usually much slower than minibatch kmeans.

import numpy as np
from hierarchy import _hierarchy

# Base class for all estimators in scikit-learn.
from sklearn.base import BaseEstimator
# Mixin class for all cluster estimators in scikit-learn.
from sklearn.base import ClusterMixin
# Constraint representing a typed interval.
from sklearn.utils._param_validation import Interval
from numbers import Integral

from sklearn.cluster import MiniBatchKMeans

class KMeansAgglomerativeClustering(ClusterMixin, BaseEstimator):
    """
    Kmeans before Agglomerative Clustering.

    Use minibatch-kmeans to cluster into a maximum number of clusters.

    And then use Agglomerative Clustering to recursively merge pair of clusters.

    Parameters
    ----------
    max_n_clusters : int or None, default=200
        The maximum number of clusters for kmeans. It must be larger than ``n_clusters``
        if ``n_clusters`` is not None.
    batch_size : int, default=1024
        Size of the mini batches for kmeans.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.
        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.
    """

    _parameter_constraints: dict = {
        "max_n_clusters": [Interval(Integral, 1, None, closed="left"), None],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        max_n_clusters=200,
        *,
        batch_size=1024,
        random_state=None,
    ):
        self.max_n_clusters = max_n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit the hierarchical clustering from features, or distance matrix.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        self._validate_params()
        X = self._validate_data(X, ensure_min_samples=2)
        return self._fit(X)
    
    def _fit(self, X):
        """Fit without validation
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.
        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        kmeans = MiniBatchKMeans(n_clusters=self.max_n_clusters, batch_size=self.batch_size, random_state=self.random_state)
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_
        self.kmeans_preds_ = kmeans.predict(X)

        unique_preds = sorted(list(set(self.kmeans_preds_)))
        sizes = []
        for up in unique_preds:
            sizes.append((self.kmeans_preds_ == up).sum())
        
        pdist = np.zeros(int(self.max_n_clusters * (self.max_n_clusters - 1) / 2))
        for i in range(self.max_n_clusters):
            for j in range(self.max_n_clusters):
                if i == j:
                    continue
                cidx = condensed_index(self.max_n_clusters, i, j)
                pdist[cidx] = ((self.centers_[i] - self.centers_[j]) ** 2).sum()
                pdist[cidx] *= sizes[i] * sizes[j] / (sizes[i] + sizes[j])
        
        # breakpoint()
        out = _hierarchy.nn_chain_from_middle(pdist, np.asarray(sizes, dtype=np.int32), self.max_n_clusters, 5)

        self.children_ = out[:, :2].astype(np.intp)
        return self
        

def condensed_index(n, i, j):
    if i < j:
        return int(n * i - (i * (i + 1) / 2) + (j - i - 1))
    elif i > j:
        return int(n * j - (j * (j + 1) / 2) + (i - j - 1))