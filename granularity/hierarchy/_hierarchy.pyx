# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, INFINITY
from libc.string cimport memset
from cpython.mem cimport PyMem_Malloc, PyMem_Free


ctypedef unsigned char uchar

np.import_array()

# _hierarchy_distance_update.pxi includes the definition of linkage_distance_update
# and the distance update functions for the supported linkage methods.
include "_hierarchy_distance_update.pxi"
cdef linkage_distance_update *linkage_methods = [
    _single, _complete, _average, _centroid, _median, _ward, _weighted]

cdef inline np.npy_int64 condensed_index(np.npy_int64 n, np.npy_int64 i,
                                         np.npy_int64 j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)

def nn_chain(double[:] dists, int n, int method):
    """Perform hierarchy clustering using nearest-neighbor chain algorithm.
    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted
    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """
    Z_arr = np.empty((n - 1, 4))
    cdef double[:, :] Z = Z_arr

    cdef double[:] D = dists.copy()  # Distances between clusters.
    cdef int[:] size = np.ones(n, dtype=np.intc)  # Sizes of clusters.

    cdef linkage_distance_update new_dist = linkage_methods[method]

    # Variables to store neighbors chain.
    cdef int[:] cluster_chain = np.ndarray(n, dtype=np.intc)
    cdef int chain_length = 0

    cdef int i, j, k, x, y = 0, nx, ny, ni
    cdef double dist, current_min

    for k in range(n - 1):
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]

            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = INFINITY

            for i in range(n):
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue

            D[condensed_index(n, i, y)] = new_dist(
                D[condensed_index(n, i, x)],
                D[condensed_index(n, i, y)],
                current_min, nx, ny, ni)

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]

    # Find correct cluster labels inplace.
    label(Z_arr, n)

    return Z_arr

def nn_chain_from_middle(double[:] dists, int[:] size, int n, int method):
    """Perform hierarchy clustering using nearest-neighbor chain algorithm.
    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the clusters.
    size : ndarray
        An array that saves the sizes of clusters
    n: int
        The number of input clusters.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted
    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """
    Z_arr = np.empty((n - 1, 4))
    cdef double[:, :] Z = Z_arr

    cdef double[:] D = dists.copy()  # Distances between clusters.
    cdef int[:] S = size.copy()  # Sizes of clusters.

    cdef linkage_distance_update new_dist = linkage_methods[method]

    # Variables to store neighbors chain.
    cdef int[:] cluster_chain = np.ndarray(n, dtype=np.intc)
    cdef int chain_length = 0

    cdef int i, j, k, x, y = 0, nx, ny, ni
    cdef double dist, current_min

    # still n-1 steps before merging into single cluster
    for k in range(n - 1):
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if S[i] > 0:
                    cluster_chain[0] = i
                    break
        
        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]

            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = INFINITY
            
            for i in range(n):
                if S[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i
            
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = y
            chain_length += 1
        
        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = S[x]
        ny = S[y]

        # Record the new node.
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        S[x] = 0  # Cluster x will be dropped.
        S[y] = nx + ny  # Cluster y will be replaced with the new cluster

        # Update the distance matrix.
        for i in range(n):
            ni = S[i]
            if ni == 0 or i == y:
                continue

            D[condensed_index(n, i, y)] = new_dist(
                D[condensed_index(n, i, x)],
                D[condensed_index(n, i, y)],
                current_min, nx, ny, ni)
    
    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]

    # Find correct cluster labels inplace.
    label_from_middle(Z_arr, n, size)

    return Z_arr

cdef class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    cdef int[:] parent
    cdef int[:] size
    cdef int next_label

    def __init__(self, int n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    cdef int merge(self, int x, int y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        cdef int size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    cdef find(self, int x):
        cdef int p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


cdef class LinkageUnionFindFromMiddle:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    cdef int[:] parent
    cdef int[:] size
    cdef int next_label

    def __init__(self, int n, int[:] ini_size):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)
        for i in range(n):
            self.size[i] = ini_size[i]

    cdef int merge(self, int x, int y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        cdef int size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    cdef find(self, int x):
        cdef int p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


cdef label(double[:, :] Z, int n):
    """Correctly label clusters in unsorted dendrogram."""
    cdef LinkageUnionFind uf = LinkageUnionFind(n)
    cdef int i, x, y, x_root, y_root
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)

cdef label_from_middle(double[:, :] Z, int n, int[:] size):
    """Correctly label clusters in unsorted dendrogram."""
    cdef LinkageUnionFindFromMiddle uf = LinkageUnionFindFromMiddle(n, size)
    cdef int i, x, y, x_root, y_root
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)