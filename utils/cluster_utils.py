import numpy as np
from dtaidistance import clustering, dtw_ndim, dtw
from dtaidistance.clustering.kmeans import KMeans
from DBA_multivariate import*

#seq = list of timesseries <-- operations
#returns normalized distance matrix
def n_dtw(seq, window = None):
    N = len(seq)
    distMatrix = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1, N):
            dist, paths = dtw_ndim.warping_paths(seq[i], seq[j], window = window)
            pathLen = len(dtw.best_path(paths))
            normalizedDist = dist/pathLen
            distMatrix[i,j] = normalizedDist 
            distMatrix[j,i] = normalizedDist #making matrix symmetrical
    
    return distMatrix

def assign_centroids(X, centroids, cost_mat, delta_mat, tmp_delta_mat):
    labels = [] 
    for x in X:
        dists = [] #distance to each centroid
        for centroid in centroids:
            dists.append(squared_DTW(x, centroid, cost_mat, delta_mat, tmp_delta_mat))
        
        labels.append(np.argmin(dists))

    labels = np.array(labels)
    return labels


def update_centroids(X, k, labels):
    new_centroids = []
    for i in range(k):
        series_in_cluster = []
        for x, l in zip(X, labels):
            if l == i:
                series_in_cluster.append(x)
        
        new_centroids.append(performDBA(series_in_cluster))

    return new_centroids

def compute_inertia(X, N, labels, centroids, cost_mat, delta_mat, tmp_delta_mat):
    inertia = 0
    for i in range(N):
        inertia += squared_DTW(X[i], centroids[labels[i]], cost_mat, delta_mat, tmp_delta_mat)

    return inertia


def kMeans_nDTW(X, k = 2, n_runs = 10, max_iter=100, random_state = None):
    '''
    X: operations
    k: # clusters
    n_runs: number of times we run algorithm before choosing best result
    max_iter: number of mean in kmeans algorithm
    random_state: param used to init random number generator


    return:
        - clustering labels
        - inertia
    '''

    rng = np.random.RandomState(random_state)
    N = len(X)
    bestInertia = np.inf
    bestLabels = None 

    #allocate matrices for NDTW computations
    max_length = max(x.shape[0] for x in X)
    cost_mat = np.zeros((max_length, max_length))
    delta_mat = np.zeros((max_length, max_length))
    tmp_delta_mat = np.zeros((max_length, max_length))
    path_mat = np.zeros((max_length, max_length), dtype=np.int8)



    for run in range(n_runs):

        #init centroids 
        centroid_indices = rng.choice(N, k, replace=False)
        centroids = [X[i] for i in centroid_indices]
        labels = np.zeros(N)

        for iteration in range(max_iter):
            prev_labels = labels.copy() 

            # assign each operation to nearest centroid (lowest nDTW dist)
            labels = assign_centroids(X, centroids, cost_mat, delta_mat, tmp_delta_mat)

            #update centroid
            centroids = update_centroids(X,k,labels)

            # check convergence
            if np.allclose(labels, prev_labels):
                break


        #compute inertia
        inertia = compute_inertia(X, N, labels, centroids, cost_mat, delta_mat, tmp_delta_mat)


        #updating best inertia 
        if inertia < bestInertia:
            bestInertia = inertia
            bestLabel = labels
    
    return bestInertia, bestLabel

