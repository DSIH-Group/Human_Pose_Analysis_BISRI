'''
/*******************************************************************************
 * Copyright (C) 2018 Francois Petitjean
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


__author__ ="Francois Petitjean"

# Building on top of multivariate DBA implementation to create a normalized version: NDBA
# assumes series are shaped (length, n_dims)

'''
Building on top of multivariate DBA implementation to create a normalized version: NDBA

Input:
- series: list of time series u want to average
    - assumes series are shaped (length, n_dims)
- n_iterations: number of iterations of NDBA algorithm

Return:
- Average of time series'
'''
def performDBA(series, n_iterations=10):
    n_series = len(series)

    max_length = 0 #longest series 

    for s in series:
        max_length = max(max_length,s.shape[0]) 

    cost_mat = np.zeros((max_length, max_length)) #NDTW cost matrix

    #pointwise distances between center and series. Used for computation, overwritten for each series.
    delta_mat = np.zeros((max_length, max_length)) 

    #temporary map for intermediary distance computations
    tmp_delta_mat = np.zeros((max_length, max_length)) 

    path_mat = np.zeros((max_length, max_length), dtype=np.int8) #stores optimal warping path


    #barycenter initialization
    medoid_ind = approximate_medoid_index(series,cost_mat,delta_mat,tmp_delta_mat)
    center = series[medoid_ind]

    for i in range(0,n_iterations):
        center = DBA_update(center, series, cost_mat, path_mat, delta_mat,tmp_delta_mat)

    return center

def approximate_medoid_index(series,cost_mat,delta_mat,tmp_delta_mat):

    #if we have more than 50 series we pick random subset ot find medoid
    if len(series)<=50:
        indices = range(0,len(series))
    else:
        indices = np.random.choice(range(0,len(series)),50,replace=False)

    medoid_ind = -1
    best_ss = 1e20
    for index_candidate in indices:
        candidate = series[index_candidate]

        #looking for series with lowest ss (min total NDTW distance)
        ss = sum_of_squares(candidate,series,cost_mat,delta_mat,tmp_delta_mat)

        if(medoid_ind==-1 or ss<best_ss):
            best_ss = ss
            medoid_ind = index_candidate

    return medoid_ind

def sum_of_squares(s,series,cost_mat,delta_mat,tmp_delta_mat):
    #For each timeseries t in series, we compute NDTW distance between t and s then sum that up 
    return sum(map(lambda t:squared_DTW(s,t,cost_mat,delta_mat,tmp_delta_mat),series))

def DTW(s,t,cost_mat,delta_mat):
    return np.sqrt(squared_DTW(s,t,cost_mat,delta_mat))

def squared_DTW(s,t,cost_mat,delta_mat,tmp_delta_mat):
    s_len = s.shape[0]
    t_len = t.shape[0]

    fill_delta_mat_dtw(s, t, delta_mat,tmp_delta_mat)

    #filling dtw cost matrix
    cost_mat[0, 0] = delta_mat[0, 0] #corner

    #init edges
    for i in range(1, s_len):
        cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]

    for j in range(1, t_len):
        cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]

    #filling interior
    for i in range(1, s_len):
        for j in range(1, t_len):
            diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
            if(diag <=left):
                if(diag<=top):
                    res = diag
                else:
                    res = top
            else:
                if(left<=top):
                    res = left
                else:
                    res = top
            cost_mat[i, j] = res+delta_mat[i, j]

    optimal_cost = cost_mat[s_len-1,t_len-1]      

    #doing the opposite as above to compute length of warping path
    i,j  = s_len-1, t_len-1
    path_len = 1

    while i > 0 or j > 0:

        #handling border cases
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:

            diag = cost_mat[i-1, j-1]
            left = cost_mat[i, j-1]
            top  = cost_mat[i-1, j]
            best = min(diag, left, top)

            #going back up direction with lowest cost
            if best == diag:
                i -= 1; j -= 1
            elif best == left:
                j -= 1
            else:
                i -= 1
        path_len += 1


    normalized = optimal_cost/path_len

    return normalized

#computes squared distance between every pair of timesteps
def fill_delta_mat_dtw(center, s, delta_mat, tmp_delta_mat):

    n_dims = center.shape[1]
    len_center = center.shape[0]
    len_s=  s.shape[0]

    #region of delta_mat actually being used
    slim = delta_mat[:len_center,:len_s]
    slim_tmp = tmp_delta_mat[:len_center,:len_s]

    #first dimension - not in the loop to avoid initialisation of delta_mat
    np.subtract.outer(center[:, 0], s[:,0],out = slim)
    np.square(slim, out=slim) #squares every element

    for d in range(1,center.shape[1]):
        np.subtract.outer(center[:, d], s[:, d],out = slim_tmp)
        np.square(slim_tmp, out=slim_tmp)
        np.add(slim,slim_tmp,out=slim)

    assert(np.abs(np.sum(np.square(center[0]-s[0]))-delta_mat[0,0])<=1e-6)

def DBA_update(center, series, cost_mat, path_mat, delta_mat, tmp_delta_mat):
    options_argmin = [(-1, -1), (0, -1), (-1, 0)]
    updated_center = np.zeros(center.shape)
    center_length = center.shape[0]
    n_elements = np.zeros(center_length, dtype=int)

    for s in series:
        s_len = s.shape[0]
        fill_delta_mat_dtw(center, s, delta_mat, tmp_delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        path_mat[0, 0] = -1

        for i in range(1, center_length):
            cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]
            path_mat[i, 0] = 2

        for j in range(1, s_len):
            cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]
            path_mat[0, j] = 1

        for i in range(1, center_length):
            for j in range(1, s_len):
                diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
                if(diag <=left):
                    if(diag<=top):
                        res = diag
                        path_mat[i,j] = 0
                    else:
                        res = top
                        path_mat[i,j] = 2
                else:
                    if(left<=top):
                        res = left
                        path_mat[i,j] = 1
                    else:
                        res = top
                        path_mat[i,j] = 2

                cost_mat[i, j] = res+delta_mat[i, j]

        i = center_length-1
        j = s_len-1

        while(path_mat[i, j] != -1):
            updated_center[i] += s[j]
            n_elements[i] += 1
            move = options_argmin[path_mat[i, j]]
            i += move[0]
            j += move[1]
        assert(i == 0 and j == 0)
        updated_center[i] += s[j]
        n_elements[i] += 1

    return np.divide(updated_center, n_elements[:, np.newaxis])
