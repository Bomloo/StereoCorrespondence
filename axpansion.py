import cv2
import os
import numpy as np
import math
from numba import njit, prange
import maxflow

import utility as util

@njit
def relevant_assignments(shape, ref_features, disparity, occluded, alpha):
    """
    A matrix the size of the image with disparity values at pixel coordinates. 
    A_alpha contains nonactive pairs with alpha disparity and 
    A_other contains active pairs with non-alpha disparity. -1 elsewhere.
    A_alpha_q and A_other_q are the same but disparity values are located on coordinates of the counterpart.
    """
    A_alpha = []
    A_other = []
    A_alpha_p = np.ones(shape) * -1
    A_alpha_q = np.ones(shape) * -1
    A_other_p = np.ones(shape) * -1
    A_other_q = np.ones(shape) * -1

    for i in prange(len(ref_features)):
        d = disparity[i]
        ref = ref_features[i]
        if d != alpha:
            A_other.append((ref, (ref[0]+d, ref[1])))
            A_other_p[ref[1]][ref[0]] = d
            A_other_q[ref[1]][ref[0] + d] = d
            if ref[0] + alpha < shape[1]:
                A_alpha.append((ref, (ref[0]+alpha, ref[1])))
                A_alpha_p[ref[1]][ref[0]] = alpha
                A_alpha_q[ref[1]][ref[0] + alpha] = alpha

    for i in prange(len(occluded)):
        occlu = occluded[i]
        if occlu[0] + alpha < shape[1]:
            A_alpha.append((occlu, (occlu[0]+alpha, occlu[1])))
            A_alpha_p[occlu[1]][occlu[0]] = alpha
            A_alpha_q[occlu[1]][occlu[0] + alpha] = alpha
    
    return A_alpha, A_other, A_alpha_p, A_other_p, A_alpha_q, A_other_q

def create_graph(shape):
    """
    Builds a graph for alpha expansion, returns grids with node number at coordinate location.
    """
    graph = maxflow.GraphFloat()
    grid_alpha = graph.add_grid_nodes(shape)
    grid_other = graph.add_grid_nodes(shape)
    return graph, grid_alpha, grid_other

def add_edges(graph, grid_alpha, grid_other, grid_unique, source_caps_alpha, sink_caps_alpha, source_caps_other, sink_caps_other, weights_alpha, weights_other, weights_unique):
    """
    Takes weights and adds edges to graph.
    """
    top_structure = np.array([[0, 1, 0], 
                              [0, 0, 0], 
                              [0, 0, 0]])
    bot_structure = np.array([[0, 0, 0], 
                              [0, 0, 0], 
                              [0, 1, 0]])
    right_structure = np.array([[0, 0, 0], 
                                [0, 0, 1], 
                                [0, 0, 0]])
    left_structure = np.array([[0, 0, 0], 
                               [1, 0, 0],   
                               [0, 0, 0]])
    structures = [top_structure, bot_structure, right_structure, left_structure]
    for i in range(4):
        graph.add_grid_edges(grid_alpha, weights=weights_alpha[i], structure=structures[i], symmetric=False)
        graph.add_grid_edges(grid_other, weights=weights_other[i], structure=structures[i], symmetric=False)

    graph.add_grid_tedges(grid_alpha, source_caps_alpha, sink_caps_alpha)
    graph.add_grid_tedges(grid_other, source_caps_other, sink_caps_other)
    graph.add_grid_edges(grid_unique, weights=weights_unique, structure=top_structure, symmetric=False)

    return graph

@njit
def e_data_occlusion(A_alpha_p, A_other_p, ref, sample, max_offset):
    """
    data:
      inactive pairs with disparity alpha:
          s -> node = (l - r)**2 pixel value capped at 30
          node -> t = 0
      active pairs with disparity not alpha:
          s -> node = 0
          node -> t = (l - r)**2 pixel value capped at 30
    """
    population = []
    shape = ref.shape
    n = shape[1]
    source_caps_alpha = np.zeros(shape).astype(np.float64)
    sink_caps_alpha = np.zeros(shape).astype(np.float64)
    source_caps_other = np.zeros(shape).astype(np.float64)
    sink_caps_other = np.zeros(shape).astype(np.float64)
    for i in prange(shape[0]):
        for j in prange(shape[1]):
            p_val1 = A_alpha_p[i][j]
            p_val2 = A_other_p[i][j]
            if p_val1 > -1 or p_val2 > -1:
                if p_val1 > -1:
                    sam_x = np.uint64(j + p_val1)
                    ssd = np.float64(ref[i][j]) - np.float64(sample[i][sam_x])
                    if ssd > 30:
                        ssd = 30
                    ssd = ssd**2
                    if j + max_offset < n:
                        population.append(ssd)
                    source_caps_alpha[i][j] = source_caps_alpha[i][j] + ssd
                    sink_caps_alpha[i][j] = sink_caps_alpha[i][j] + 1
                if p_val2 > -1:
                    sam_x = np.uint64(j + p_val2)
                    ssd = np.float64(ref[i][j]) - np.float64(sample[i][sam_x])
                    if ssd > 30:
                        ssd = 30
                    ssd = ssd**2
                    if j + max_offset < n:
                        population.append(ssd)
                    sink_caps_other[i][j] = sink_caps_other[i][j] + ssd
                    source_caps_other[i][j] = source_caps_other[i][j] + 1
    k = np.percentile(population, 90)
    if k < 3:
        k = 3
    lam = k/5
    sink_caps_alpha = sink_caps_alpha * k
    source_caps_other = source_caps_other * k
    return lam, source_caps_alpha, sink_caps_alpha, source_caps_other, sink_caps_other

@njit
def neighbors(shape, x, y, d):
    """
    Up, down, right, left
    """
    m = shape[0]
    n = shape[1]

    neighbs = []

    if x + d < n:
        if y - 1 > -1:
            neighbs.append(((x, y-1), (x+d, y-1)))
        else:
            neighbs.append(((-1, -1), (-1, -1)))
        if y + 1 < m:
            neighbs.append(((x, y+1), (x+d, y+1)))
        else:
            neighbs.append(((-1, -1), (-1, -1)))
    else:
        neighbs.append(((-1, -1), (-1, -1)))
        neighbs.append(((-1, -1), (-1, -1)))
    
    if x + 1 + d < n:
        neighbs.append(((x+1, y), (x+1+d, y)))
    else:
        neighbs.append(((-1, -1), (-1, -1)))
    if x - 1 + d < n:
        neighbs.append(((x-1, y), (x-1+d, y)))
    else:
        neighbs.append(((-1, -1), (-1, -1)))
    
    return neighbs

@njit
def e_smoothness(assignment, ref, sample, lam):
    """
    smoothness 4 neighbors:
      v is 3 lam if values of p and p1 along with q and q1 differ by less than 8
      v is lam if values differ more
      inactive pairs with disparity alpha with 
      inactive pairs with disparity alpha and adjacent:
          s -> node1 = 0
          node1 -> t = v
          node2 -> node1 = v + v - 0 - 0
          node2 -> t = 0 - v
      active pairs with disparity not alpha with 
      active pairs with disparity not alpha but matching and adjacent:
          s -> node1 = 0
          node1 -> t = v
          node2 -> node1 = v + v - 0 - 0
          node2 -> t = 0 - v
      inactive pairs with disparity alpha with 
      active pairs with disparity alpha and adjacent:
          s -> node = 0
          node -> t = v
      active pairs with disparity not alpha with 
      inactive pairs with disparity not alpha but matching and adjacent:
          s -> node = 0
          node -> t = v
    """
    shape = ref.shape
    sink_caps = np.zeros(shape).astype(np.float64)

    weight_top = np.zeros(shape).astype(np.float64)
    weight_bot = np.zeros(shape).astype(np.float64)
    weight_right = np.zeros(shape).astype(np.float64)
    weight_left = np.zeros(shape).astype(np.float64)
    check_assignment = set(assignment)
    for a_num in prange(len(assignment)):
        a = assignment[a_num]
        p = a[0]
        q = a[1]
        x = p[0]
        y = p[1]
        val = np.float64(ref[y][x])
        val1 = np.float64(sample[q[1]][q[0]])
        neighbs = neighbors(shape, x, y, q[0] - p[0])
        for num in prange(4):
            n = neighbs[num]
            if n != ((-1, -1), (-1, -1)):
                v = lam
                nx = n[0][0]
                ny = n[0][1]
                if max(abs(np.float64(ref[ny][nx]) - val), abs(np.float64(sample[n[1][1]][n[1][0]]) - val1)) < 8:
                    v = 3 * lam

                sink_caps[y][x] = sink_caps[y][x] + v
            
                if n in check_assignment:
                    # up
                    if num == 0:
                        weight_bot[ny][nx] = weight_bot[ny][nx] + v * 2
                    # down
                    elif num == 1:
                        weight_top[ny][nx] = weight_top[ny][nx] + v * 2
                    # right
                    elif num == 2:
                        weight_left[ny][nx] = weight_left[ny][nx] + v * 2
                    # left
                    else:
                        weight_right[ny][nx] = weight_right[ny][nx] + v * 2
                    sink_caps[ny][nx] = sink_caps[ny][nx] - v
    return [weight_top, weight_bot, weight_right, weight_left], sink_caps

@njit
def e_unique(A_alpha_p, A_other_p, A_alpha_q, A_other_q, grid_alpha, grid_other):
    """
    uniqueness:
      inactive pairs with disparity alpha and active pairs with disparity not alpha sharing p:
          s -> node1 = 0
          node1 -> t = 0
          node2 -> node1 = 0 + inf - 0 - 0
          node2 -> t = 0
      inactive pairs with disparity alpha and active pairs with disparity not alpha sharing q:
          s -> node1 = 0
          node1 -> t = 0
          node2 -> node1 = 0 + inf - 0 - 0
          node2 -> t = 0
    """
    unique_alpha = []
    unique_other = []

    for i in prange(grid_alpha.shape[0]):
        for j in prange(grid_alpha.shape[1]):
            p_val1 = A_alpha_p[i][j]
            p_val2 = A_other_p[i][j]
            q_val1 = A_alpha_q[i][j]
            q_val2 = A_other_q[i][j]
            if p_val1 > -1 and p_val2 > -1:
                p_val1 = np.int64(p_val1)
                p_val2 = np.int64(p_val2)
                unique_alpha.append(grid_alpha[i][j])
                unique_other.append(grid_other[i][j])
            if q_val1 > -1 and q_val2 > -1:
                q_val1 = np.int64(q_val1)
                q_val2 = np.int64(q_val2)
                unique_alpha.append(grid_alpha[i][j-q_val1])
                unique_other.append(grid_other[i][j-q_val2])
    grid_unique = np.array([unique_alpha, unique_other])
    weights_unique = np.vstack((np.zeros(grid_unique.shape[1]), np.ones(grid_unique.shape[1]) * 1000))
    return grid_unique, weights_unique

def minimize_energy(ref, sample, ref_features, disparity, occluded, max_offset, alpha):
    """
    Builds a graph and minimizes energy based on data, occlusion, smoothness, and uniqueness.
    """
    print('Creating graph')
    A_alpha, A_other, A_alpha_p, A_other_p, A_alpha_q, A_other_q = relevant_assignments(ref.shape, ref_features, disparity, occluded, alpha)
    graph, grid_alpha, grid_other = create_graph(ref.shape)
    
    print('Creating data weights')
    print('Creating occlusion weights')
    lam, d_source_caps_alpha, o_sink_caps_alpha, o_source_caps_other, d_sink_caps_other = e_data_occlusion(A_alpha_p, A_other_p, ref, sample, max_offset)

    print('Creating smoothness weights')
    weights_alpha, s_sink_caps_alpha = e_smoothness(A_alpha, ref, sample, lam)
    weights_other, s_sink_caps_other = e_smoothness(A_other, ref, sample, lam)

    print('Creating uniqueness weights')
    grid_unique, weights_unique = e_unique(A_alpha_p, A_other_p, A_alpha_q, A_other_q, grid_alpha, grid_other)

    source_caps_alpha = d_source_caps_alpha
    sink_caps_alpha = o_sink_caps_alpha + s_sink_caps_alpha

    source_caps_other = o_source_caps_other
    sink_caps_other = d_sink_caps_other + s_sink_caps_other

    print('Adding edges')
    graph = add_edges(graph, grid_alpha, grid_other, grid_unique, source_caps_alpha, sink_caps_alpha, source_caps_other, sink_caps_other, weights_alpha, weights_other, weights_unique)

    prev_value = np.sum(sink_caps_alpha) + np.sum(sink_caps_other)
    cut_value = graph.maxflow()
    success = cut_value < prev_value

    grid_alpha_activate = graph.get_grid_segments(grid_alpha)
    grid_other_deactivate = graph.get_grid_segments(grid_other)

    print('Minimization finished')

    return grid_alpha_activate, grid_other_deactivate, success

@njit
def process_changes(grid_alpha_activate, grid_other_deactivate, ref_features, sample_features, disparity, occluded, alpha):
    new_ref_features = []
    new_sample_features = []
    new_disparity = []
    new_occluded = []

    for i in prange(len(ref_features)):
        p = ref_features[i]
        x = p[0]
        y = p[1]
        q = sample_features[i]
        d = disparity[i]
        activate = grid_alpha_activate[y][x]
        deactivate = grid_other_deactivate[y][x]
        if d == alpha:
            new_ref_features.append(p)
            new_sample_features.append(q)
            new_disparity.append(alpha)
        elif deactivate and activate:
            new_ref_features.append(p)
            new_sample_features.append((p[0]+alpha, p[1]))
            new_disparity.append(alpha)
        elif deactivate and not activate:
            new_occluded.append(p)
        else:
            new_ref_features.append(p)
            new_sample_features.append(q)
            new_disparity.append(d)

    for i in prange(len(occluded)):
        p = occluded[i]
        x = p[0]
        y = p[1]
        activate = grid_alpha_activate[y][x]
        if activate:
            new_ref_features.append(p)
            new_sample_features.append((p[0]+alpha, p[1]))
            new_disparity.append(alpha)
        else:
            new_occluded.append(p)
    
    return new_ref_features, new_sample_features, new_disparity, new_occluded

@njit
def start(shape):
    occluded = []
    for i in prange(shape[0]):
        for j in prange(shape[1]):
            occluded.append((j, i))
    occluded.pop(0)
    return occluded

def alpha_expansion(ref, sample, max_offset, disparity_norm):
    """
    Performs alpha expansion on stereo images and returns data for a disparity map.
    """
    ref_features = [(0, 0)]
    sample_features = [(0, 0)]
    disparity = [0]
    occluded = start(ref.shape)
    track = np.zeros(max_offset)
    for i in range(4):
        print('Starting Iteration: ', i + 1)
        alpha_iterations = np.random.choice(range(max_offset), max_offset, False)
        for a_num in range(len(alpha_iterations)):
            alpha = alpha_iterations[a_num]
            print('Iteration: ', i + 1)
            print('Level: ', str(a_num + 1) + '/' + str(max_offset))
            if track[alpha] == 0:
                grid_alpha_activate, grid_other_deactivate, success = minimize_energy(ref, sample, ref_features, disparity, occluded, max_offset, alpha)
                if success:
                    result = process_changes(grid_alpha_activate, grid_other_deactivate, ref_features, sample_features, disparity, occluded, alpha)
                    ref_features = result[0]
                    sample_features = result[1]
                    disparity = result[2]
                    occluded = result[3]
                    track[:] = 0
                track[alpha] = 1
                if np.sum(track) == max_offset:
                    return ref_features, disparity
        print('Finish Iteration: ', i + 1)

    map = util.generate_map(ref.shape, ref_features, disparity, disparity_norm)
    return map