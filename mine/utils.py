#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

__author__ = 'Florents Tselai'


from collections import defaultdict
from itertools import tee, zip

import matplotlib.pyplot as plt
import numpy as np
from numpy import float64

p_x, p_y = lambda p: p[0], lambda p: p[1]


def get_rightest_point(points):
    return max(points, key=p_x)


def get_leftest_point(points):
    return min(points, key=p_x)


def get_highest_point(points):
    return max(points, key=p_y)


def get_lowest_point(points):
    return min(points, key=p_y)


def last_abscissa(x_bin):
    return p_x(get_rightest_point(x_bin))


def last_ordinate(y_bin):
    return p_y(get_highest_point(y_bin))


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def is_sorted_increasing_by(D, increasing_by='x'):
    assert increasing_by == 'x' or increasing_by == 'y'

    if increasing_by == 'x':
        return all(p_x(D[i]) <= p_x(D[i + 1]) for i in range(len(D) - 1))
    else:
        return all(p_y(D[i]) <= p_y(D[i + 1]) for i in range(len(D) - 1))


def get_distribution_of_points(ordinals):
    return np.fromiter((o2 + 1 if o1 < 0 else o2 - o1 for o1, o2 in pairwise(ordinals)), dtype=int)


def number_of_points_in_partition(ordinals):
    return ordinals[-1] - ordinals[0]


def get_partition_histogram(ordinals):
    distribution_of_points = get_distribution_of_points(ordinals)

    histogram = distribution_of_points / float64(number_of_points_in_partition(ordinals))
    return histogram


def sort_D_increasing_by(D, increasing_by='x'):
    assert increasing_by == 'x' or increasing_by == 'y'

    return sorted(D, key=p_x) if increasing_by == 'x' else sorted(D, key=p_y)


def GroupPointsByPartition(P):
    d = defaultdict(list)
    for k, v in P.iteritems(): d[v].append(k)
    return dict(d)


def visualize(x_axis_parition={}, y_axis_partition={}, step=0.2):
    points = set()

    for partition in x_axis_parition.values():
        for p in partition:
            points.add(p)

    for partition in y_axis_partition.values():
        for p in partition:
            points.add(p)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Scatter points
    ax.scatter(map(p_x, points), map(p_y, points))

    x_bin_edge = lambda x_bin: last_abscissa(x_bin) + step
    y_bin_edge = lambda y_bin: last_ordinate(y_bin) + step

    x_ticks = map(x_bin_edge, x_axis_parition.values())
    y_ticks = map(y_bin_edge, y_axis_partition.values())

    ax.get_xaxis().set_ticks(x_ticks)
    ax.get_yaxis().set_ticks(y_ticks)

    # Format grid appearance
    ax.grid(True, alpha=0.5, color='red', linestyle='-', linewidth=1.5)

    x_partition_size = len(x_axis_parition.values())
    y_partition_size = len(y_axis_partition.values())
    plt.title(str(x_partition_size) + ' - by - ' + str(y_partition_size) + ' Grid')
    plt.show()


def GetPartitionMapFromOrdinals(D, ordinals, axis='x'):
    assert is_sorted_increasing_by(D, axis)

    partition_map = {}
    current_partition = 0
    for p_begin, p_end in pairwise(ordinals):
        partition_points = []
        for point_index in range(p_begin + 1, p_end + 1):
            partition_points.append(D[point_index])
        partition_map[current_partition] = partition_points
        current_partition += 1
    return partition_map


def partition_size(ordinals):
    return len(ordinals) - 1


def GetGridHistogram(P_ordinals, Q):
    Dx = sort_D_increasing_by(Q.keys(), 'x')

    rows = GroupPointsByPartition(Q)
    columns = GetPartitionMapFromOrdinals(Dx, P_ordinals)
    m = number_of_points_in_partition(P_ordinals)

    def grid_cell_cize(row_index, column_index):
        return len(set(rows[row_index]) & set(columns[column_index]))

    grid_points_distribution = (grid_cell_cize(r, c) for r in reversed(xrange(len(rows))) for c in xrange(len(columns)))
    grid_distribution = np.fromiter(grid_points_distribution, dtype=int)

    histogram = grid_distribution / float64(m)
    return histogram


def GetPartitionOrdinalsFromMap(D, P, axis='x'):
    assert is_sorted_increasing_by(D, axis)

    P_tilde = GroupPointsByPartition(P)

    if axis == 'x':
        return [D.index(get_leftest_point(P_tilde[0])) - 1] + [D.index(get_rightest_point(P_tilde[k])) for k in
                                                               sorted(P_tilde.keys())]
    elif axis == 'y':
        return [D.index(get_lowest_point(P_tilde[0])) - 1] + [D.index(get_highest_point(P_tilde[k])) for k in
                                                              sorted(P_tilde.keys())]
