#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

__author__ = 'Florents Tselai'

from collections import Counter

from .utils import *


def H(distribution):
    def entropy(P):
        '''
        Return the Shannon entropy of a probability vector P
        See http://www.scholarpedia.org/article/Entropy#Shannon_entropy
        '''
        h = -np.fromiter((i * np.log2(i) for i in P if i > 0), dtype=np.float64).sum()
        return h

    return entropy(distribution.ravel())


def HQ(Q):
    n = len(Q)
    histogram = np.fromiter(Counter(Q.values()).values(), dtype=int)
    return H(histogram / np.float64(n))


def HPQ(P, Q):
    return H(GetGridHistogram(P, Q))


def HP(P):
    return H(get_partition_histogram(P))


def get_x_distribution(grid_histogram):
    return grid_histogram.sum(axis=0)


def get_y_distribution(grid_histogram):
    return grid_histogram.sum(axis=1)


def I(joint_distribution_histogram):
    x_distribution = get_x_distribution(joint_distribution_histogram)
    y_distribution = get_y_distribution(joint_distribution_histogram)
    joint_distribution = joint_distribution_histogram.ravel()
    return H(x_distribution) + H(y_distribution) - H(joint_distribution)
