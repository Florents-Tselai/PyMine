# Copyright 2014 Florents Tselai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from numpy import array
from numpy.testing import assert_array_equal
import unittest

import numpy as np
from utils import *


class utils_test(unittest.TestCase):
    def setUp(self):
        self.Dx = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]
        
    def test_number_of_points_in_partition(self):
        assert number_of_points_in_partition([-1, 2, 7]) == 8
        assert number_of_points_in_partition([-1, 2, 5, 7, 9]) == 10
        assert number_of_points_in_partition([2, 7]) == 5
        
    def test_get_distribution_of_points(self):
        assert np.array_equal(get_distribution_of_points([-1, 2, 7]), array([3, 5]))
        assert np.array_equal(get_distribution_of_points([-1, 2, 5, 7, 9]), array([3, 3, 2, 2]))
        assert np.array_equal(get_distribution_of_points([2, 7]), array([5]))
        
    def test_get_partition_histogram(self):
        assert np.array_equal(get_partition_histogram([-1, 2, 7]), array([3. / 8, 5. / 8]))
        assert np.array_equal(get_partition_histogram([-1, 2, 5, 7, 9]), array([3. / 10, 3. / 10, 2. / 10, 2. / 10]))
        assert np.array_equal(get_partition_histogram([2, 7]), array([5. / 5]))
        assert np.array_equal(get_partition_histogram([-1, 9]), array([10. / 10]))
        
    def test_get_grid_histogram(self):
        assert np.array_equal(GetGridHistogram([-1, 2, 5, 7, 9], {(0, 1):0, (1, 1):0, (2, 1):0, (3, 1):1, (4, 1):1, (5, 1):1, (6, 1):1, (7, 1):1, (8, 1):1, (9, 1):1}), array([0. / 10, 3. / 10, 2. / 10, 2. / 10, 3. / 10, 0. / 10, 0. / 10, 0. / 10]))

    def test_GetPartitionMapFromOrdinals(self):
        assert GetPartitionMapFromOrdinals(self.Dx, [-1, 2, 7]) == {0: [(0, 1), (1, 1), (2, 1)], 1: [(3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]}
    
    def test_visualize(self):
        D = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        Dx = sort_D_increasing_by(D, 'x')
        Q = GroupPointsByPartition({(0, 1):0, (1, 2):0, (2, 3):0, (3, 4):1, (4, 5):1, (5, 6):1, (6, 7):1, (7, 8):1, (8, 9):1, (9, 10):1})
        P = GetPartitionMapFromOrdinals(Dx, [-1, 2, 5, 7, 9])
        visualize(P, Q)

if __name__ == '__main__':
    unittest.main()
