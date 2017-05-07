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


import unittest

import numpy as np
from numpy import array
from numpy.testing import assert_array_equal

from mine.mine import *


class computations_test(unittest.TestCase):
    def setUp(self):
        # Example from wiki https://en.wikipedia.org/wiki/Marginal_distribution
        # Appears also in Cover et.al. book pg. 18
        self.wiki_example = array([[4. / 32, 2. / 32, 1. / 32, 1. / 32],
                                   [2. / 32, 4. / 32, 1. / 32, 1. / 32],
                                   [2. / 32, 2. / 32, 2. / 32, 2. / 32],
                                   [8. / 32, 0., 0., 0.]])

        # Example from Guttler's presentation
        self.guttler_example = array([[1. / 20, 1. / 20, 1. / 20],
                                      [3. / 20, 5. / 20, 2. / 20],
                                      [3. / 20, 2. / 20, 2. / 20]])

        assert np.sum(self.wiki_example) == 1.

    def test_H(self):
        X = get_x_distribution(self.wiki_example)
        assert H(X) == 7. / 4
        Y = get_y_distribution(self.wiki_example)
        assert H(Y) == 2.

        # Joint entropy
        assert H(self.wiki_example) == 27. / 8

    def test_getXDistribution(self):
        assert_array_equal(get_x_distribution(self.wiki_example), array([1. / 2, 1. / 4, 1. / 8, 1. / 8]))

    def test_getYDistribution(self):
        assert_array_equal(get_y_distribution(self.wiki_example), array([1. / 4, 1. / 4, 1. / 4, 1. / 4]))

    def test_I(self):
        assert I(self.wiki_example) == 3. / 8

    def test_HP(self):
        """
        4  |   |     |x|
        3  |   |  x  | |
        2  |   |x    | |
        1  |x x|     | |
        0 x|   |    x| |
          0|1 2|3 4 5|6|
        """
        assert HP(np.array([-1, 0, 2, 5, 6])) == H(np.array([1. / 7, 2. / 7, 3. / 7, 1. / 7]))
        assert HP(np.array([0, 2, 5])) == H(np.array([2. / 5, 3. / 5]))
        assert HP(np.array([-1, 0])) == H(np.array([1. / 1]))
        assert HP(np.array([-1, 5, 6])) == H(np.array([6. / 7, 1. / 7]))

    def test_HQ(self):
        """
          4             x
          3         x   
          2       x     
            *----+---+-----+-
          1   x         
             *----+---+-----+-
          0 x        x
            0 1 2 3 4 5 6
                    
         """

        assert HQ(
            {
                (0, 0): 0,
                (5, 0): 0,
                (1, 1): 1,
                (3, 2): 2,
                (4, 3): 2,
                (6, 4): 2
            }
        ) == H(np.array([2. / 6, 1. / 6, 3. / 6]))

        """
          4             x
          3         x   
          2       x     
             *----+---+-----+-
          1   x x       
          0 x         x 
            0 1 2 3 4 5 6
                      
         """
        assert HQ(
            {
                (0, 0): 0,
                (5, 0): 0,
                (1, 1): 0,
                (2, 1): 0,
                (3, 2): 1,
                (4, 3): 1,
                (6, 4): 1
            }
        ) == H(np.array([4. / 7, 3. / 7]))

    def test_HPQ(self):
        """
          4  |   |     |x
          3  |   |  x  |
          2  |   |x    |
             *----+---+-----+-
          1  |x x|     |
             *----+---+-----+-
          0 x|   |    x|
            0|1 2|3 4 5 6
             |   |     |
         """
        P_ordinals = [-1, 0, 2, 5]
        Q_map = {
            (0, 0): 0,
            (5, 0): 0,

            (1, 1): 1,
            (2, 1): 1,

            (3, 2): 2,
            (4, 3): 2,
            (6, 4): 2
        }

        self.assertEqual(HPQ(P_ordinals, Q_map),
                         H(np.array([0. / 6, 0. / 6, 2. / 6, 0. / 6, 2. / 6, 0. / 6, 1. / 6, 0. / 6, 1. / 6])))


if __name__ == '__main__':
    unittest.main()
