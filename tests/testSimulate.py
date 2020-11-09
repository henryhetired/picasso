import unittest
from numba import njit
import numpy as _np
from picasso.simulate import calculate_zpsf
@njit
class SimulateTestCase(unittest.TestCase):
    """
    This is the test case for the simulate module
    """
    def test_calculate_zpsf(self):
        cx = _np.array([1, 2, 3, 4, 5, 6, 7])
        cy = _np.array([1, 2, 3, 4, 5, 6, 7])
        z = _np.array([1, 2, 3, 4, 5, 6, 7])
        wx, wy = calculate_zpsf(z, cx, cy)

        result = [4.90350522e+01,   7.13644987e+02,   5.52316597e+03,
                2.61621620e+04,   9.06621337e+04,   2.54548124e+05,
                6.14947219e+05]

        delta = wx - result
        assert sum(delta**2) < 0.001
        
