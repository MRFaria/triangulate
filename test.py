import unittest
import least_squares as ls
import numpy as np
import navigation as na


class TestVesselNavigator(unittest.TestCase):
    stations = [[610, 1040], [220, 720], [140, 150]]
    guess_pos = np.array([830, 420])

    var = 1/(2.5*2.5)
    ranges = np.array([660, 680, 740])
    vessel = na.VesselNavigator2D(guess_pos, stations, var, ranges)

    def test_get_position(self):
        pos, sd, rms = self.vessel.get_position()
        correct_pos = [829.57, 417.86]
        correct_sd = [1.94, 2.34]

        np.testing.assert_almost_equal(pos, correct_pos, 2)
        np.testing.assert_almost_equal(sd, correct_sd, 2)


class TestLeastSquares(unittest.TestCase):
    stations = [[610, 1040], [220, 720], [140, 150]]
    guess_pos = np.array([830, 420])

    def test_transform_matrix(self):
        r0 = ((self.guess_pos[0] - self.stations[0][0])**2 +
              (self.guess_pos[1] - self.stations[0][1])**2)**0.5

        T1 = (self.guess_pos[0] - self.stations[0][0]) / r0

        transformation_matrix = ls.transformation_matrix(
            self.stations, self.guess_pos)
        self.assertAlmostEqual(transformation_matrix[0, 0], T1)

    def test_least_squares(self):
        pos = self.guess_pos
        movement = np.zeros(2)

        r0 = ls.distance(self.stations, pos)

        transformation_matrix = ls.transformation_matrix(
            self.stations, pos)
        residuals = np.array([660, 680, 740]) - r0

        var = 1/(2.5*2.5)
        movement, covar = ls.least_squares(transformation_matrix,
                                           residuals,
                                           [[var, 0, 0],
                                            [0, var, 0],
                                            [0, 0, var]])
        pos = pos + movement

        # From problem sheet
        correct_pos = [829.57, 417.86]
        correct_sd = [1.94, 2.34]

        np.testing.assert_almost_equal(pos, correct_pos, 2)
        np.testing.assert_almost_equal(np.diag(covar) ** 0.5, correct_sd, 2)

    def test_rms(self):
        r0 = ls.distance(self.stations, self.guess_pos)
        residuals = np.array([660, 680, 740]) - r0
        rms = ls.rms(residuals)
        self.assertAlmostEqual(0.46663792535237764, rms)
