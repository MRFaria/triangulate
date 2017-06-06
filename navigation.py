import least_squares as ls
import numpy as np
from typing import List


class VesselNavigator2D:
    """
    Calculates the position of the vessel, based on
    the station coordinates
    """

    _max_iterations = 7

    def __init__(self, guess_pos: List[float],
                 stations: List[List[float]],
                 variance: float,
                 ranges: List[float]):
        self.guess_pos = guess_pos
        self.stations = stations
        self.variance = variance
        self.ranges = ranges
        self.var_matrix = np.identity(len(stations)) * variance

        if len(ranges) != len(stations):
            print('ERROR - Need a range value for every measurement station')
            raise ValueError

    def get_position(self):
        """
        Returns a vessels position based on station
        location and range measurements
        """
        pos = self.guess_pos

        for iteration in range(self._max_iterations):
            transformation_matrix = ls.transformation_matrix(
                self.stations, pos)

            old_pos = pos
            pos, sd, rms = self._correct_position(pos, transformation_matrix)

            if all(np.isclose(pos, old_pos, atol=0.001)):
                return pos, sd, rms
        else:
            print('No solution, old_pos : pos = ', old_pos, ':', pos)
            return None

    def _correct_position(self,
                          pos: List[float],
                          transformation_matrix: np.ndarray):
        """
        Correct initial unprecise position vector by triangulating distances
        to signal towers.
        This is done using the least squares technique
        """

        r0 = ls.distance(self.stations, pos)
        residuals = self.ranges - r0
        movement, covar = ls.least_squares(transformation_matrix,
                                           residuals,
                                           self.var_matrix)
        sd = np.diag(covar) ** 0.5
        rms = ls.rms(residuals)

        return pos + movement, sd, rms


if __name__ == '__main__':
    stations = np.matrix([[610, 1040], [220, 720], [140, 150]])
    guess_pos = np.array([830, 420])
    var = 1/(2.5*2.5)
    ranges = np.array([660, 680, 740])

    vessel = VesselNavigator2D(guess_pos, stations, var, ranges)
    print(vessel.get_position())
