import least_squares as ls
import numpy as np


class VesselNavigator2D:
    """
    Calculates the position of the vessel, based on
    the station coordinates
    """

    _max_iterations = 7

    def __init__(self, guess_pos, stations, variance, ranges):
        """
        args:
        [float, float], [[float, float], ...], float, [[float, float], ...]
        """
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
            pos, sd = self._correct_position(pos, transformation_matrix)

            if all(np.isclose(pos, old_pos, atol=0.001)):
                return pos, sd
        else:
            print('No solution, old_pos : pos = ', old_pos, ':', pos)
            raise ls.NoSolutionError

    def _correct_position(self, pos, transformation_matrix):
        """
        Correct initial unprecise position vector by triangulating distances
        to signal towers.
        This is done using the least squares technique
        """

        r0 = ls.distance(self.stations, pos)
        residuals = self.ranges - r0
        try:
            movement, covar = ls.least_squares(transformation_matrix,
                                               residuals,
                                               self.var_matrix)
            sd = np.diag(covar) ** 0.5
        except np.linalg.LinAlgError:
            return pos, sd

        return pos + movement, sd


if __name__ == '__main__':
    stations = [[610, 1040], [220, 720], [140, 150]]
    guess_pos = np.array([830, 420])
    var = 1/(2.5*2.5)
    ranges = np.array([660, 680, 740])

    vessel = VesselNavigator2D(guess_pos, stations, var, ranges)
    print(vessel.get_position())
