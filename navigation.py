import numpy as np


class LeastSquares:
    def __init__(self, stations, residuals, initial_guess=(830, 420)):
        self.stations = [[610, 1040], [220, 720], [140, 150]]
        self.x0 = initial_guess[0]
        self.y0 = initial_guess[1]

    @property
    def transformation_matrix(self):
        _transform = []
        for station in self.stations:
            r0 = ((self.x0 - station[0])**2 + (self.y0 - station[1])**2)**0.5
            row = [(self.x0 - station[0])/r0, (self.y0 - station[1])/r0]
            _transform.append(row)
        return np.matrix(_transform)

    def least_squares(self, residuals, weighting):
        """ Calculate (dx, dy) in A(dx, dy) = b,
            where b is the vector of residuals,
            and A is the transformation matrix, using the
            least squares method

            x = (A^T W A)^-1 A^T W b, where W is the error weighting
            """
        transform = self.transformation_matrix
        residuals = np.matrix(residuals).transpose()
        weighting = np.matrix(weighting)

        # Make sure vectors are column vectors
        assert residuals.shape[1] == 1

        square = transform.transpose()*weighting*transform
        covar = np.linalg.inv(square)

        movement = covar * transform.transpose() * weighting * residuals

        return movement, covar


class Vessel:
    def position(coords: list, ranges: list):
        """ Returns a vessels position based on station
        location and range measurements """
        assert len(coords) == len(ranges)
        return coords



if __name__ == '__main__':
    print("running...")
