import numpy as np


def transformation_matrix(stations, initial_guess):
    x0 = initial_guess[0]
    y0 = initial_guess[1]
    _transform = []

    for station in stations:
        r0 = ((x0 - station[0])**2 + (y0 - station[1])**2)**0.5
        row = [(x0 - station[0])/r0, (y0 - station[1])/r0]
        _transform.append(row)

    return np.matrix(_transform)


def least_squares(transformation_matrix, residuals, weighting):
    """ Calculate (dx, dy) in A(dx, dy) = b,
        where b is the vector of residuals,
        and A is the transformation matrix, using the
        least squares method

        x = (A^T W A)^-1 A^T W b, where W is the error weighting
        """
    transform = transformation_matrix
    residuals = np.matrix(residuals).transpose()
    weighting = np.matrix(weighting)

    # Make sure vectors are column vectors
    assert residuals.shape[1] == 1

    square = transform.transpose()*weighting*transform
    covar = np.linalg.inv(square)

    movement = covar * transform.transpose() * weighting * residuals

    return movement, covar

