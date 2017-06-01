import numpy as np


class NoSolutionError(Exception):
    """
    Least squares output does not converge to a solution
    """


def distance(stations, pos):
    """
    Calculates the distances to the measured point, from the
    various signal towers.
    """

    r0 = np.zeros(len(stations))
    for i, _ in enumerate(stations):
        r0[i] = ((pos[0] - stations[i][0])**2 +
                 (pos[1] - stations[i][1])**2)**0.5
    return r0


def transformation_matrix(stations, pos):
    """
    This produces the linear transformation matrix corresponding
    to the range equation of the vessel from the signal towers
    """

    _transform = []
    r0 = distance(stations, pos)

    for i, station in enumerate(stations):
        row = [(pos[0] - station[0])/r0[i], (pos[1] - station[1])/r0[i]]
        _transform.append(row)

    return np.matrix(_transform)


def least_squares(transformation_matrix, residuals, weighting):
    """
    Calculate (dx, dy) in A(dx, dy) = b,
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

    # Check if matrix is singular
    det = np.linalg.det(square)
    if abs(det) < 0.001:
        print('Error - Singular matrix')
        raise np.linalg.LinAlgError

    covar = np.linalg.inv(square)
    movement = covar * transform.transpose() * weighting * residuals

    return np.array(movement).flatten(), covar
