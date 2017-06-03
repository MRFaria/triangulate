import numpy as np
from typing import List


class NoSolutionError(Exception):
    """
    Least squares output does not converge to a solution
    """


class SingularMatrixError(NoSolutionError):
    """
    Determinent of matrix tends to zero
    """


def distance(reference_points: List[List[float]],
             point: List[float]):
    """
    Calculates the distances to a point, from a list of
    reference_points.
    """

    reference_points = np.array(reference_points)
    point = np.array(point)
    r0 = np.zeros(len(reference_points))

    for i, _ in enumerate(reference_points):
        r0[i] = ((point[0] - reference_points[i, 0])**2 +
                 (point[1] - reference_points[i, 1])**2)**0.5
    return r0


def transformation_matrix(reference_points: List[List[float]],
                          point: List[float]):
    """
    This produces the linear transformation matrix corresponding
    to the range equation of the vessel from the signal towers
    """

    _transform = []
    r0 = distance(reference_points, point)
    reference_points = np.array(reference_points)
    point = np.array(point)

    for i, station in enumerate(reference_points):
        row = [(point[0] - station[0])/r0[i], (point[1] - station[1])/r0[i]]
        _transform.append(row)

    return np.matrix(_transform)


def least_squares(transformation_matrix: np.ndarray,
                  residuals: List[float],
                  weighting: List[List[float]]):
    """
    Calculate (dx, dy) in A(dx, dy) = b,
    where b is the vector of residuals,
    and A is the transformation matrix, using the
    least squares method

    x = (A^T W A)^-1 A^T W b, where W is the error weighting
    """

    transform = transformation_matrix
    residuals = np.matrix(residuals)
    # Make sure the residuals are a column vector
    if residuals.shape[1] != 1:
        residuals = np.matrix(residuals).transpose()
    weighting = np.matrix(weighting)

    # Make sure vectors are column vectors

    square = transform.transpose()*weighting*transform

    # Check if matrix is singular
    det = np.linalg.det(square)
    if abs(det) < 0.001:
        raise SingularMatrixError

    covar = np.linalg.inv(square)
    movement = covar * transform.transpose() * weighting * residuals

    return np.array(movement).flatten(), covar


def rms(residuals: List[float], soln_length=2):
    return sum(residuals)/len(residuals-soln_length)
