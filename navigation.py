import numpy as np

def least_squares(residuals, taylor_transform):
    taylor_transform = np.matrix(taylor_transform)
    residuals = np.matrix(residuals)

    transpose = taylor_transform.transpose()
    square = taylor_transform * transpose

    try:
        inverse = np.linalg.inv(square)
    except np.linalg.LinAlgError as e:
        inverse = None

    try:
        movement = inverse * transpose * np.matrix(residuals)
    except ValueError as e:
        print(e)
    except TypeError as e:
        print(e)
    else:
        return movement


def calculate_position(coords: list, ranges: list):
    """ Returns a vessels position based on station location and range measurements """
    assert len(coords) == len(ranges)
    return coords


if __name__ == '__main__':
    print("running...")


