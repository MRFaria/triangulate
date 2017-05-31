import unittest
import least_squares


class TestVesselNavigation(unittest.TestCase):
    stations = [[610, 1040], [220, 720], [140, 150]]
    initial_pos = [830, 420]

#    def test_calculate_position(self):
#        coords = [[610, 1049], [220, 720], [140, 150]]
#        assert navigation.position(coords, [660, 680, 740]) == [830, 418]

    def test_transform_matrix(self):
        r0 = ((self.initial_pos[0] - self.stations[0][0])**2 +
              (self.initial_pos[1] - self.stations[0][1])**2)**0.5

        T1 = (self.initial_pos[0] - self.stations[0][0]) / r0

        transformation_matrix = least_squares.transformation_matrix(
            self.stations, self.initial_pos)
        self.assertAlmostEqual(transformation_matrix[0, 0], T1)

    def test_least_squares(self):
        transformation_matrix = least_squares.transformation_matrix(
            self.stations, self.initial_pos)
        a = least_squares.least_squares(
            transformation_matrix, [200, 200, 200],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        print(a)
