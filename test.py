import unittest
import navigation

class TestVesselNavigation(unittest.TestCase):
    def test_calculate_position(self):
        coords = [[610, 1049], [220, 720], [140, 150]]
        assert navigation.calculate_position(coords, [660, 680, 740]) == [830, 418]
