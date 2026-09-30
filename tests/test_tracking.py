import unittest

from vision.tracking import (
    deviation_angle,
    heading_angle,
)


class TrackingTests(unittest.TestCase):
    def test_heading_is_measured_from_rear_to_front(self) -> None:
        self.assertEqual(heading_angle((0.0, 0.0), (10.0, 0.0)), 0.0)
        self.assertEqual(heading_angle((0.0, 0.0), (0.0, 10.0)), 90.0)
        self.assertEqual(heading_angle((0.0, 0.0), (0.0, -10.0)), -90.0)

    def test_deviation_is_relative_to_rightward_track_axis(self) -> None:
        self.assertEqual(deviation_angle(0.0, 0.0), 0.0)
        self.assertEqual(deviation_angle(12.0, 0.0), 12.0)
        self.assertEqual(deviation_angle(-12.0, 0.0), -12.0)
        self.assertEqual(deviation_angle(180.0, 0.0), -180.0)


if __name__ == "__main__":
    unittest.main()