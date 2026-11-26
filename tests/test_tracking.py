import unittest

from vision.rastreador import RastreadorSimples
from vision.tracking import (
    circular_mean,
    classify_deviation,
    deviation_angle,
    heading_angle,
    motion_heading,
)


class AngulosTests(unittest.TestCase):
    def test_heading_is_measured_from_rear_to_front(self) -> None:
        self.assertEqual(heading_angle((0.0, 0.0), (10.0, 0.0)), 0.0)
        self.assertEqual(heading_angle((0.0, 0.0), (0.0, 10.0)), 90.0)
        self.assertEqual(heading_angle((0.0, 0.0), (0.0, -10.0)), -90.0)

    def test_deviation_is_relative_to_reference(self) -> None:
        self.assertEqual(deviation_angle(0.0, 0.0), 0.0)
        self.assertEqual(deviation_angle(12.0, 0.0), 12.0)
        self.assertEqual(deviation_angle(-12.0, 0.0), -12.0)
        self.assertEqual(deviation_angle(180.0, 0.0), -180.0)
        self.assertAlmostEqual(deviation_angle(-175.0, 175.0), 10.0)

    def test_circular_mean_handles_wraparound(self) -> None:
        self.assertAlmostEqual(abs(circular_mean([179.0, -179.0])), 180.0, places=6)
        self.assertAlmostEqual(circular_mean([10.0, 20.0, 30.0]), 20.0, places=6)
        self.assertIsNone(circular_mean([]))

    def test_motion_heading_ignores_small_moves(self) -> None:
        self.assertIsNone(motion_heading([(0, 0), (1, 1)], min_distance=5))
        self.assertEqual(motion_heading([(0, 0), (5, 0), (10, 0)], min_distance=5), 0.0)
        self.assertEqual(motion_heading([(0, 10), (0, 0)], min_distance=5), -90.0)

    def test_classify_deviation(self) -> None:
        self.assertEqual(classify_deviation(3.0, 5.0), "ALINHADO")
        self.assertEqual(classify_deviation(-20.0, 5.0), "DESVIADO")
        self.assertEqual(classify_deviation(170.0, 5.0), "CONTRAMAO")


class RastreadorTests(unittest.TestCase):
    def test_keeps_id_for_nearby_detection(self) -> None:
        rastreador = RastreadorSimples(distancia_max=50)
        primeiro = rastreador.atualizar([(100, 100), (400, 100)])
        segundo = rastreador.atualizar([(405, 102), (104, 98)])
        self.assertEqual(segundo[0].id, primeiro[1].id)
        self.assertEqual(segundo[1].id, primeiro[0].id)

    def test_new_id_when_far_and_forgets_after_tolerance(self) -> None:
        rastreador = RastreadorSimples(distancia_max=50, frames_tolerados=2)
        a = rastreador.atualizar([(0, 0)])[0]
        b = rastreador.atualizar([(500, 500)])[0]
        self.assertNotEqual(a.id, b.id)
        for _ in range(3):
            rastreador.atualizar([])
        self.assertEqual(rastreador.trilhas, {})


if __name__ == "__main__":
    unittest.main()
