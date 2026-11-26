"""Contas de ângulo e direção usadas pelo monitor.

Convenção (coordenadas da imagem): 0° aponta para a direita, +90° para baixo,
-90° para cima e ±180° para a esquerda.
"""

import math
from collections.abc import Iterable, Sequence

Point = tuple[float, float]


def normalize_direction_angle(angle: float) -> float:
    """Normaliza um ângulo direcional para o intervalo [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def heading_angle(rear: Point, front: Point) -> float:
    """Ângulo do vetor traseira -> frente em relação ao eixo x da imagem."""
    delta_x = front[0] - rear[0]
    delta_y = front[1] - rear[1]
    return normalize_direction_angle(math.degrees(math.atan2(delta_y, delta_x)))


def deviation_angle(angle: float, reference_angle: float) -> float:
    """Quanto `angle` está girado em relação a `reference_angle`, em [-180, 180)."""
    return normalize_direction_angle(angle - reference_angle)


def circular_mean(angles: Iterable[float]) -> float | None:
    """Média de ângulos em graus que respeita a volta em ±180°."""
    sum_x = sum_y = 0.0
    count = 0
    for angle in angles:
        radians = math.radians(angle)
        sum_x += math.cos(radians)
        sum_y += math.sin(radians)
        count += 1
    if count == 0 or math.hypot(sum_x, sum_y) < 1e-9:
        return None
    return normalize_direction_angle(math.degrees(math.atan2(sum_y, sum_x)))


def motion_heading(path: Sequence[Point], min_distance: float) -> float | None:
    """Direção do deslocamento entre o ponto mais antigo e o mais recente.

    Retorna None se o objeto andou menos que `min_distance` pixels (parado).
    """
    if len(path) < 2:
        return None
    start, end = path[0], path[-1]
    if math.dist(start, end) < min_distance:
        return None
    return heading_angle(start, end)


def classify_deviation(deviation: float, tolerance: float) -> str:
    """ALINHADO, DESVIADO ou CONTRAMAO conforme o desvio em relação à pista."""
    magnitude = abs(deviation)
    if magnitude <= tolerance:
        return "ALINHADO"
    if magnitude > 90.0:
        return "CONTRAMAO"
    return "DESVIADO"
