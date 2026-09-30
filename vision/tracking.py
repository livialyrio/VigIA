import math


def normalize_direction_angle(angle: float) -> float:
    """Normalize a directed movement angle to [-180, 180) degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def heading_angle(
    rear: tuple[float, float], front: tuple[float, float]
) -> float:
    delta_x = front[0] - rear[0]
    delta_y = front[1] - rear[1]
    return normalize_direction_angle(math.degrees(math.atan2(delta_y, delta_x)))


def deviation_angle(angle: float, reference_angle: float) -> float:
    return normalize_direction_angle(angle - reference_angle)