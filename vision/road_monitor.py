import argparse
import math

import cv2
import numpy as np

if __package__:
    from .tracking import deviation_angle, heading_angle
else:
    from tracking import deviation_angle, heading_angle


def select_rear_and_front(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = frame.shape[:2]
    scale = min(1.0, 1100 / max(height, width))
    preview_size = (int(width * scale), int(height * scale))
    points: list[tuple[float, float]] = []
    window = "Marque traseira e frente do carrinho"

    def on_mouse(event: int, x: int, y: int, flags: int, parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / scale, y / scale))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = cv2.resize(frame, preview_size)
        instructions = [
            "Clique primeiro no centro da TRASEIRA, depois no centro da FRENTE.",
            "Escolha pontos com detalhes visiveis no carrinho. Enter confirma; Esc cancela.",
        ]
        for index, text in enumerate(instructions):
            cv2.putText(
                preview,
                text,
                (12, 26 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        for index, point in enumerate(points):
            location = (int(point[0] * scale), int(point[1] * scale))
            color = (255, 100, 0) if index == 0 else (0, 0, 255)
            cv2.circle(preview, location, 7, color, -1)
            cv2.putText(
                preview,
                "Traseira" if index == 0 else "Frente",
                (location[0] + 8, location[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) == 2:
            cv2.destroyWindow(window)
            return (
                np.asarray(points[0], dtype=np.float32),
                np.asarray(points[1], dtype=np.float32),
            )
        if key == 8 and points:
            points.pop()
        if key == 27 or key == ord("q"):
            cv2.destroyWindow(window)
            return None


def seed_features(gray: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    mask = np.zeros_like(gray)
    center = tuple(np.round(point).astype(int))
    cv2.circle(mask, center, 32, 255, -1)
    features = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=7,
        mask=mask,
    )
    if features is None or len(features) < 3:
        return None
    return features.astype(np.float32)


def follow_landmark(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    point: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    next_features, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        features,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_features is None or forward_status is None:
        return None

    previous_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_features,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if previous_points is None or backward_status is None:
        return None

    old_points = features.reshape(-1, 2)
    new_points = next_features.reshape(-1, 2)
    returned_points = previous_points.reshape(-1, 2)
    valid = (forward_status.ravel() == 1) & (backward_status.ravel() == 1)
    valid &= np.linalg.norm(old_points - returned_points, axis=1) < 2.0
    if np.count_nonzero(valid) < 3:
        return None

    displacement = np.median(new_points[valid] - old_points[valid], axis=0)
    new_landmark = point + displacement
    return new_landmark.astype(np.float32), new_points[valid].reshape(-1, 1, 2)


def draw_reference_line(frame: np.ndarray, angle: float) -> None:
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    length = int(math.hypot(width, height))
    radians = math.radians(angle)
    delta_x = int(length * math.cos(radians))
    delta_y = int(length * math.sin(radians))
    end = (center_x + delta_x, center_y + delta_y)
    cv2.arrowedLine(frame, (center_x, center_y), end, (255, 180, 0), 2)
    cv2.putText(
        frame,
        "Direcao da pista",
        (12, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 180, 0),
        2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acompanha a direcao do carrinho ao vivo, sem modelo treinado."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice da camera; normalmente 0 para a camera padrao.",
    )
    parser.add_argument(
        "--pista-graus",
        type=float,
        default=0.0,
        help="Angulo da pista na imagem em graus; 0 significa para a direita.",
    )
    parser.add_argument("--tolerancia", type=float, default=5.0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        parser.error(
            f"Nao foi possivel abrir a camera {args.camera}. "
            "Conecte uma camera ou informe outro indice com --camera."
        )

    reference_angle = args.pista_graus
    window_name = "Monitoramento do carrinho"
    print("Camera iniciada. Pressione 'r' para marcar/redefinir frente e traseira; 'q' para sair.")
    previous_gray: np.ndarray | None = None
    rear: np.ndarray | None = None
    front: np.ndarray | None = None
    rear_features: np.ndarray | None = None
    front_features: np.ndarray | None = None
    tracking_lost = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("A camera nao forneceu um frame.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if rear is None or front is None:
                cv2.putText(
                    frame,
                    "Pressione 'r' para marcar a traseira e a frente do carrinho",
                    (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    selected = select_rear_and_front(frame)
                    if selected is not None:
                        rear, front = selected
                        previous_gray = gray
                        rear_features = seed_features(gray, rear)
                        front_features = seed_features(gray, front)
                        tracking_lost = rear_features is None or front_features is None
                        if tracking_lost:
                            rear = None
                            front = None
                continue

            tracking_ok = (
                not tracking_lost
                and previous_gray is not None
                and rear_features is not None
                and front_features is not None
            )
            if tracking_ok:
                rear_result = follow_landmark(
                    previous_gray, gray, rear, rear_features
                )
                front_result = follow_landmark(
                    previous_gray, gray, front, front_features
                )
                tracking_ok = rear_result is not None and front_result is not None
                if tracking_ok:
                    rear, rear_features = rear_result
                    front, front_features = front_result
                    refreshed_rear = seed_features(gray, rear)
                    refreshed_front = seed_features(gray, front)
                    if refreshed_rear is not None:
                        rear_features = refreshed_rear
                    if refreshed_front is not None:
                        front_features = refreshed_front
                else:
                    tracking_lost = True

            if tracking_ok:
                angle = heading_angle(tuple(rear), tuple(front))
                deviation = deviation_angle(angle, reference_angle)
                status = "ALINHADO" if abs(deviation) <= args.tolerancia else "DESVIADO"
                direction_text = (
                    "desvio para baixo" if deviation > 0 else "desvio para cima"
                )
                cv2.arrowedLine(
                    frame,
                    tuple(np.round(rear).astype(int)),
                    tuple(np.round(front).astype(int)),
                    (0, 255, 0),
                    4,
                    tipLength=0.2,
                )
                cv2.circle(frame, tuple(np.round(rear).astype(int)), 7, (255, 100, 0), -1)
                cv2.circle(frame, tuple(np.round(front).astype(int)), 7, (0, 0, 255), -1)
                label = f"{status} | desvio {deviation:+.1f} graus"
                if abs(deviation) > args.tolerancia:
                    label += f" ({direction_text})"
            else:
                label = "RASTREAMENTO PERDIDO | pressione 'r' para remarcar"

            draw_reference_line(frame, reference_angle)
            cv2.putText(
                frame,
                label,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if tracking_ok else (0, 0, 255),
                2,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                selected = select_rear_and_front(frame)
                if selected is not None:
                    rear, front = selected
                    rear_features = seed_features(gray, rear)
                    front_features = seed_features(gray, front)
                    tracking_lost = rear_features is None or front_features is None
                    if tracking_lost:
                        rear = None
                        front = None
                    previous_gray = gray
            elif tracking_ok:
                tracking_lost = False
            previous_gray = gray
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()