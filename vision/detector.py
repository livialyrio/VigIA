"""Detector de carros baseado no modelo YOLO de pontos-chave treinado.

O modelo devolve, para cada carro encontrado na imagem, uma caixa e dois
pontos-chave: 0 = traseira, 1 = frente.
"""

from dataclasses import dataclass

import numpy as np

KEYPOINT_REAR = 0
KEYPOINT_FRONT = 1


@dataclass
class Deteccao:
    caixa: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
    confianca: float
    traseira: tuple[float, float]
    frente: tuple[float, float]
    confianca_traseira: float
    confianca_frente: float

    @property
    def centro(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.caixa
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def pontos_confiaveis(self, minimo: float) -> bool:
        return self.confianca_traseira >= minimo and self.confianca_frente >= minimo


class DetectorYOLO:
    def __init__(self, caminho_modelo: str, confianca: float = 0.5, tamanho: int = 640) -> None:
        from ultralytics import YOLO  # importado aqui para os testes não exigirem a biblioteca

        self.modelo = YOLO(caminho_modelo)
        self.confianca = confianca
        self.tamanho = tamanho

    def detectar(self, frame: np.ndarray) -> list[Deteccao]:
        resultado = self.modelo.predict(
            frame, conf=self.confianca, imgsz=self.tamanho, verbose=False
        )[0]
        return converter_resultado(resultado)


def converter_resultado(resultado) -> list[Deteccao]:
    """Converte um `ultralytics.engine.results.Results` em uma lista de Deteccao."""
    if resultado.boxes is None or len(resultado.boxes) == 0 or resultado.keypoints is None:
        return []
    caixas = resultado.boxes.xyxy.cpu().numpy()
    confiancas = resultado.boxes.conf.cpu().numpy()
    pontos = resultado.keypoints.xy.cpu().numpy()
    if resultado.keypoints.conf is not None:
        conf_pontos = resultado.keypoints.conf.cpu().numpy()
    else:
        conf_pontos = np.ones(pontos.shape[:2], dtype=np.float32)

    deteccoes = []
    for caixa, conf, kp, kp_conf in zip(caixas, confiancas, pontos, conf_pontos):
        deteccoes.append(
            Deteccao(
                caixa=tuple(float(v) for v in caixa),
                confianca=float(conf),
                traseira=(float(kp[KEYPOINT_REAR][0]), float(kp[KEYPOINT_REAR][1])),
                frente=(float(kp[KEYPOINT_FRONT][0]), float(kp[KEYPOINT_FRONT][1])),
                confianca_traseira=float(kp_conf[KEYPOINT_REAR]),
                confianca_frente=float(kp_conf[KEYPOINT_FRONT]),
            )
        )
    return deteccoes
