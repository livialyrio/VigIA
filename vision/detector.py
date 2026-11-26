"""Detector de carros baseado no modelo YOLO de pontos-chave treinado.

O modelo devolve, para cada carro encontrado na imagem, uma caixa e três
pontos-chave: 0 = traseira, 1 = meio, 2 = frente.
"""

from dataclasses import dataclass

import numpy as np

KEYPOINT_REAR = 0
KEYPOINT_MIDDLE = 1
KEYPOINT_FRONT = 2
NUM_KEYPOINTS = 3


@dataclass
class Deteccao:
    caixa: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
    confianca: float
    traseira: tuple[float, float]
    meio: tuple[float, float]
    frente: tuple[float, float]
    confianca_traseira: float
    confianca_meio: float
    confianca_frente: float

    @property
    def centro(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.caixa
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def pontos_confiaveis(self, minimo: float) -> bool:
        """True se os três pontos têm confiança suficiente."""
        return min(self.confianca_traseira, self.confianca_meio, self.confianca_frente) >= minimo

    def eixo_confiavel(self, minimo: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Melhor par (ponto de trás, ponto da frente) para medir a orientação.

        Usa traseira → frente quando os dois são confiáveis. Se um deles falhar
        (carro cortado na imagem, por exemplo), usa o meio no lugar dele.
        """
        tras_ok = self.confianca_traseira >= minimo
        meio_ok = self.confianca_meio >= minimo
        frente_ok = self.confianca_frente >= minimo
        if tras_ok and frente_ok:
            return self.traseira, self.frente
        if meio_ok and frente_ok:
            return self.meio, self.frente
        if tras_ok and meio_ok:
            return self.traseira, self.meio
        return None


class DetectorYOLO:
    def __init__(self, caminho_modelo: str, confianca: float = 0.5, tamanho: int = 640) -> None:
        from ultralytics import YOLO  # importado aqui para os testes não exigirem a biblioteca

        self.modelo = YOLO(caminho_modelo)
        kpt_shape = getattr(self.modelo.model, "kpt_shape", None) or self.modelo.model.yaml.get("kpt_shape")
        if kpt_shape is not None and kpt_shape[0] != NUM_KEYPOINTS:
            raise ValueError(
                f"O modelo '{caminho_modelo}' marca {kpt_shape[0]} pontos por carro, mas o VigIA agora usa "
                f"{NUM_KEYPOINTS} (traseira, meio, frente). Retreine o modelo com os novos rótulos."
            )
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
    if pontos.shape[1] != NUM_KEYPOINTS:
        raise ValueError(
            f"O modelo devolveu {pontos.shape[1]} pontos por carro; esperava {NUM_KEYPOINTS} "
            "(traseira, meio, frente). Retreine o modelo com os novos rótulos."
        )
    if resultado.keypoints.conf is not None:
        conf_pontos = resultado.keypoints.conf.cpu().numpy()
    else:
        conf_pontos = np.ones(pontos.shape[:2], dtype=np.float32)

    def ponto(kp, indice):
        return (float(kp[indice][0]), float(kp[indice][1]))

    deteccoes = []
    for caixa, conf, kp, kp_conf in zip(caixas, confiancas, pontos, conf_pontos):
        deteccoes.append(
            Deteccao(
                caixa=tuple(float(v) for v in caixa),
                confianca=float(conf),
                traseira=ponto(kp, KEYPOINT_REAR),
                meio=ponto(kp, KEYPOINT_MIDDLE),
                frente=ponto(kp, KEYPOINT_FRONT),
                confianca_traseira=float(kp_conf[KEYPOINT_REAR]),
                confianca_meio=float(kp_conf[KEYPOINT_MIDDLE]),
                confianca_frente=float(kp_conf[KEYPOINT_FRONT]),
            )
        )
    return deteccoes
