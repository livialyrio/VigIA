#Roda o monitor de ponta a ponta num vídeo sintético, com um detector falso no lugar do YOLO.

import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision import road_monitor
from vision.detector import Deteccao


def posicao(frame: int) -> tuple[float, float]:
    return 100 + 8 * frame, 200.0  # carro andando para a direita


class DetectorFalso:
    """Devolve um carro que anda para a direita, levemente inclinado (+10°)."""

    def __init__(self) -> None:
        self.frame = 0

    def detectar(self, _imagem):
        cx, cy = posicao(self.frame)
        self.frame += 1
        rad = math.radians(10)
        meio = 30
        return [
            Deteccao(
                caixa=(cx - 40, cy - 25, cx + 40, cy + 25),
                confianca=0.9,
                traseira=(cx - meio * math.cos(rad), cy - meio * math.sin(rad)),
                frente=(cx + meio * math.cos(rad), cy + meio * math.sin(rad)),
                confianca_traseira=0.9,
                confianca_frente=0.9,
            )
        ]


class MonitorTests(unittest.TestCase):
    def test_end_to_end_with_fake_detector(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            video = pasta / "entrada.mp4"
            gravador = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 400))
            for _ in range(20):
                gravador.write(np.zeros((400, 640, 3), np.uint8))
            gravador.release()

            pista = pasta / "pista.json"
            pista.write_text(json.dumps({"inicio": [0, 300], "fim": [600, 300]}))
            saida = pasta / "estados.jsonl"
            road_monitor.main(
                ["--video", str(video), "--sem-janela", "--pista", str(pista),
                 "--saida-json", str(saida), "--saida-video", str(pasta / "anotado.mp4"), "--tolerancia", "5"],
                detector=DetectorFalso(),
            )

            registros = [json.loads(l) for l in saida.read_text().splitlines()]
            self.assertEqual(len(registros), 20)
            ultimo = registros[-1]["carros"][0]
            self.assertEqual(ultimo["id"], 1)  # mesmo carro o vídeo inteiro
            self.assertAlmostEqual(ultimo["orientacao_graus"], 10.0, places=1)
            self.assertAlmostEqual(ultimo["desvio_graus"], 10.0, places=1)
            self.assertEqual(ultimo["status"], "DESVIADO")
            self.assertEqual(ultimo["movimento_graus"], 0.0)
            self.assertFalse(ultimo["parado"])
            self.assertFalse(ultimo["de_re"])
            self.assertTrue((pasta / "anotado.mp4").stat().st_size > 0)


class _Tensor:
    def __init__(self, valores):
        self._v = np.asarray(valores, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._v


class _Caixas:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _Tensor(xyxy), _Tensor(conf)

    def __len__(self):
        return len(self.xyxy.numpy())


class _Pontos:
    def __init__(self, xy, conf):
        self.xy, self.conf = _Tensor(xy), _Tensor(conf)


class _Resultado:
    def __init__(self, boxes, keypoints):
        self.boxes, self.keypoints = boxes, keypoints


class ConversaoTests(unittest.TestCase):
    """Imita o objeto Results do Ultralytics (mesmos nomes de atributos)."""

    def test_converts_boxes_and_keypoints(self) -> None:
        from vision.detector import converter_resultado

        resultado = _Resultado(
            _Caixas([[10, 20, 110, 80]], [0.87]),
            _Pontos([[[30, 50], [90, 45]]], [[0.95, 0.4]]),
        )
        [d] = converter_resultado(resultado)
        self.assertEqual(d.caixa, (10.0, 20.0, 110.0, 80.0))
        self.assertEqual(d.traseira, (30.0, 50.0))
        self.assertEqual(d.frente, (90.0, 45.0))
        self.assertEqual(d.centro, (60.0, 50.0))
        self.assertTrue(d.pontos_confiaveis(0.3))
        self.assertFalse(d.pontos_confiaveis(0.5))

    def test_no_detection(self) -> None:
        from vision.detector import converter_resultado

        vazio = _Resultado(_Caixas(np.zeros((0, 4)), []), _Pontos(np.zeros((0, 2, 2)), np.zeros((0, 2))))
        self.assertEqual(converter_resultado(vazio), [])


if __name__ == "__main__":
    unittest.main()
