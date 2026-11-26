import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ferramentas import dividir_dataset, preparar_imagens
from ferramentas.formato_yolo import Rotulo, de_linha, escrever_arquivo, ler_arquivo, para_linha
from ferramentas.rotular import (
    ESPERANDO_CAIXA,
    ESPERANDO_FRENTE,
    ESPERANDO_MEIO,
    ESPERANDO_TRASEIRA,
    EditorRotulos,
)


class FormatoTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        rotulo = Rotulo(caixa=(100, 50, 300, 250), traseira=(120, 150), meio=(200, 145), frente=(280, 140))
        linha = para_linha(rotulo, 640, 480)
        self.assertEqual(len(linha.split()), 14)
        volta = de_linha(linha, 640, 480)
        for a, b in zip(volta.caixa, rotulo.caixa):
            self.assertAlmostEqual(a, b, places=2)
        self.assertAlmostEqual(volta.frente[0], 280, places=2)
        self.assertAlmostEqual(volta.meio[0], 200, places=2)
        self.assertAlmostEqual(volta.meio[1], 145, places=2)

    def test_old_two_point_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            de_linha("0 0.5 0.5 0.2 0.2 0.4 0.5 2 0.6 0.5 2", 640, 480)

    def test_empty_file_means_no_car(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "x.txt"
            escrever_arquivo(caminho, [], 640, 480)
            self.assertEqual(caminho.read_text(), "")
            self.assertEqual(ler_arquivo(caminho, 640, 480), [])


class EditorTests(unittest.TestCase):
    def test_box_then_rear_then_middle_then_front(self) -> None:
        editor = EditorRotulos()
        self.assertEqual(editor.etapa, ESPERANDO_CAIXA)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
        editor.mouse(cv2.EVENT_MOUSEMOVE, 60, 40)
        editor.mouse(cv2.EVENT_LBUTTONUP, 100, 80)
        self.assertEqual(editor.etapa, ESPERANDO_TRASEIRA)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 20, 45)
        self.assertEqual(editor.etapa, ESPERANDO_MEIO)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 55, 45)
        self.assertEqual(editor.etapa, ESPERANDO_FRENTE)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 90, 45)
        self.assertEqual(editor.etapa, ESPERANDO_CAIXA)
        self.assertEqual(len(editor.rotulos), 1)
        self.assertEqual(editor.rotulos[0].caixa, (10, 10, 100, 80))
        self.assertEqual(editor.rotulos[0].meio, (55, 45))
        editor.desfazer()
        self.assertEqual(editor.rotulos, [])

    def test_undo_middle_point(self) -> None:
        editor = EditorRotulos()
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
        editor.mouse(cv2.EVENT_LBUTTONUP, 100, 80)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 20, 45)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 55, 45)
        editor.desfazer()
        self.assertEqual(editor.etapa, ESPERANDO_MEIO)
        self.assertEqual(editor.traseira, (20, 45))

    def test_tiny_drag_is_ignored(self) -> None:
        editor = EditorRotulos()
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
        editor.mouse(cv2.EVENT_LBUTTONUP, 12, 11)
        self.assertEqual(editor.etapa, ESPERANDO_CAIXA)


class DivisaoTests(unittest.TestCase):
    def test_frames_of_same_video_stay_together(self) -> None:
        nomes = [f"Decocco_VID{v}_f{i:05d}" for v in range(5) for i in range(0, 60, 15)]
        nomes += [f"Samu_IMG{i}" for i in range(10)]
        treino, val = dividir_dataset.dividir(nomes, 0.2, semente=1)
        self.assertEqual(len(treino) + len(val), len(nomes))
        grupos_treino = {dividir_dataset.grupo_da_imagem(n) for n in treino}
        grupos_val = {dividir_dataset.grupo_da_imagem(n) for n in val}
        self.assertFalse(grupos_treino & grupos_val)

    def test_sequential_frames_are_grouped_in_blocks(self) -> None:
        nomes = [f"frames_dataset_novo_frame_{i:05d}" for i in range(300)]
        self.assertEqual(dividir_dataset.grupo_da_imagem(nomes[0]), dividir_dataset.grupo_da_imagem(nomes[29]))
        self.assertNotEqual(dividir_dataset.grupo_da_imagem(nomes[29]), dividir_dataset.grupo_da_imagem(nomes[30]))
        treino, val = dividir_dataset.dividir(nomes, 0.2, semente=1)
        self.assertEqual(len(val), 60)
        grupos_treino = {dividir_dataset.grupo_da_imagem(n) for n in treino}
        grupos_val = {dividir_dataset.grupo_da_imagem(n) for n in val}
        self.assertFalse(grupos_treino & grupos_val)


class PrepararTests(unittest.TestCase):
    def test_photos_and_video_frames(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            origem = Path(pasta) / "Decocco"
            origem.mkdir()
            cv2.imwrite(str(origem / "foto.jpg"), np.full((2000, 1500, 3), 128, np.uint8))
            gravador = cv2.VideoWriter(str(origem / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
            for i in range(30):
                gravador.write(np.full((240, 320, 3), i * 8, np.uint8))
            gravador.release()
            saida = Path(pasta) / "saida"
            preparar_imagens.main([str(origem), "--saida", str(saida), "--a-cada", "10"])
            nomes = sorted(p.name for p in saida.glob("*.jpg"))
            self.assertIn("Decocco_foto.jpg", nomes)
            self.assertEqual(sum(n.startswith("Decocco_v_f") for n in nomes), 3)
            foto = cv2.imread(str(saida / "Decocco_foto.jpg"))
            self.assertEqual(max(foto.shape[:2]), 1280)


if __name__ == "__main__":
    unittest.main()
