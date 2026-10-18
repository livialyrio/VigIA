"""Ferramenta para rotular carros: caixa + ponto da traseira + ponto da frente.

    python -m ferramentas.rotular
    python -m ferramentas.rotular --modelo modelos/vigia-pose.pt   # o modelo sugere, você corrige

Para cada carro na imagem:
  1. arraste com o mouse uma caixa em volta do carro inteiro;
  2. clique no centro da TRASEIRA;
  3. clique no centro da FRENTE.
Pode marcar vários carros na mesma imagem.

Teclas:
  Enter/Espaço  salva e vai para a próxima
  n             imagem SEM carro (salva vazio, também ensina o modelo) e vai para a próxima
  z             desfaz o último passo
  a / d         imagem anterior / próxima (salva o que mudou)
  x             apaga o rótulo desta imagem (volta a ficar "não rotulada")
  q / Esc       salva e sai
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .formato_yolo import Rotulo, escrever_arquivo, ler_arquivo

COR_CAIXA = (0, 220, 0)
COR_TRASEIRA = (255, 100, 0)
COR_FRENTE = (0, 0, 255)
COR_SUGESTAO = (0, 220, 220)
ALTURA_FAIXA = 100

ESPERANDO_CAIXA = "arraste uma caixa em volta do carro"
ESPERANDO_TRASEIRA = "clique na TRASEIRA"
ESPERANDO_FRENTE = "clique na FRENTE"


class EditorRotulos:
    """Estado da edição de uma imagem (separado da janela para poder ser testado)."""

    def __init__(self, rotulos: list[Rotulo] | None = None, sugestao: bool = False) -> None:
        self.rotulos: list[Rotulo] = list(rotulos or [])
        self.sugestao = sugestao
        self.modificado = sugestao
        self.caixa_inicio: tuple[float, float] | None = None
        self.caixa_atual: tuple[float, float, float, float] | None = None
        self.traseira: tuple[float, float] | None = None
        self.arrastando = False
        self._caixa_temporaria: tuple[float, float, float, float] | None = None

    @property
    def etapa(self) -> str:
        if self.caixa_atual is None:
            return ESPERANDO_CAIXA
        if self.traseira is None:
            return ESPERANDO_TRASEIRA
        return ESPERANDO_FRENTE

    @property
    def incompleto(self) -> bool:
        return self.caixa_atual is not None

    def mouse(self, evento: int, x: float, y: float) -> None:
        if self.etapa == ESPERANDO_CAIXA:
            if evento == cv2.EVENT_LBUTTONDOWN:
                self.arrastando = True
                self.caixa_inicio = (x, y)
            elif evento == cv2.EVENT_MOUSEMOVE and self.arrastando:
                self._caixa_temporaria = (*self.caixa_inicio, x, y)
            elif evento == cv2.EVENT_LBUTTONUP and self.arrastando:
                self.arrastando = False
                x0, y0 = self.caixa_inicio
                if abs(x - x0) >= 5 and abs(y - y0) >= 5:
                    self.caixa_atual = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
                self.caixa_inicio = None
                self._caixa_temporaria = None
        elif evento == cv2.EVENT_LBUTTONDOWN:
            if self.etapa == ESPERANDO_TRASEIRA:
                self.traseira = (x, y)
            else:
                self.rotulos.append(Rotulo(caixa=self.caixa_atual, traseira=self.traseira, frente=(x, y)))
                self.caixa_atual = None
                self.traseira = None
                self.modificado = True
                self.sugestao = False

    def desfazer(self) -> None:
        if self.traseira is not None:
            self.traseira = None
        elif self.caixa_atual is not None:
            self.caixa_atual = None
        elif self.rotulos:
            self.rotulos.pop()
            self.modificado = True

    def caixa_em_desenho(self):
        return self._caixa_temporaria


def desenhar(imagem: np.ndarray, editor: EditorRotulos, escala: float, cabecalho: tuple[str, str]) -> np.ndarray:
    tela = cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    def p(ponto):
        return int(ponto[0] * escala), int(ponto[1] * escala)

    cor_caixa = COR_SUGESTAO if editor.sugestao else COR_CAIXA
    for numero, rotulo in enumerate(editor.rotulos, 1):
        x1, y1, x2, y2 = rotulo.caixa
        cv2.rectangle(tela, p((x1, y1)), p((x2, y2)), cor_caixa, 2)
        cv2.arrowedLine(tela, p(rotulo.traseira), p(rotulo.frente), cor_caixa, 2, tipLength=0.25)
        cv2.circle(tela, p(rotulo.traseira), 6, COR_TRASEIRA, -1)
        cv2.circle(tela, p(rotulo.frente), 6, COR_FRENTE, -1)
        cv2.putText(tela, str(numero), p((x1 + 4, y1 + 22 / escala)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_caixa, 2)
    temporaria = editor.caixa_em_desenho()
    if temporaria:
        x1, y1, x2, y2 = temporaria
        cv2.rectangle(tela, p((x1, y1)), p((x2, y2)), (200, 200, 200), 1)
    if editor.caixa_atual:
        x1, y1, x2, y2 = editor.caixa_atual
        cv2.rectangle(tela, p((x1, y1)), p((x2, y2)), (0, 255, 255), 2)
    if editor.traseira:
        cv2.circle(tela, p(editor.traseira), 6, COR_TRASEIRA, -1)

    faixa = np.zeros((ALTURA_FAIXA, tela.shape[1], 3), dtype=np.uint8)
    instrucao = editor.etapa
    if editor.sugestao:
        instrucao = "SUGESTAO DO MODELO: Enter aceita, z remove o ultimo"
    linhas = [
        (cabecalho[0], (255, 255, 255)),
        (cabecalho[1], (255, 255, 255)),
        (f"> {instrucao}", (0, 255, 255)),
        ("Enter salva | n sem carro | z desfaz | a/d navega | q sai", (170, 170, 170)),
    ]
    for i, (texto, cor) in enumerate(linhas):
        cv2.putText(faixa, texto, (10, 20 + 23 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
    return np.vstack([faixa, tela])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagens", type=Path, default=Path("dados/imagens"))
    parser.add_argument("--rotulos", type=Path, default=Path("dados/rotulos"))
    parser.add_argument("--modelo", help="Modelo .pt para sugerir rótulos nas imagens ainda não rotuladas.")
    parser.add_argument("--filtro", default="", help="Só imagens cujo nome contém este texto (ex.: Decocco).")
    parser.add_argument("--altura-tela", type=int, default=750, help="Altura máxima da janela, em pixels.")
    args = parser.parse_args(argv)

    imagens = sorted(p for p in args.imagens.glob("*.jpg") if args.filtro in p.name)
    if not imagens:
        parser.error(f"Nenhuma imagem em {args.imagens}. Rode antes: python -m ferramentas.preparar_imagens <pasta>")
    args.rotulos.mkdir(parents=True, exist_ok=True)

    detector = None
    if args.modelo:
        from vision.detector import DetectorYOLO

        detector = DetectorYOLO(args.modelo, confianca=0.4)

    def caminho_rotulo(imagem: Path) -> Path:
        return args.rotulos / f"{imagem.stem}.txt"

    def contar_rotuladas() -> int:
        return sum(caminho_rotulo(i).exists() for i in imagens)

    indice = next((i for i, img in enumerate(imagens) if not caminho_rotulo(img).exists()), 0)
    janela = "VigIA - rotulagem"
    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)

    while 0 <= indice < len(imagens):
        caminho = imagens[indice]
        imagem = cv2.imread(str(caminho))
        altura, largura = imagem.shape[:2]
        escala = min(1.0, args.altura_tela / altura, 1400 / largura)
        arquivo = caminho_rotulo(caminho)
        if arquivo.exists():
            editor = EditorRotulos(ler_arquivo(arquivo, largura, altura))
        elif detector is not None:
            sugeridos = [
                Rotulo(caixa=d.caixa, traseira=d.traseira, frente=d.frente) for d in detector.detectar(imagem)
            ]
            editor = EditorRotulos(sugeridos, sugestao=bool(sugeridos))
        else:
            editor = EditorRotulos()

        def ao_mouse(evento, x, y, *_):
            editor.mouse(evento, x / escala, (y - ALTURA_FAIXA) / escala)

        cv2.setMouseCallback(janela, ao_mouse)
        total_rotuladas = contar_rotuladas()
        proximo = indice
        while proximo == indice:
            situacao = "rotulada" if arquivo.exists() else "NAO rotulada"
            cabecalho = (
                f"{indice + 1}/{len(imagens)}  {caminho.name}",
                f"[{situacao}]  carros aqui: {len(editor.rotulos)}  |  rotuladas: {total_rotuladas}",
            )
            cv2.imshow(janela, desenhar(imagem, editor, escala, cabecalho))
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == 255:
                continue

            def salvar(rotulos):
                escrever_arquivo(arquivo, rotulos, largura, altura)

            if tecla in (13, 10, 32):
                if editor.incompleto:
                    print("Termine o carro em andamento (traseira e frente) ou aperte z.")
                    continue
                salvar(editor.rotulos)
                proximo = indice + 1
            elif tecla == ord("n"):
                salvar([])
                proximo = indice + 1
            elif tecla == ord("z"):
                editor.desfazer()
            elif tecla == ord("x"):
                arquivo.unlink(missing_ok=True)
                total_rotuladas = contar_rotuladas()
                editor = EditorRotulos()
            elif tecla in (ord("a"), ord("d"), ord("q"), 27):
                if editor.modificado and not editor.incompleto and not editor.sugestao:
                    salvar(editor.rotulos)
                if tecla == ord("a"):
                    proximo = max(0, indice - 1)
                    if proximo == indice:
                        continue
                elif tecla == ord("d"):
                    proximo = indice + 1
                else:
                    proximo = -1
        indice = proximo

    cv2.destroyAllWindows()
    print(f"{contar_rotuladas()} de {len(imagens)} imagens rotuladas.")


if __name__ == "__main__":
    main()
