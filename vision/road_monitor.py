"""isso aq deve encontra os carros sozinho em cada frame e mede direção e desvio.

como usar:
    python -m vision.road_monitor                       # testar com câmera
    python -m vision.road_monitor --video gravacao.mp4  # testar com um vídeo gravado

p = marcar a pista (2 cliques, no sentido do trânsito), q = sair.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import Deteccao
from .rastreador import RastreadorSimples
from .tracking import (
    circular_mean,
    classify_deviation,
    deviation_angle,
    heading_angle,
    motion_heading,
)

CORES_STATUS = {
    "ALINHADO": (0, 200, 0),
    "DESVIADO": (0, 165, 255),
    "CONTRAMAO": (0, 0, 255),
    "SEM PISTA": (255, 255, 255),
    "INCERTO": (180, 180, 180),
}
COR_PISTA = (255, 180, 0)
COR_TRASEIRA = (255, 100, 0)
COR_FRENTE = (0, 0, 255)


# ---------------------------------------------------------------- pista

def carregar_pista(caminho: Path) -> tuple[tuple[float, float], tuple[float, float]] | None:
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return tuple(dados["inicio"]), tuple(dados["fim"])
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return None


def salvar_pista(caminho: Path, inicio, fim) -> None:
    caminho.write_text(
        json.dumps({"inicio": list(inicio), "fim": list(fim)}, indent=2), encoding="utf-8"
    )


def ajustar_para_tela(frame: np.ndarray, altura_max: int, largura_max: int = 1400):
    """Reduz o frame para caber na tela. Devolve (imagem reduzida, escala usada)."""
    altura, largura = frame.shape[:2]
    escala = min(1.0, altura_max / altura, largura_max / largura)
    if escala >= 1.0:
        return frame, 1.0
    tamanho = (int(largura * escala), int(altura * escala))
    return cv2.resize(frame, tamanho, interpolation=cv2.INTER_AREA), escala


def marcar_pista(frame: np.ndarray, altura_max: int = 800):
    """Abre uma janela para clicar dois pontos sobre a pista, no sentido do trânsito."""
    base, escala = ajustar_para_tela(frame, altura_max)
    pontos: list[tuple[float, float]] = []
    janela = "Marque a pista"

    def ao_clicar(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pontos) < 2:
            pontos.append((x / escala, y / escala))
            print(f"  ponto {len(pontos)} de 2 marcado")

    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(janela, ao_clicar)
    print("Marque a pista: clique o INICIO e depois o FIM; Enter confirma (com a janela 'Marque a pista' selecionada).")
    while True:
        tela = base.copy()
        if len(pontos) < 2:
            textos = [f"Clique o {'INICIO' if not pontos else 'FIM'} da pista ({len(pontos)}/2)",
                      "Backspace desfaz | Esc cancela"]
        else:
            textos = ["Enter confirma | Backspace desfaz | Esc cancela", ""]
        for i, texto in enumerate(textos):
            cv2.putText(tela, texto, (10, 25 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(tela, texto, (10, 25 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        pts = [(int(x * escala), int(y * escala)) for x, y in pontos]
        for p in pts:
            cv2.circle(tela, p, 8, COR_PISTA, -1)
        if len(pts) == 2:
            cv2.arrowedLine(tela, pts[0], pts[1], COR_PISTA, 3, tipLength=0.05)
        cv2.imshow(janela, tela)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (10, 13) and len(pontos) == 2:
            cv2.destroyWindow(janela)
            return pontos[0], pontos[1]
        if tecla == 8 and pontos:
            pontos.pop()
        if tecla == 27:
            cv2.destroyWindow(janela)
            return None
        try:
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                return None  # janela fechada no X
        except cv2.error:
            return None


def desenhar_pista(frame: np.ndarray, inicio, fim) -> None:
    p1 = tuple(int(v) for v in inicio)
    p2 = tuple(int(v) for v in fim)
    cv2.arrowedLine(frame, p1, p2, COR_PISTA, 3, tipLength=0.04)
    cv2.putText(frame, "pista", (p1[0] + 8, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_PISTA, 2)


# ---------------------------------------------------------------- análise

def analisar(
    deteccoes: list[Deteccao],
    rastreador: RastreadorSimples,
    angulo_pista: float | None,
    tolerancia: float,
    conf_pontos: float,
    deslocamento_min: float,
) -> list[dict]:
    """Transforma as detecções do frame em um estado por carro (o que os agentes vão perceber)."""
    trilhas = rastreador.atualizar([d.centro for d in deteccoes])
    estados = []
    for deteccao, trilha in zip(deteccoes, trilhas):
        orientacao = None
        if deteccao.pontos_confiaveis(conf_pontos):
            trilha.angulos.append(heading_angle(deteccao.traseira, deteccao.frente))
            orientacao = circular_mean(trilha.angulos)
        movimento = motion_heading(list(trilha.centros), deslocamento_min)

        desvio = None
        if orientacao is None:
            status = "INCERTO"
        elif angulo_pista is None:
            status = "SEM PISTA"
        else:
            desvio = deviation_angle(orientacao, angulo_pista)
            status = classify_deviation(desvio, tolerancia)

        de_re = (
            orientacao is not None
            and movimento is not None
            and abs(deviation_angle(movimento, orientacao)) > 120.0
        )
        estados.append(
            {
                "id": trilha.id,
                "centro": [round(deteccao.centro[0], 1), round(deteccao.centro[1], 1)],
                "confianca": round(deteccao.confianca, 3),
                "orientacao_graus": None if orientacao is None else round(orientacao, 1),
                "movimento_graus": None if movimento is None else round(movimento, 1),
                "parado": movimento is None,
                "de_re": de_re,
                "desvio_graus": None if desvio is None else round(desvio, 1),
                "status": status,
            }
        )
    return estados


def desenhar_carro(frame: np.ndarray, deteccao: Deteccao, estado: dict) -> None:
    cor = CORES_STATUS[estado["status"]]
    x1, y1, x2, y2 = (int(v) for v in deteccao.caixa)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
    traseira = tuple(int(v) for v in deteccao.traseira)
    frente = tuple(int(v) for v in deteccao.frente)
    cv2.arrowedLine(frame, traseira, frente, cor, 4, tipLength=0.25)
    cv2.circle(frame, traseira, 6, COR_TRASEIRA, -1)
    cv2.circle(frame, frente, 6, COR_FRENTE, -1)

    if estado["movimento_graus"] is not None:
        cx, cy = (int(v) for v in deteccao.centro)
        rad = math.radians(estado["movimento_graus"])
        fim = (int(cx + 60 * math.cos(rad)), int(cy + 60 * math.sin(rad)))
        cv2.arrowedLine(frame, (cx, cy), fim, (255, 0, 255), 2, tipLength=0.3)

    linhas = [f"#{estado['id']} {estado['status']}"]
    if estado["orientacao_graus"] is not None:
        linhas.append(f"angulo {estado['orientacao_graus']:+.1f}")
    if estado["desvio_graus"] is not None:
        linhas.append(f"desvio {estado['desvio_graus']:+.1f}")
    linhas.append("parado" if estado["parado"] else ("de re" if estado["de_re"] else "andando"))
    for i, texto in enumerate(linhas):
        y = max(20, y1 - 10 - 22 * (len(linhas) - 1 - i))
        cv2.putText(frame, texto, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, texto, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


# ---------------------------------------------------------------- execução

def criar_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor VigIA: detecta os carros e mede direção e desvio.")
    parser.add_argument("--modelo", default="modelos/vigia-pose.pt", help="Arquivo .pt treinado.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (0 = padrão).")
    parser.add_argument("--video", help="Usa um arquivo de vídeo em vez da câmera.")
    parser.add_argument("--confianca", type=float, default=0.5, help="Confiança mínima da detecção.")
    parser.add_argument("--conf-pontos", type=float, default=0.5, help="Confiança mínima dos pontos frente/traseira.")
    parser.add_argument("--tolerancia", type=float, default=5.0, help="Desvio máximo (graus) para ALINHADO.")
    parser.add_argument("--pista", default="pista.json", help="Arquivo onde a pista marcada é salva.")
    parser.add_argument("--saida-json", help="Grava o estado de cada frame (um JSON por linha).")
    parser.add_argument("--saida-video", help="Grava o vídeo anotado (.mp4).")
    parser.add_argument("--sem-janela", action="store_true", help="Não abre janela (útil para processar vídeos).")
    parser.add_argument("--altura-tela", type=int, default=800, help="Altura máxima da janela, em pixels.")
    return parser


def main(argv: list[str] | None = None, detector=None) -> None:
    parser = criar_argumentos()
    args = parser.parse_args(argv)

    if detector is None:
        if not Path(args.modelo).exists():
            parser.error(
                f"Modelo '{args.modelo}' não encontrado. Treine o modelo (veja o README) "
                "ou informe o caminho com --modelo."
            )
        from .detector import DetectorYOLO

        detector = DetectorYOLO(args.modelo, confianca=args.confianca)

    fonte = args.video if args.video else args.camera
    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        parser.error(f"Não foi possível abrir {'o vídeo' if args.video else 'a câmera'} {fonte}.")

    caminho_pista = Path(args.pista)
    pista = carregar_pista(caminho_pista)
    rastreador: RastreadorSimples | None = None
    arquivo_json = open(args.saida_json, "w", encoding="utf-8") if args.saida_json else None
    gravador = None
    janela = "VigIA - monitor"
    indice_frame = 0
    inicio = time.perf_counter()

    if not args.sem_janela:
        print("Teclas: p = marcar pista | q = sair")

    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            altura, largura = frame.shape[:2]
            if rastreador is None:
                diagonal = math.hypot(largura, altura)
                rastreador = RastreadorSimples(distancia_max=0.15 * diagonal)
                deslocamento_min = 0.02 * diagonal

            deteccoes = detector.detectar(frame)
            angulo_pista = heading_angle(*pista) if pista else None
            estados = analisar(
                deteccoes, rastreador, angulo_pista, args.tolerancia, args.conf_pontos, deslocamento_min
            )

            if pista:
                desenhar_pista(frame, *pista)
            else:
                cv2.putText(frame, "Pressione 'p' para marcar a pista", (12, altura - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for deteccao, estado in zip(deteccoes, estados):
                desenhar_carro(frame, deteccao, estado)
            fps = (indice_frame + 1) / max(1e-6, time.perf_counter() - inicio)
            cv2.putText(frame, f"{len(deteccoes)} carro(s) | {fps:.1f} fps", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if arquivo_json:
                registro = {"frame": indice_frame, "pista_graus": angulo_pista, "carros": estados}
                arquivo_json.write(json.dumps(registro, ensure_ascii=False) + "\n")
            if args.saida_video:
                if gravador is None:
                    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30.0
                    gravador = cv2.VideoWriter(args.saida_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_video, (largura, altura))
                gravador.write(frame)

            indice_frame += 1
            if args.sem_janela:
                continue
            cv2.imshow(janela, ajustar_para_tela(frame, args.altura_tela)[0])
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("p"):
                marcada = marcar_pista(frame, args.altura_tela)
                if marcada:
                    pista = marcada
                    salvar_pista(caminho_pista, *pista)
                    print(f"Pista salva em {caminho_pista} ({heading_angle(*pista):+.1f} graus)")
    finally:
        captura.release()
        if gravador is not None:
            gravador.release()
        if arquivo_json:
            arquivo_json.close()
        cv2.destroyAllWindows()
    print(f"{indice_frame} frames processados.")


if __name__ == "__main__":
    main()