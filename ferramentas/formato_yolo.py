"""Leitura e escrita de rótulos no formato YOLO pose.

Cada linha de um arquivo .txt descreve um carro:
    classe cx cy w h  x_tras y_tras v_tras  x_frente y_frente v_frente
Todos os valores de posição são normalizados (0 a 1) pelo tamanho da imagem.
v = 2 significa ponto visível. Arquivo vazio = imagem sem carro (exemplo negativo).
"""

from dataclasses import dataclass
from pathlib import Path

CLASSES = ["carro"]
NOMES_PONTOS = ["traseira", "frente"]


@dataclass
class Rotulo:
    caixa: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
    traseira: tuple[float, float]
    frente: tuple[float, float]
    classe: int = 0


def _limitar(valor: float) -> float:
    return min(1.0, max(0.0, valor))


def para_linha(rotulo: Rotulo, largura: int, altura: int) -> str:
    x1, y1, x2, y2 = rotulo.caixa
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cx = _limitar((x1 + x2) / 2 / largura)
    cy = _limitar((y1 + y2) / 2 / altura)
    w = _limitar((x2 - x1) / largura)
    h = _limitar((y2 - y1) / altura)
    tx, ty = _limitar(rotulo.traseira[0] / largura), _limitar(rotulo.traseira[1] / altura)
    fx, fy = _limitar(rotulo.frente[0] / largura), _limitar(rotulo.frente[1] / altura)
    return (
        f"{rotulo.classe} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
        f"{tx:.6f} {ty:.6f} 2 {fx:.6f} {fy:.6f} 2"
    )


def de_linha(linha: str, largura: int, altura: int) -> Rotulo:
    valores = linha.split()
    if len(valores) != 11:
        raise ValueError(f"Esperava 11 valores por linha, encontrei {len(valores)}: {linha!r}")
    classe = int(valores[0])
    cx, cy, w, h, tx, ty, _, fx, fy, _ = (float(v) for v in valores[1:])
    return Rotulo(
        caixa=(
            (cx - w / 2) * largura,
            (cy - h / 2) * altura,
            (cx + w / 2) * largura,
            (cy + h / 2) * altura,
        ),
        traseira=(tx * largura, ty * altura),
        frente=(fx * largura, fy * altura),
        classe=classe,
    )


def ler_arquivo(caminho: Path, largura: int, altura: int) -> list[Rotulo]:
    texto = caminho.read_text(encoding="utf-8")
    return [de_linha(linha, largura, altura) for linha in texto.splitlines() if linha.strip()]


def escrever_arquivo(caminho: Path, rotulos: list[Rotulo], largura: int, altura: int) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    linhas = [para_linha(r, largura, altura) for r in rotulos]
    caminho.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")


def yaml_dataset(raiz: Path) -> str:
    """Conteúdo do arquivo de configuração do dataset para o Ultralytics."""
    nomes = "\n".join(f"  {i}: {nome}" for i, nome in enumerate(CLASSES))
    return (
        f"path: {raiz.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [2, 3]  # 2 pontos (traseira, frente), cada um com x, y, visibilidade\n"
        "flip_idx: [0, 1]   # espelhar a imagem nao troca traseira com frente\n"
        f"names:\n{nomes}\n"
    )
