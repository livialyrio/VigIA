"""Separa as imagens rotuladas em treino e validação no formato que o YOLO espera.

    python -m ferramentas.dividir_dataset
    
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from .formato_yolo import yaml_dataset

PADRAO_FRAME = re.compile(r"^(.*)_f\d{5}$")
PADRAO_SEQUENCIA = re.compile(r"^(.*_frame)_(\d+)$")
FRAMES_POR_BLOCO = 30


def grupo_da_imagem(nome: str) -> str:
    """Agrupa imagens quase iguais para que fiquem todas no treino ou todas na validação.

    - Frames de vídeo (`..._f00030`) são agrupados pelo vídeo.
    - Frames numerados em sequência (`..._frame_00123`) são agrupados em blocos de
      FRAMES_POR_BLOCO frames seguidos.
    - Fotos ficam sozinhas.
    """
    achado = PADRAO_FRAME.match(nome)
    if achado:
        return achado.group(1)
    achado = PADRAO_SEQUENCIA.match(nome)
    if achado:
        return f"{achado.group(1)}_bloco{int(achado.group(2)) // FRAMES_POR_BLOCO:03d}"
    return nome


def dividir(nomes: list[str], fracao_val: float, semente: int) -> tuple[list[str], list[str]]:
    grupos: dict[str, list[str]] = defaultdict(list)
    for nome in nomes:
        grupos[grupo_da_imagem(nome)].append(nome)
    chaves = sorted(grupos)
    random.Random(semente).shuffle(chaves)
    alvo = fracao_val * len(nomes)
    grupos_val: list[str] = []
    total_val = 0
    for chave in chaves:
        # Só entra na validação se isso deixar o total mais perto do alvo.
        if total_val + len(grupos[chave]) / 2 <= alvo:
            grupos_val.append(chave)
            total_val += len(grupos[chave])
    if not grupos_val and fracao_val > 0:
        grupos_val.append(min(chaves, key=lambda c: len(grupos[c])))
    val = [nome for chave in grupos_val for nome in grupos[chave]]
    treino = [nome for chave in chaves if chave not in grupos_val for nome in grupos[chave]]
    return sorted(treino), sorted(val)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagens", type=Path, default=Path("dados/imagens"))
    parser.add_argument("--rotulos", type=Path, default=Path("dados/rotulos"))
    parser.add_argument("--saida", type=Path, default=Path("dados/yolo"))
    parser.add_argument("--val", type=float, default=0.2, help="Fração para validação.")
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args(argv)

    rotuladas = sorted(p.stem for p in args.rotulos.glob("*.txt") if (args.imagens / f"{p.stem}.jpg").exists())
    if len(rotuladas) < 10:
        parser.error(f"Só {len(rotuladas)} imagens rotuladas. Rotule mais antes de treinar.")

    treino, val = dividir(rotuladas, args.val, args.semente)
    if args.saida.exists():
        shutil.rmtree(args.saida)
    for parte, nomes in (("train", treino), ("val", val)):
        (args.saida / "images" / parte).mkdir(parents=True)
        (args.saida / "labels" / parte).mkdir(parents=True)
        for nome in nomes:
            shutil.copy2(args.imagens / f"{nome}.jpg", args.saida / "images" / parte / f"{nome}.jpg")
            shutil.copy2(args.rotulos / f"{nome}.txt", args.saida / "labels" / parte / f"{nome}.txt")

    (args.saida / "vigia.yaml").write_text(yaml_dataset(args.saida), encoding="utf-8")
    vazias = sum(1 for n in rotuladas if not (args.rotulos / f"{n}.txt").read_text().strip())
    print(f"Treino: {len(treino)} imagens | Validação: {len(val)} imagens | sem carro: {vazias}")
    print(f"Dataset pronto em {args.saida}")


if __name__ == "__main__":
    main()
