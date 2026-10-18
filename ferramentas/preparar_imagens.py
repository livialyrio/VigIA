"""Junta fotos e frames de vídeo em uma única pasta, já redimensionados, para rotular.

Exemplos:
    # todas as fotos e vídeos do repositório de referência
    python -m ferramentas.preparar_imagens ..\\chonbot-computer-vision\\dataset

    # só o Decocco, 1 frame a cada 10 dos vídeos
    python -m ferramentas.preparar_imagens ..\\chonbot-computer-vision\\dataset\\Decocco --a-cada 10

    # vídeos novos gravados com a câmera fixa da pista
    python -m ferramentas.preparar_imagens gravacoes\\
"""

import argparse
from pathlib import Path

import cv2

EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def redimensionar(imagem, lado_max: int):
    altura, largura = imagem.shape[:2]
    escala = lado_max / max(altura, largura)
    if escala >= 1.0:
        return imagem
    return cv2.resize(imagem, (round(largura * escala), round(altura * escala)), interpolation=cv2.INTER_AREA)


def nitidez(imagem) -> float:
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(cinza, cv2.CV_64F).var()


def nome_seguro(texto: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in texto)


def listar_arquivos(origens: list[Path]) -> list[Path]:
    arquivos = []
    for origem in origens:
        if origem.is_dir():
            arquivos.extend(p for p in sorted(origem.rglob("*")) if p.suffix.lower() in EXT_IMAGEM | EXT_VIDEO)
        elif origem.suffix.lower() in EXT_IMAGEM | EXT_VIDEO:
            arquivos.append(origem)
    return arquivos


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("origens", nargs="+", type=Path, help="Pastas ou arquivos de foto/vídeo.")
    parser.add_argument("--saida", type=Path, default=Path("dados/imagens"))
    parser.add_argument("--lado-max", type=int, default=1280, help="Maior lado da imagem salva, em pixels.")
    parser.add_argument("--a-cada", type=int, default=15, help="Nos vídeos, salva 1 frame a cada N.")
    parser.add_argument("--nitidez-min", type=float, default=0.0,
                        help="Descarta frames de vídeo mais borrados que isso (ex.: 30). 0 desliga.")
    args = parser.parse_args(argv)

    args.saida.mkdir(parents=True, exist_ok=True)
    salvas = descartadas = 0
    for arquivo in listar_arquivos(args.origens):
        prefixo = nome_seguro(f"{arquivo.parent.name}_{arquivo.stem}")
        if arquivo.suffix.lower() in EXT_IMAGEM:
            destino = args.saida / f"{prefixo}.jpg"
            if destino.exists():
                continue
            imagem = cv2.imread(str(arquivo))
            if imagem is None:
                print(f"  ! não consegui abrir {arquivo}")
                continue
            cv2.imwrite(str(destino), redimensionar(imagem, args.lado_max), [cv2.IMWRITE_JPEG_QUALITY, 92])
            salvas += 1
            continue

        captura = cv2.VideoCapture(str(arquivo))
        indice = 0
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            if indice % args.a_cada == 0:
                destino = args.saida / f"{prefixo}_f{indice:05d}.jpg"
                if not destino.exists():
                    if args.nitidez_min and nitidez(frame) < args.nitidez_min:
                        descartadas += 1
                    else:
                        cv2.imwrite(str(destino), redimensionar(frame, args.lado_max), [cv2.IMWRITE_JPEG_QUALITY, 92])
                        salvas += 1
            indice += 1
        captura.release()
        print(f"  {arquivo.name}: {indice} frames lidos")

    print(f"{salvas} imagens salvas em {args.saida}" + (f" ({descartadas} borradas descartadas)" if descartadas else ""))


if __name__ == "__main__":
    main()
