"""Treina o modelo YOLO de pontos-chave (caixa + traseira + meio + frente).

    python treinar.py                     # usa GPU se houver
    python treinar.py --epocas 150 --lote 32

"""

import argparse
import shutil
from pathlib import Path

from ferramentas.formato_yolo import yaml_dataset


def mostrar_metricas(metricas) -> None:
    """Resume o quanto o modelo acerta nas imagens de validação (que ele não viu no treino)."""
    caixa, pontos = metricas.box, metricas.pose
    print("\n=== Quanto o modelo está acertando (imagens de validação) ===")
    print(f"Precisão  (das caixas que ele marcou, quantas eram carro de verdade): {caixa.mp:.1%}")
    print(f"Recall    (dos carros que existiam, quantos ele encontrou):           {caixa.mr:.1%}")
    print(f"mAP50 caixa  (acerto geral da caixa, 100% = perfeito):                {caixa.map50:.1%}")
    print(f"mAP50 pontos (acerto de traseira/meio/frente, 100% = perfeito):       {pontos.map50:.1%}")
    print(f"mAP50-95 pontos (mesma ideia, mas exigindo mais precisão):            {pontos.map:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("dados/yolo"))
    parser.add_argument("--base", default="yolo11n-pose.pt",
                        help="Modelo pré-treinado de partida (n = menor e mais rápido; s = mais preciso).")
    parser.add_argument("--epocas", type=int, default=100)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--dispositivo", default=None, help="0 = primeira GPU, cpu = processador.")
    parser.add_argument("--saida", type=Path, default=Path("modelos/vigia-pose.pt"))
    args = parser.parse_args()

    if not (args.dataset / "images" / "train").exists():
        parser.error(f"Dataset não encontrado em {args.dataset}. Rode: python -m ferramentas.dividir_dataset")

    # Reescreve o yaml com o caminho absoluto desta máquina (o dataset pode ter vindo de outro PC).
    arquivo_yaml = args.dataset / "vigia.yaml"
    arquivo_yaml.write_text(yaml_dataset(args.dataset), encoding="utf-8")

    from ultralytics import YOLO

    modelo = YOLO(args.base)
    modelo.train(
        data=str(arquivo_yaml),
        epochs=args.epocas,
        imgsz=args.img,
        batch=args.lote,
        device=args.dispositivo,
        project="runs",
        name="vigia",
        exist_ok=True,
        patience=30,
        # Câmera de cima: o carro pode aparecer em qualquer rotação.
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
    )

    melhor = Path(modelo.trainer.best)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(melhor, args.saida)
    mostrar_metricas(YOLO(str(melhor)).val(data=str(arquivo_yaml), imgsz=args.img, device=args.dispositivo,
                                           project="runs", name="vigia_val", exist_ok=True))
    print(f"\nModelo salvo em {args.saida}")
    print("Teste com: python -m vision.road_monitor --video Video/VID_20230816_195820.mp4")


if __name__ == "__main__":
    main()
