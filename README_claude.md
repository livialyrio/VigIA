# VigIA — monitoramento de rodovia com visão computacional

O VigIA encontra os carros (robôs) **sozinho, só pela imagem**, em cada frame da
câmera. Para cada carro ele mede:

- **posição** (centro da caixa);
- **orientação**: ângulo do vetor traseira → frente em relação ao eixo x da imagem;
- **direção de movimento**: para onde o carro de fato andou nos últimos frames;
- **desvio em relação à pista** e status `ALINHADO`, `DESVIADO` ou `CONTRAMAO`.

Para isso usamos um modelo **YOLO de pontos-chave** (`yolo11n-pose`), treinado
pelo grupo para marcar em cada carro uma caixa e três pontos: **traseira**,
**meio** e **frente**. A orientação usa traseira → frente; se um desses pontos
falhar (carro cortado na imagem, por exemplo), o meio entra no lugar. Não há símbolo nem marcador no carro.

```
câmera → YOLO (caixa + traseira + meio + frente) → rastreador (ID por carro) → ângulos → tela / JSON
```

## Estrutura

| Pasta / arquivo | O que é |
|---|---|
| `vision/road_monitor.py` | Monitor ao vivo (câmera ou vídeo) |
| `vision/detector.py` | Carrega o modelo treinado e devolve as detecções |
| `vision/rastreador.py` | Mantém o mesmo ID para o mesmo carro entre frames |
| `vision/tracking.py` | Contas de ângulo (orientação, desvio, movimento) |
| `ferramentas/preparar_imagens.py` | Junta fotos e frames de vídeo para rotular |
| `ferramentas/rotular.py` | Ferramenta de rotulagem (caixa + traseira + meio + frente) |
| `ferramentas/dividir_dataset.py` | Separa treino/validação no formato do YOLO |
| `treinar.py` | Treino local (se tiver GPU) |
| `treino_colab.ipynb` | Treino no Google Colab (GPU grátis) — recomendado |
| `modelos/vigia-pose.pt` | Modelo treinado (gerado pelo treino) |
| `dados/` | Imagens e rótulos (fora do git, são pesados) |

## Instalação 

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Passo a passo para treinar o modelo

### 1. Preparar as imagens

Usei as fotos e vídeos do repositório de referência (`chon-lab/chonbot-computer-vision`,
pasta `dataset/`).

```powershell
# exemplo: o repositório de referência clonado ao lado desta pasta
python -m ferramentas.preparar_imagens ..\chonbot-computer-vision\dataset --a-cada 15
python -m ferramentas.preparar_imagens frames_dataset_video
python -m ferramentas.preparar_imagens frames_dataset_novo      # frames novos (3 pontos)
python -m ferramentas.preparar_imagens gravacoes --a-cada 10   # para os futuros videos novos
```

Tudo vai para `dados/imagens/`, redimensionado (maior lado 1280 px). `--a-cada 15`
pega 1 frame a cada 15 dos vídeos, para não salvar frames quase iguais.

### 2. Rotular

```powershell
python -m ferramentas.rotular
python -m ferramentas.rotular --filtro Decocco    # só as imagens de um carro
```

Para cada carro: **arraste uma caixa** em volta dele, **clique na traseira**, **clique
no meio** (centro do teto) e **clique na frente**. `Enter` salva e passa. `n` marca a imagem como "sem carro" (também
ajuda o modelo a não ver carro onde não tem). `z` desfaz. `a`/`d` navegam. `q` sai.


```powershell
python -m ferramentas.rotular --modelo modelos/vigia-pose.pt
```

  Nas imagens ainda não rotuladas, o modelo já desenha a sugestão em amarelo:
  `Enter` aceita, ou `z` remove e você marca de novo.

### 3. Montar o dataset

```powershell
python -m ferramentas.dividir_dataset
```

Cria `dados/yolo/` com treino (80%) e validação (20%). Frames do mesmo vídeo ficam
todos do mesmo lado, para a validação ser honesta.

### 4. Treinar no Google Colab

```powershell
Compress-Archive -Path dados\yolo\* -DestinationPath vigia_dataset.zip -Force
```
Abra `treino_colab.ipynb` no Colab e rode as células: o notebook treina, mostra o
quanto o modelo acerta na validação (precisão, recall, mAP) e baixa o `vigia-pose.pt`.
Também dá para treinar localmente com `python treinar.py`, que mostra as mesmas métricas.

## Rodar o monitor

```powershell
python -m vision.road_monitor                        # câmera 0
python -m vision.road_monitor --camera 1
python -m vision.road_monitor --video teste.mp4      # vídeo gravado
```

No fim, o monitor mostra em quantos frames o carro foi encontrado e a confiança média
da detecção e de cada ponto.

1. Aperte **`p`** e clique **dois pontos sobre a pista**, no sentido do trânsito
   (início → fim). A pista fica salva em `pista.json` e é carregada nas próximas vezes.
2. Cada carro aparece com caixa, os pontos traseira (azul) → meio (laranja) → frente (vermelho), seta roxa
   (movimento), ID, ângulo, desvio e status.
3. `q` sai.

Opções úteis:

| Opção | Para quê |
|---|---|
| `--tolerancia 8` | Desvio máximo em graus para ALINHADO (padrão 5) |
| `--confianca 0.4` | Aceita detecções menos certas (padrão 0.5) |
| `--saida-json estados.jsonl` | Grava o estado de cada frame (é o que os agentes JaCaMo vão perceber) |
| `--saida-video anotado.mp4 --sem-janela` | Processa um vídeo sem abrir janela e salva o resultado |

Exemplo de linha do `--saida-json`:

```json
{"frame": 120, "pista_graus": 0.0, "carros": [{"id": 1, "centro": [512.3, 288.0],
 "confianca": 0.91, "confianca_pontos": [0.95, 0.97, 0.93], "orientacao_graus": 8.4, "movimento_graus": 6.9, "parado": false,
 "de_re": false, "desvio_graus": 8.4, "status": "DESVIADO"}]}
```

## Convenção dos ângulos

0° = para a direita da imagem, +90° = para baixo, −90° = para cima, ±180° = para a
esquerda (o eixo y da imagem cresce para baixo). Desvio positivo = carro girado no
sentido horário em relação à pista.

## Testes

```powershell
python -m unittest discover -s tests -v
```

Os testes cobrem as contas de ângulo, o rastreador, o formato dos rótulos, a
ferramenta de rotulagem, a divisão do dataset e o monitor de ponta a ponta (com um
detector falso no lugar do modelo, num vídeo sintético). Eles não precisam de câmera
nem do modelo treinado.

## Próximos passos

- Deixar a pista como área invés de reta.
- treinar de novo o modelo com as imagens/vídeos no ângulo que vamos usar.
- tentar resolver a identificação quando a imagem do carro estivcer cortada
- Ponte Python → JaCaMo: enviar o estado de cada frame para um artefato CArtAgO.
- Agentes Jason que reagem a `DESVIADO` / `CONTRAMAO`.
- Calibrar a câmera (homografia) para medir ângulos no plano real da pista.
