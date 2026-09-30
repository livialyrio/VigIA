# Direção e desvio do carrinho

O programa funciona **em tempo real pela câmera**. As imagens em
`frames_dataset_video/` servem somente como referência visual para reconhecer o
carrinho e escolher onde marcar a traseira e a frente. O programa não abre nem
processa essa pasta durante a execução.

Não usamos modelo treinado nem pesos. O OpenCV acompanha os pontos marcados na
imagem ao vivo.

## Como funciona

1. O programa abre a câmera e mostra o vídeo.
2. Pressione `r` para selecionar os pontos no carrinho.
3. Clique primeiro no centro da traseira e depois no centro da frente; pressione
	`Enter` para confirmar.
4. O programa acompanha os detalhes visuais ao redor desses pontos nos frames
	seguintes.
5. A seta mostra para onde a frente está apontando. O ângulo e o desvio são
	comparados com a direção da pista.

Se o rastreamento perder algum ponto, pressione `r` e marque novamente a
traseira e a frente. O ângulo não é mostrado como válido enquanto os pontos não
estiverem acompanhados.

## Executar

Com a câmera conectada, abra o terminal na pasta do projeto e execute:

```powershell
pip install -r requirements.txt
python -m vision.road_monitor
```

O programa usa a câmera de índice `0` por padrão. Para escolher outra câmera:

```powershell
python -m vision.road_monitor --camera 1
```

## Direção de referência e tolerância

Por padrão, considera-se que a pista aponta para a direita da imagem, ou seja,
`0°`. Se a pista estiver inclinada no enquadramento, informe seu ângulo. O
limite padrão para considerar o carrinho alinhado é `5°`:

```powershell
python -m vision.road_monitor --pista-graus 0 --tolerancia 8
```

O valor positivo indica inclinação para baixo na imagem; o negativo, inclinação
para cima. O valor absoluto informa quantos graus a frente do carrinho se
desvia da direção configurada para a pista. Pressione `q` para fechar.

## Testar sem câmera

Os testes unitários verificam somente as contas dos ângulos. Execute:

```powershell
python -m unittest discover -s tests -v
```

Esses testes não abrem vídeo nem simulam a câmera. Para testar o acompanhamento
visual é necessário conectar uma câmera e marcar o carrinho no vídeo ao vivo.