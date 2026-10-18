"""Associa as detecções de um frame às do frame anterior para manter um ID por carro.

O modelo encontra os carros do zero em cada frame; este rastreador só serve para
saber que o carro 1 de agora é o mesmo carro 1 de antes, o que permite suavizar o
ângulo e calcular a direção de movimento.
"""

import math
from collections import deque
from dataclasses import dataclass, field

from .tracking import Point


@dataclass
class Trilha:
    id: int
    centros: deque = field(default_factory=lambda: deque(maxlen=15))
    angulos: deque = field(default_factory=lambda: deque(maxlen=5))
    frames_sem_ver: int = 0


class RastreadorSimples:
    def __init__(self, distancia_max: float, frames_tolerados: int = 10) -> None:
        self.distancia_max = distancia_max
        self.frames_tolerados = frames_tolerados
        self.trilhas: dict[int, Trilha] = {}
        self._proximo_id = 1

    def atualizar(self, centros: list[Point]) -> list[Trilha]:
        """Recebe os centros detectados no frame e devolve a trilha de cada um, na mesma ordem."""
        pares = sorted(
            (
                (math.dist(trilha.centros[-1], centro), trilha_id, indice)
                for trilha_id, trilha in self.trilhas.items()
                for indice, centro in enumerate(centros)
            ),
            key=lambda par: par[0],
        )
        atribuicao: dict[int, int] = {}
        trilhas_usadas: set[int] = set()
        for distancia, trilha_id, indice in pares:
            if distancia > self.distancia_max:
                break
            if indice in atribuicao or trilha_id in trilhas_usadas:
                continue
            atribuicao[indice] = trilha_id
            trilhas_usadas.add(trilha_id)

        resultado: list[Trilha] = []
        for indice, centro in enumerate(centros):
            trilha_id = atribuicao.get(indice)
            if trilha_id is None:
                trilha_id = self._proximo_id
                self._proximo_id += 1
                self.trilhas[trilha_id] = Trilha(id=trilha_id)
            trilha = self.trilhas[trilha_id]
            trilha.centros.append(centro)
            trilha.frames_sem_ver = 0
            resultado.append(trilha)

        vistas = {trilha.id for trilha in resultado}
        for trilha_id in list(self.trilhas):
            if trilha_id not in vistas:
                trilha = self.trilhas[trilha_id]
                trilha.frames_sem_ver += 1
                if trilha.frames_sem_ver > self.frames_tolerados:
                    del self.trilhas[trilha_id]
        return resultado
