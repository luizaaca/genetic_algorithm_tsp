from dataclasses import dataclass, field
from typing import List, Any
import networkx as nx


# Classe para armazenar informações dos segmentos e totais
@dataclass
class RouteSegmentsInfo:
    segments: List[dict] = field(default_factory=list)
    total_eta: float = 0.0
    total_length: float = 0.0
    total_cost: float | None = None


class RouteCalculator:
    """
    Classe para análise e plotagem de rotas em grafos OSMnx.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Inicializa o RouteCalculator com o grafo projetado.

        Args:
            graph: Grafo OSMnx projetado.
        """
        self.graph = graph

    def compute_route_segments_info(
        self,
        route: List[tuple],
        weight_function: Any = "length",
        cost_type: str | None = None,
    ) -> RouteSegmentsInfo:
        """
        Calcula os segmentos, ETA e comprimento total, retornando um objeto RouteSegmentsInfo.

        Args:
            route: Lista de tuplas (nome, node_id, (x, y)) representando a rota.
            weight_function: Função para calcular o peso (tempo) de cada aresta.
            cost_type: Tipo opcional para calcular o custo a partir do ETA (ex: custo de combustível).

        Returns:
            RouteSegmentsInfo: Objeto contendo informações dos segmentos, ETA e comprimento total.
        """
        segments = []
        total_eta = 0
        total_length = 0
        total_cost = 0 if cost_type else None
        num_segments = len(route)
        for i in range(num_segments):
            start = route[i][1]
            end = route[(i + 1) % num_segments][1]
            eta, segment = nx.single_source_dijkstra(
                self.graph, start, end, weight=weight_function
            )
            length = self._path_length_sum(segment)
            total_eta += eta
            total_length += length
            cost = None
            if cost_type:
                cost_function = self._get_cost_function(cost_type)
                cost = cost_function(end, eta)
                total_cost += cost
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "eta": eta,
                    "length": length,
                    "segment": segment,
                    "cost": cost,
                    "name": route[(i + 1) % num_segments][0],
                    "coords": route[(i + 1) % num_segments][2],
                }
            )
        return RouteSegmentsInfo(
            segments=segments,
            total_eta=total_eta,
            total_length=total_length,
            total_cost=total_cost,
        )

    def get_weight_function(self):
        """
        Retorna a função de peso (ETA) para o grafo.

        Returns:
            Função que recebe (u, v, data) e retorna o peso (ETA) da aresta.
        """

        def calculate_weight(u: Any, v: Any, d: dict) -> float:
            length = d.get("length")  # em metros
            maxspeed = d.get("maxspeed", 50)  # em km/h

            if isinstance(maxspeed, list):
                maxspeed = min(
                    [float(x) for x in maxspeed if str(x).replace(".", "", 1).isdigit()]
                )
            elif isinstance(maxspeed, str):
                if maxspeed.replace(".", "", 1).isdigit():
                    maxspeed = float(maxspeed)
                else:
                    maxspeed = 50
            else:
                maxspeed = float(maxspeed)

            reduction_factor = 1 - min(0.7, 50 / max(length, 1) * 0.9)
            adjusted_speed = maxspeed * reduction_factor

            node = self.graph.nodes[v]
            if node.get("highway") == "traffic_signals":
                adjusted_speed *= 0.8

            return length / (max(adjusted_speed, 1) * 1000 / 3600)

        def weight_function(u: Any, v: Any, d: Any) -> float:
            weight = []
            if isinstance(d, dict):
                for key in d:
                    sub_d = d[key]
                    weight.append(calculate_weight(u, v, sub_d))
            else:
                weight.append(calculate_weight(u, v, d))
            return sum(weight) / len(weight)

        return weight_function

    def _path_length_sum(self, path: List[Any], weight: str = "length") -> float:
        """
        Calcula a soma dos pesos (como length) ao longo de um caminho.

        Args:
            path: Lista de nós representando o caminho.
            weight: Nome do atributo a ser somado (default: "length").

        Returns:
            float: Soma dos pesos ao longo do caminho.
        """
        total = 0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                edge = list(edge_data.values())[0]
            else:
                edge = edge_data
            total += edge.get(weight, 0)
        return total

    def _get_cost_function(self, cost_type: str) -> Any:
        """
        Retorna uma função de custo baseada no tipo especificado.

        Args:
            node_id: ID do nó para o qual calcular o custo.
            cost_type: Tipo de custo (ex: "priority").

        Returns:
            Função que recebe o ETA e retorna o custo correspondente.
        """

        def priority(node_id, eta: float) -> float:
            node = self.graph.nodes[node_id]
            priority_value = node.get("priority", 1)
            # Ajuste matemático: cada prioridade aumenta o ETA em 20% sobre o valor base
            # Exemplo: prioridade 1 = 100%, 2 = 120%, 3 = 140%, ...
            percent = 1 + 0.2 * (priority_value - 1)
            return eta * percent

        if cost_type == "priority":
            return priority
        else:
            raise ValueError(f"Tipo de custo desconhecido: {cost_type}")
