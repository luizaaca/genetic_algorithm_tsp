from dataclasses import dataclass, field
from typing import List, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import osmnx as ox
import networkx as nx
import random


# Classe para armazenar informações dos segmentos e totais
@dataclass
class RouteSegmentsInfo:
    segments: List[dict] = field(default_factory=list)
    total_eta: float = 0.0
    total_length: float = 0.0
    total_cost: float | None = None


class RoutePlotter:
    """
    Classe para análise e plotagem de rotas em grafos OSMnx.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Inicializa o RoutePlotter com o grafo projetado.

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

    def plot_route_segments_info(self, route_segments_info: RouteSegmentsInfo) -> None:
        """
        Plota os segmentos da rota a partir de um objeto RouteSegmentsInfo.

        Args:
            route_segments_info: Objeto RouteSegmentsInfo com informações dos segmentos.

        Returns:
            None
        """
        num_segments = len(route_segments_info.segments)
        palette = plt.get_cmap("turbo", num_segments)
        colors = [palette(i) for i in range(num_segments)]
        random.shuffle(colors)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        ox.plot_graph(
            self.graph,
            bgcolor="white",
            node_size=1,
            edge_color="gray",
            node_color="red",
            edge_linewidth=0.4,
            ax=ax,
            show=False,
        )

        for i, seg in enumerate(route_segments_info.segments):
            color = mcolors.to_hex(colors[i])
            ox.plot_graph_route(
                self.graph,
                seg["segment"],
                route_color=color,
                route_linewidth=3,
                ax=ax,
                orig_dest_node_color="none",
                show=False,
                close=False,
            )
            # Plotar apenas ponto de chegada: número do segmento
            end_node = seg["end"]
            x_end = self.graph.nodes[end_node]["x"]
            y_end = self.graph.nodes[end_node]["y"]
            ax.text(
                x_end,
                y_end,
                str(i + 1),
                color=color,
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor=color,
                    boxstyle="circle,pad=0.3",
                    alpha=0.9,
                ),
                zorder=6,
            )
            # Plotar pontos de interesse (nome do local) próximo ao ponto de chegada
            ax.scatter(
                seg["coords"][0],
                seg["coords"][1],
                c="red",
                s=100,
                marker="X",
                label="Pontos de Interesse",
                zorder=5,
            )
            ax.annotate(
                seg["name"],
                (seg["coords"][0], seg["coords"][1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )
        legend_elements = []
        for i, seg in enumerate(route_segments_info.segments):
            color = mcolors.to_hex(colors[i])
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=3,
                    label=f"Seg {i+1}: {seg['length']:.1f} m, {seg['eta']/60:.1f} min",
                )
            )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                lw=0.4,
                label=f"Total: {route_segments_info.total_length:.1f} m, {route_segments_info.total_eta/60:.1f} min",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                lw=0.4,
                label=f"Total cost: {route_segments_info.total_cost:.1f}",
            )
        )
        ax.legend(handles=legend_elements, loc="best", fontsize=10)
        plt.show()

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
