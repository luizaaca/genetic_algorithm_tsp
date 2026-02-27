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


global_graph = None


def compute_route_segments_info(graph, route, weight_function):
    """
    Calcula os segmentos, ETA e comprimento total, retornando um objeto RouteSegmentsInfo.
    """
    global global_graph
    global_graph = graph
    segments = []
    total_eta = 0
    total_length = 0
    num_segments = len(route)
    for i in range(num_segments):
        start = route[i]
        end = route[(i + 1) % num_segments]
        eta, segment = nx.single_source_dijkstra(
            graph, start, end, weight=weight_function
        )
        length = path_weight_sum(graph, segment)
        total_eta += eta
        total_length += length
        segments.append(
            {
                "start": start,
                "end": end,
                "eta": eta,
                "length": length,
                "segment": segment,
            }
        )
    return RouteSegmentsInfo(
        segments=segments, total_eta=total_eta, total_length=total_length
    )


def plot_route_segments_info(graph, route_segments_info):
    """
    Plota os segmentos da rota a partir de um objeto RouteSegmentsInfo.
    """

    num_segments = len(route_segments_info.segments)
    palette = plt.get_cmap("turbo", num_segments)
    colors = [palette(i) for i in range(num_segments)]
    random.shuffle(colors)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ox.plot_graph(
        graph,
        bgcolor="white",
        node_size=10,
        edge_color="gray",
        node_color="red",
        edge_linewidth=0.4,
        ax=ax,
        show=False,
    )
    for i, seg in enumerate(route_segments_info.segments):
        color = mcolors.to_hex(colors[i])
        ox.plot_graph_route(
            graph,
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
        x_end = graph.nodes[end_node]["x"]
        y_end = graph.nodes[end_node]["y"]
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
                facecolor="white", edgecolor=color, boxstyle="circle,pad=0.3", alpha=0.9
            ),
            zorder=6,
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
    ax.legend(handles=legend_elements, loc="best", fontsize=10)
    plt.show()


def calculate_weight(u, v, d):
    length = d.get("length")  # em metros
    maxspeed = d.get("maxspeed", 50)  # em km/h
    # surface = d.get("surface", "unknown")  # tipo de superfície

    # Se for lista, pega o menor valor e converte para float
    if isinstance(maxspeed, list):
        maxspeed = min(
            [float(x) for x in maxspeed if str(x).replace(".", "", 1).isdigit()]
        )
    # Se for string, converte para float
    elif isinstance(maxspeed, str):
        if maxspeed.replace(".", "", 1).isdigit():
            maxspeed = float(maxspeed)
        else:
            maxspeed = 50  # valor padrão se não for número
    # Se não, tenta converter para float
    else:
        maxspeed = float(maxspeed)

    # Redução: quanto menor o length, maior a redução de velocidade
    # Exemplo: redução de até 50% para ruas curtas (<50m)
    reduction_factor = 1 - min(
        0.7, 50 / max(length, 1) * 0.9
    )  # ajuste conforme necessário
    adjusted_speed = maxspeed * reduction_factor

    global global_graph
    # buscas nodes na edge que sejam "highway == 'traffic_signals'" e reduz a velocidade em 30% para simular o impacto dos semáforos
    if (
        global_graph is not None
        and global_graph.nodes[v].get("highway") == "traffic_signals"
    ):
        adjusted_speed *= 0.8  # redução de 20% para semáforos

    return length / (
        max(adjusted_speed, 1) * 1000 / 3600
    )  # tempo = distancia / velocidade em segundos


def weight_function(u, v, d):
    """Calcula o peso de uma aresta usando length e maxspeed, com redução para ruas curtas."""
    weight = []
    # se d é um dicionario de dicionarios, precisamos efetuar o calculo para cada subdicionario
    if isinstance(d, dict):
        # se for um dicionario de dicionarios, precisamos iterar sobre os subdicionarios
        for key in d:
            sub_d = d[key]
            weight.append(
                calculate_weight(u, v, sub_d)
            )  # tempo = distancia / velocidade
    # se d é um dicionario simples, podemos efetuar o calculo diretamente
    else:
        weight.append(calculate_weight(u, v, d))  # tempo = distancia / velocidade
    return sum(weight) / len(weight)  # calcular o tempo médio


def path_weight_sum(graph, path, weight="length"):
    """Calcula a soma dos pesos (como length) ao longo de um caminho."""
    total = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if isinstance(edge_data, dict):
            # Caso haja múltiplas arestas, pega a primeira
            edge = list(edge_data.values())[0]
        else:
            edge = edge_data
        total += edge.get(weight, 0)
    return total
