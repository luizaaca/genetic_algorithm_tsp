import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx


def calculate_weight(d):
    length = d.get("length")  # em metros
    maxspeed = d.get("maxspeed", 50)  # em km/h
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
        0.5, 50 / max(length, 1) * 0.5
    )  # ajuste conforme necessário
    adjusted_speed = maxspeed * reduction_factor
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
            weight.append(calculate_weight(sub_d))  # tempo = distancia / velocidade
    # se d é um dicionario simples, podemos efetuar o calculo diretamente
    else:
        weight.append(calculate_weight(d))  # tempo = distancia / velocidade
    return sum(weight) / len(weight)  # calcular o tempo médio


def path_weight_sum(graph, path, weight="length"):
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


def plot_route_with_eta(graph, route, weight_function):
    total_eta = 0
    total_length = 0
    num_segments = len(route)
    colors = plt.get_cmap("tab10", num_segments)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    import osmnx as ox

    ox.plot_graph(
        graph,
        bgcolor="white",
        node_size=10,
        edge_color="gray",
        node_color="red",
        edge_linewidth=0.4,
        ax=ax,
        show=False,
        close=False,
    )
    for i in range(num_segments):
        start = route[i]
        end = route[(i + 1) % num_segments]  # fecha o ciclo
        eta, segment = nx.bidirectional_dijkstra(
            graph, start, end, weight=weight_function
        )
        length = path_weight_sum(graph, segment)
        total_eta += eta
        total_length += length
        color = mcolors.to_hex(colors(i))
        ox.plot_graph_route(
            graph,
            segment,
            route_color=color,
            route_linewidth=3,
            ax=ax,
            orig_dest_node_color="none",
            show=False,
            close=False,
        )
        mid_idx = len(segment) // 2
        mid_node = segment[mid_idx]
        x = graph.nodes[mid_node]["x"]
        y = graph.nodes[mid_node]["y"]
        ax.text(
            x,
            y,
            f"{eta/60:.1f} min - {length:.1f} m",
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )
    legend_elements = [
        Line2D([0], [0], color=mcolors.to_hex(colors(i)), lw=3, label=f"Segmento {i+1}")
        for i in range(num_segments)
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.text(
        0.01,
        0.99,
        f"ETA total: {total_eta/60:.1f} min\nLength total: {total_length:.1f} m",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.show()
