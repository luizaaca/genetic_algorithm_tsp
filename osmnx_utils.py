from typing import List, Tuple, Union
from pyproj import Transformer
from shapely.geometry import MultiPoint
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import random
from route_plot_utils import RouteSegmentsInfo


def get_short_name_from_coord(lat_lon, dist=50):
    """
    Busca o nome do objeto mais próximo (rua, praça, prédio)
    usando a infraestrutura de cache do OSMnx.

    Args:
        lat_lon: Tupla (lat, lon) da coordenada.
        dist: Raio de busca em metros.

    Returns:
        Nome do objeto mais próximo ou string com coordenadas.
    """
    try:
        # Busca qualquer objeto 'highway' ou 'name' num raio de 50m
        # Isso usa o cache local do OSMnx automaticamente
        gdf = ox.features_from_point(
            lat_lon,
            tags={"highway": True, "building": True, "amenity": True},
            dist=dist,
        )

        if not gdf.empty and "name" in gdf.columns:
            # Pega o nome do objeto mais próximo que possui a coluna 'name'
            names = gdf["name"].dropna()
            if not names.empty:
                return names.iloc[0]  # Retorna o primeiro nome encontrado (curto)

        return f"Coords: {lat_lon:.4f}"
    except Exception:
        return "Local desconhecido"


def set_coords_and_names_for_locations(
    locations: List[Union[str, Tuple[float, float]]],
) -> List[Tuple[Tuple[float, float], str]]:
    """
    Para cada local na lista, obtém as coordenadas e o nome curto.

    Args:
        locations: Lista de strings (endereços) ou tuplas (lat, lon).

    Returns:
        Lista de tuplas ((lat, lon), nome).
    """
    result = []
    for loc in locations:
        if isinstance(loc, str):
            # Endereço -> Coordenada
            lat_lon = ox.geocoder.geocode(loc)
            name = loc
        else:
            # Coordenada -> Endereço (Reverse Geocoding)
            lat_lon = loc
            name = get_short_name_from_coord(loc)
        result.append((lat_lon, name))
    return result


def get_center_and_dist(
    coords: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], float]:
    """
    Calcula o centroide e o raio de abrangência de uma lista de locais.

    Args:
        coords: Lista de tuplas (lat, lon).

    Returns:
        Tuple contendo:
            - centroide_tuple: tupla (lat, lon) do centroide
            - distancia_em_metros: raio máximo em metros
    """
    # Criar um objeto MultiPoint (lon, lat para o shapely)
    points = MultiPoint([(lon, lat) for lat, lon in coords])

    # 1. Calcular o Centróide
    centroid = (points.centroid.y, points.centroid.x)

    # 2. Calcular a distância máxima do centro até o ponto mais longe
    # Usamos o Great Circle Distance do OSMnx para precisão em metros
    distances = [
        ox.distance.great_circle(centroid[0], centroid[1], lat, lon)
        for lat, lon in coords
    ]

    # Adicionamos uma margem de segurança (ex: 15% ou 200m)
    # para não cortar as ruas nas bordas dos pontos
    max_dist = max(distances) + 200

    return centroid, max_dist


def find_route_nodes(
    g_proj: nx.MultiDiGraph,
    locations: List[Tuple[Tuple[float, float], str]],
    verbose: bool = False,
) -> List[Tuple[str, int, Tuple[float, float]]]:
    """
    Encontra os nós do grafo mais próximos dos locais informados.

    Args:
        g_proj: Grafo OSMnx projetado (com CRS em metros).
        locations: Lista de tuplas ((lat, lon), nome).
        verbose: Se True, imprime informações detalhadas.

    Returns:
        Lista de tuplas (nome, node_id, (x, y)), onde (x, y) são as coordenadas projetadas.
    """
    # Transformer para converter lat/lon -> CRS do grafo
    transformer = Transformer.from_crs("EPSG:4326", g_proj.graph["crs"], always_xy=True)
    route_nodes = []

    for loc in locations:
        lat_lon, nome = loc
        lat, lon = lat_lon
        # lon = x, lat = y
        x, y = transformer.transform(lon, lat)
        node_id = ox.distance.nearest_nodes(g_proj, X=x, Y=y)
        route_nodes.append((nome, node_id, (x, y)))
        if verbose:
            print(f"Local: {nome} -> Node ID: {node_id} | (x, y): ({x:.1f}, {y:.1f})")
    return route_nodes


def set_node_priorities(
    g_proj: nx.MultiDiGraph,
    route_nodes: List[Tuple[str, int, Tuple[float, float]]],
    priorities: List[int],
):
    """
    Define prioridades para os nós do grafo com base na lista de route_nodes e prioridades.

    Args:
        g_proj: Grafo OSMnx projetado.
        route_nodes: Lista de tuplas (nome, node_id, (x, y)).
        priorities: Lista de inteiros representando as prioridades correspondentes aos nós.
    """
    for (nome, node_id, (x, y)), priority in zip(route_nodes, priorities):
        g_proj.nodes[node_id]["name"] = nome
        g_proj.nodes[node_id]["priority"] = priority


def create_projected_graph_from_point(
    center: Tuple[float, float], dist: float, network_type="drive", filter=None
) -> nx.MultiDiGraph:
    """
    Cria um grafo OSMnx projetado a partir de um ponto central e raio.

    Args:
        center: Tupla (lat, lon) do centro da área.
        dist: Distância em metros para definir o raio da área.
        network_type: Tipo de rede ("drive", "walk", etc.).
        filter: Filtro personalizado para OSMnx (ex: '["highway"~"primary|secondary"]').

    Returns:
        Grafo OSMnx projetado.
    """

    # Chama o método do OSMnx
    g = ox.graph_from_point(
        center_point=center,
        dist=dist,
        dist_type="bbox",  # "bbox" cria um quadrado, "network" segue a rede
        network_type=network_type if filter is None else None,
        custom_filter=filter,
        simplify=True,
    )

    g = ox.truncate.largest_component(g, strongly=True)
    g_proj = ox.project_graph(g)

    return g_proj


def initialize_graph(
    origin: str | Tuple[float, float],
    destinations: List[Tuple[Union[str, Tuple[float, float]], int]],
) -> Tuple[nx.MultiDiGraph, List[Tuple[str, int, Tuple[float, float]]]]:
    """
    Inicializa o grafo OSMnx projetado com os locais e prioridades.

    Args:
        origin: Endereço ou tupla (lat, lon) do ponto de origem.
        destinations: Lista de tuplas (endereço ou tupla (lat, lon), prioridade).

    Returns:
        Tuple contendo:
            - grafo OSMnx projetado
            - lista de nós contendo origem e destinos
    """
    ox.settings.use_cache = True
    ox.settings.useful_tags_way = [
        "highway",
        "maxspeed",
        "name",
        "length",
        "surface",
        "oneway",
    ]
    # Combina origem e destinos em uma única lista de locais
    all_locations = [origin] + [dest[0] for dest in destinations]
    all_locations_info = set_coords_and_names_for_locations(all_locations)
    center, radius = get_center_and_dist([info[0] for info in all_locations_info])
    # Cria o grafo a partir dos locais
    g_proj = create_projected_graph_from_point(center, radius)
    # Encontra os nós correspondentes aos locais
    route_nodes = find_route_nodes(g_proj, all_locations_info)
    set_node_priorities(g_proj, route_nodes[1:], [dest[1] for dest in destinations])
    return g_proj, route_nodes


def plot_route_segments_info(
    graph: nx.MultiDiGraph, route_segments_info: RouteSegmentsInfo
) -> None:
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
        graph,
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
