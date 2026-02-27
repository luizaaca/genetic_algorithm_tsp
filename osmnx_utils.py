from typing import List, Tuple, Union
from pyproj import Transformer
from shapely.geometry import MultiPoint
import osmnx as ox
import networkx as nx


def get_short_name_from_coord(lat_lon, dist=50):
    """
    Busca o nome do objeto mais próximo (rua, praça, prédio)
    usando a infraestrutura de cache do OSMnx.
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


def get_center_and_dist(
    locations: List[Union[str, Tuple[float, float]]],
) -> Tuple[Tuple[float, float], float, List[Tuple[Tuple[float, float], str]]]:
    """
    locations: lista de strings (endereços) ou tuplas (lat, lon)

    Retorna: (centroide_tuple, distancia_em_metros, lista_de_tuplas_com_coordenadas_e_nomes)
    """
    coords = []
    names = []

    for loc in locations:
        if isinstance(loc, str):
            # Endereço -> Coordenada
            lat_lon = ox.geocoder.geocode(loc)
            coords.append(lat_lon)
            names.append(loc)
        else:
            # Coordenada -> Endereço (Reverse Geocoding)
            coords.append(loc)
            # Retorna uma string com o endereço formatado
            address_name = get_short_name_from_coord(loc)
            names.append(address_name)

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

    return centroid, max_dist, list(zip(coords, names))


def find_route_nodes(
    g_proj: nx.MultiDiGraph,
    locations: List[Tuple[Tuple[float, float], str]],
    verbose: bool = False,
) -> List[Tuple[str, int, Tuple[float, float]]]:
    """
    Recebe:
        g_proj: grafo OSMnx projetado (com CRS em metros)
        locations: lista de coordenadas e nomes [(lat, lon), nome]
        verbose: se True, imprime informações
    Retorna:
        Lista de tuplas (nome, node_id, (x, y)) onde (x, y) são as coordenadas em metros no grafo
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
    route_nodes: lista de tuplas (nome, node_id, (x, y))
    priorities: lista de inteiros representando as prioridades correspondentes aos nós
    """
    for (nome, node_id, (x, y)), priority in zip(route_nodes, priorities):
        g_proj.nodes[node_id]["name"] = nome
        g_proj.nodes[node_id]["priority"] = priority


def create_projected_graph_from_point(
    center: Tuple[float, float], dist: float, network_type="drive", filter=None
) -> nx.MultiDiGraph:
    """
    center: tupla (lat, lon) do centro da área

    dist: distância em metros para definir o raio da área

    network_type: tipo de rede ("drive", "walk", etc.)

    filter: filtro personalizado para OSMnx (ex: '["highway"~"primary|secondary"]')

    Retorna: grafo OSMnx projetado
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
    g_proj: nx.MultiDiGraph,
    origin: str | Tuple[float, float],
    destinations: List[Tuple[Union[str, Tuple[float, float]], int]],
) -> Tuple[nx.MultiDiGraph, List[Tuple[str, int, Tuple[float, float]]]]:
    """
    Inicializa o grafo OSMnx projetado com as locais e prioridades.
    origin: endereço ou tupla (lat, lon) do ponto de origem
    destinations: lista de tuplas (endereço ou tupla (lat, lon), prioridade)
    Retorna: grafo OSMnx projetado e a lista de nós contendo origem e destinos
    """
    # Combina origem e destinos em uma única lista de locais
    all_locations = [origin] + [dest[0] for dest in destinations]
    center, radius, all_locations_info = get_center_and_dist(all_locations)
    # Cria o grafo a partir dos locais
    g_proj = create_projected_graph_from_point(center, radius)
    # Encontra os nós correspondentes aos locais
    route_nodes = find_route_nodes(g_proj, all_locations_info)
    set_node_priorities(g_proj, route_nodes[1:], [dest[1] for dest in destinations])
    return g_proj, route_nodes
