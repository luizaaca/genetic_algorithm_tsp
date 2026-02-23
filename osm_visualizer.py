import osmnx as ox
import matplotlib.pyplot as plt

# Caminho para o seu arquivo baixado
caminho_arquivo = "./data/norte-manaus-small.osm.xml"

print("Carregando grafo... isso pode demorar dependendo da RAM.")
# ox.graph_from_xml carrega arquivos .osm ou .pbf
# simplify=True une pontos desnecessários em retas únicas
g = ox.graph_from_xml(caminho_arquivo, simplify=True)

print("Renderizando...")
ox.plot_graph(g, node_size=0, edge_color="blue", edge_linewidth=0.2)
plt.show()
