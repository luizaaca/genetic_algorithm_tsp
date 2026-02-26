import osmnx as ox
import matplotlib.pyplot as plt

# Caminho para o seu arquivo baixado
caminho_arquivo = "./data/centro_sp_nano.osm.xml"

print("Carregando grafo... isso pode demorar dependendo da RAM.")
# ox.graph_from_xml carrega arquivos .osm ou .pbf
# simplify=True une pontos desnecessários em retas únicas
g = ox.graph_from_xml(caminho_arquivo, simplify=True)

print("Renderizando...")
# aumentar tamanho do plot, mudar fundo para branco e aumentar a resolução
fig, ax = ox.plot_graph(
    g,
    node_size=0,
    edge_color="blue",
    edge_linewidth=0.2,
    bgcolor="white",
    figsize=(12, 12),
    dpi=300,
)
plt.show()
