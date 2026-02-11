import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="EU Air Transportation Network")

# ======================================================
# CARGA DE DATOS
# ======================================================
@st.cache_data
def load_data():
    # Layers
    layers = pd.read_csv("data/EUAirTransportation_layers.txt", sep="\s+", header=0)
    layers.rename(columns={layers.columns[0]: "layerID", layers.columns[1]: "layerLabel"}, inplace=True)

    # Nodes
    nodes = pd.read_csv("data/EUAirTransportation_nodes.txt", sep="\s+", header=0)

    # Merge con airports.csv
    airports = pd.read_csv("data/airports.csv")
    nodes = nodes.merge(
        airports[["ident", "name", "type", "municipality"]],
        left_on="nodeLabel",
        right_on="ident",
        how="left"
    )
    nodes.rename(columns={"name": "airportName", "type": "airportType", "municipality": "city"}, inplace=True)

    # Filtrar solo aeropuertos grandes
    nodes = nodes[nodes["airportType"] == "large_airport"]
    node_ids = nodes["nodeID"].tolist()

    # Edges multiplex
    edges = pd.read_csv("data/EUAirTransportation_multiplex.edges", sep="\s+", header=0)
    edges.columns = ["X1.1", "X2", "layerID", "weight"]

    # Filtrar solo edges entre nodos grandes
    edges = edges[edges["X1.1"].isin(node_ids) & edges["X2"].isin(node_ids)]

    return nodes, edges, layers

nodes, edges, layers = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🔎 Configuración")

# Tipo de red
network_type = st.sidebar.radio("Tipo de red", ["Agregada", "Por layer"])

selected_layer = None
if network_type == "Por layer":
    selected_layer = st.sidebar.selectbox("Seleccionar layer", layers["layerLabel"])

# Selector de aeropuertos
selected_nodes = st.sidebar.multiselect(
    "Filtrar aeropuertos",
    nodes["nodeLabel"],
    default=nodes["nodeLabel"].tolist()
)

# Tipo de mapa
map_type = st.sidebar.selectbox(
    "Estilo de mapa",
    ["CartoDB Positron", "Esri WorldImagery", "OpenStreetMap", "Stamen Toner"]
)

# ======================================================
# FILTRADO
# ======================================================
edges_f = edges[edges["X1.1"].isin(nodes["nodeID"]) & edges["X2"].isin(nodes["nodeID"])]

if selected_layer is not None:
    layer_id = layers[layers["layerLabel"] == selected_layer]["layerID"].values[0]
    edges_f = edges_f[edges_f["layerID"] == layer_id]

nodes_f = nodes[nodes["nodeLabel"].isin(selected_nodes)]

# ======================================================
# GRAFO
# ======================================================
G = nx.from_pandas_edgelist(
    edges_f,
    source="X1.1",
    target="X2",
    edge_attr=["weight", "layerID"]
)

# ======================================================
# MAPA
# ======================================================
st.subheader("🌍 Mapa de Conexiones")

# Mapa base
tiles_map = {
    "CartoDB Positron": "CartoDB positron",
    "Esri WorldImagery": "Esri.WorldImagery",
    "OpenStreetMap": "OpenStreetMap",
    "Stamen Toner": "Stamen Toner"
}
m = folium.Map(location=[50, 10], zoom_start=4, tiles=tiles_map[map_type])

# Diccionario para coordenadas
node_pos = nodes_f.set_index("nodeID")[["nodeLat", "nodeLong"]].to_dict("index")

# Aristas
for _, row in edges_f.iterrows():
    src = node_pos.get(row["X1.1"])
    dst = node_pos.get(row["X2"])
    if src and dst:
        folium.PolyLine(
            locations=[[src["nodeLat"], src["nodeLong"]], [dst["nodeLat"], dst["nodeLong"]]],
            color="blue",
            weight=2,
            opacity=0.6
        ).add_to(m)

# Nodos
for idx, row in nodes_f.iterrows():
    folium.CircleMarker(
        location=[row["nodeLat"], row["nodeLong"]],
        radius=6,
        color="white",
        fillColor="#ff7800",
        fillOpacity=0.8,
        weight=2,
        popup=f"{row['nodeLabel']} - {row['airportName']} ({row['city']})"
    ).add_to(m)

st_folium(m, width=1200, height=600)

# ======================================================
# TABLAS
# ======================================================
tab1, tab2, tab3 = st.tabs(["Nodos", "Aristas", "Layers"])

with tab1:
    st.dataframe(nodes_f)

with tab2:
    st.dataframe(edges_f)

with tab3:
    st.dataframe(layers)
