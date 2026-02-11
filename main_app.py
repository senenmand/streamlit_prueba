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
    layers = pd.read_csv("data/EUAirTransportation_layers.txt", sep="\s+", header=0)
    layers.rename(columns={layers.columns[0]: "layerID", layers.columns[1]: "layerLabel"}, inplace=True)

    nodes = pd.read_csv("data/EUAirTransportation_nodes.txt", sep="\s+", header=0)

    airports = pd.read_csv("data/airports.csv")
    nodes = nodes.merge(
        airports[["ident", "name", "type", "municipality"]],
        left_on="nodeLabel",
        right_on="ident",
        how="left"
    )
    nodes.rename(columns={"name": "airportName", "type": "airportType", "municipality": "city"}, inplace=True)

    nodes = nodes[nodes["airportType"] == "large_airport"]
    node_ids = nodes["nodeID"].tolist()

    edges = pd.read_csv("data/EUAirTransportation_multiplex.edges", sep="\s+", header=0)
    edges.columns = ["X1.1", "X2", "layerID", "weight"]
    edges = edges[edges["X1.1"].isin(node_ids) & edges["X2"].isin(node_ids)]

    return nodes, edges, layers

nodes, edges, layers = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🔎 Configuración")

network_type = st.sidebar.radio("Tipo de red", ["Agregada", "Por layer"])
selected_layer = None
if network_type == "Por layer":
    selected_layer = st.sidebar.selectbox("Seleccionar layer", layers["layerLabel"])

selected_airport = st.sidebar.selectbox(
    "Seleccionar aeropuerto",
    nodes["nodeLabel"],
    format_func=lambda x: f"{x} - {nodes.loc[nodes['nodeLabel']==x,'airportName'].values[0]}"
)

map_type = st.sidebar.selectbox(
    "Estilo de mapa",
    ["CartoDB Positron", "Esri WorldImagery", "OpenStreetMap", "Stamen Toner"]
)

# ======================================================
# FILTRADO POR AEROPUERTO Y LAYER
# ======================================================
edges_f = edges.copy()

# Filtrar por layer si aplica
if network_type == "Por layer" and selected_layer:
    layer_id = layers[layers["layerLabel"] == selected_layer]["layerID"].values[0]
    edges_f = edges_f[edges_f["layerID"] == layer_id]

# Filtrar por aeropuerto seleccionado
selected_id = nodes[nodes["nodeLabel"] == selected_airport]["nodeID"].values[0]
edges_f = edges_f[(edges_f["X1.1"] == selected_id) | (edges_f["X2"] == selected_id)]

# Filtrar nodos que aparecen en las aristas
nodes_f = nodes[nodes["nodeID"].isin(edges_f["X1.1"].tolist() + edges_f["X2"].tolist())]

# Diccionario de coordenadas
node_pos = nodes_f.set_index("nodeID")[["nodeLat", "nodeLong"]].to_dict("index")

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
st.subheader("🌍 Mapa de Conexiones Filtradas")

tiles_map = {
    "CartoDB Positron": "CartoDB positron",
    "Esri WorldImagery": "Esri.WorldImagery",
    "OpenStreetMap": "OpenStreetMap",
    "Stamen Toner": "Stamen Toner"
}
m = folium.Map(location=[50, 10], zoom_start=4, tiles=tiles_map[map_type])

# Dibujar solo las aristas filtradas
for _, row in edges_f.iterrows():
    src = node_pos[row["X1.1"]]
    dst = node_pos[row["X2"]]
    folium.PolyLine(
        locations=[[src["nodeLat"], src["nodeLong"]], [dst["nodeLat"], dst["nodeLong"]]],
        color="red",
        weight=4,
        opacity=0.8
    ).add_to(m)

# Dibujar nodos
for idx, row in nodes_f.iterrows():
    color = "#ff0000" if row["nodeID"] == selected_id else "#ff7800"
    folium.CircleMarker(
        location=[row["nodeLat"], row["nodeLong"]],
        radius=10 if row["nodeID"] == selected_id else 6,
        color="white",
        fillColor=color,
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
