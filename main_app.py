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
    # Ya tienen nodeLong y nodeLat
    # columns: nodeID, nodeLabel, nodeLong, nodeLat

    # Edges multiplex
    edges = pd.read_csv("data/EUAirTransportation_multiplex.edges", sep="\s+", header=0)
    edges.columns = ["X1.1", "X2", "layerID", "weight"]

    return nodes, edges, layers

nodes, edges, layers = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🔎 Configuración")

network_type = st.sidebar.radio(
    "Tipo de red",
    ["Agregada", "Por layer"]
)

selected_layer = None
if network_type == "Por layer":
    selected_layer = st.sidebar.selectbox(
        "Seleccionar layer",
        layers["layerLabel"]
    )

selected_nodes = st.sidebar.multiselect(
    "Filtrar aeropuertos",
    nodes["nodeLabel"],
    default=nodes["nodeLabel"].tolist()
)

# ======================================================
# FILTRADO
# ======================================================
edges_f = edges[edges["X1.1"].isin(nodes["nodeID"]) & edges["X2"].isin(nodes["nodeID"])]

if selected_layer is not None:
    layer_id = layers[layers["layerLabel"] == selected_layer]["layerID"].values[0]
    edges_f = edges_f[edges_f["layerID"] == layer_id]

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
# METRICAS
# ======================================================
st.title("✈️ EU Air Transportation Network")

c1, c2, c3 = st.columns(3)
c1.metric("Nodos", G.number_of_nodes())
c2.metric("Aristas", G.number_of_edges())
c3.metric("Layers activas", edges_f["layerID"].nunique())

# ======================================================
# MAPA
# ======================================================
st.subheader("🌍 Mapa de Conexiones")

m = folium.Map(location=[50, 10], zoom_start=4, tiles="CartoDB positron")

node_pos = nodes.set_index("nodeID")[["nodeLat", "nodeLong"]].to_dict("index")

for _, row in edges_f.iterrows():
    src = node_pos[row["X1.1"]]
    dst = node_pos[row["X2"]]
    folium.PolyLine(
        locations=[[src["nodeLat"], src["nodeLong"]], [dst["nodeLat"], dst["nodeLong"]]],
        color="blue",
        weight=1,
        opacity=0.4
    ).add_to(m)

st_folium(m, width=1200, height=600)

# ======================================================
# TABLAS
# ======================================================
tab1, tab2, tab3 = st.tabs(["Nodos", "Aristas", "Layers"])

with tab1:
    st.dataframe(nodes)

with tab2:
    st.dataframe(edges_f)

with tab3:
    st.dataframe(layers)
