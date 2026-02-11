import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Red de Aeropuertos Europeos",
    layout="wide"
)

st.title("✈️ Dashboard Interactivo de Red de Aeropuertos Europeos")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    layers = pd.read_csv("data/EUAirTransportation_layers.txt", sep=r"\s+")
    nodes = pd.read_csv("data/EUAirTransportation_nodes.txt", sep=r"\s+")
    edges = pd.read_csv("data/EUAirTransportation_multiplex.edges", sep=r"\s+")
    airports = pd.read_csv("data/airports.csv")

    nodes = nodes.merge(
        airports[["ident", "name", "type", "municipality"]],
        left_on="nodeLabel",
        right_on="ident",
        how="inner"
    )

    nodes = nodes[nodes["type"] == "large_airport"].reset_index(drop=True)
    nodes["nodeID"] = np.arange(len(nodes))

    edges = edges[
        edges["X1.1"].isin(nodes["nodeID"]) &
        edges["X2"].isin(nodes["nodeID"])
    ]

    return layers, nodes, edges

Layers, Nodes, Edges = load_data()
NUM_LAYERS = len(Layers)

# --------------------------------------------------
# GRAPH BUILDING
# --------------------------------------------------
@st.cache_data
def build_graphs():
    graphs = {}

    # aggregated
    G_agg = nx.Graph()
    for _, r in Nodes.iterrows():
        G_agg.add_node(
            r.nodeID,
            label=r.nodeLabel,
            name=r.name,
            city=r.municipality,
            lat=r.nodeLat,
            lon=r.nodeLong
        )

    for _, e in Edges.iterrows():
        G_agg.add_edge(e.X1_1, e.X2, weight=e.X1_0)

    graphs["aggregated"] = G_agg

    # per layer
    for i in range(1, NUM_LAYERS + 1):
        G = nx.Graph()
        layer_edges = Edges[Edges["X1"] == i]

        for _, r in Nodes.iterrows():
            G.add_node(
                r.nodeID,
                label=r.nodeLabel,
                name=r.name,
                city=r.municipality,
                lat=r.nodeLat,
                lon=r.nodeLong
            )

        for _, e in layer_edges.iterrows():
            G.add_edge(e.X1_1, e.X2, weight=e.X1_0)

        graphs[i] = G

    return graphs

GRAPHS = build_graphs()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Configuración")

network_type = st.sidebar.selectbox(
    "Tipo de red",
    ["Agregada", "Por aerolínea"]
)

if network_type == "Por aerolínea":
    selected_layer = st.sidebar.selectbox(
        "Aerolínea",
        Layers["nodeLabel"]
    )
    layer_id = Layers.index[Layers["nodeLabel"] == selected_layer][0] + 1
else:
    layer_id = "aggregated"

# --------------------------------------------------
# MAP CREATION
# --------------------------------------------------
def create_map(G):
    m = folium.Map(location=[50, 10], zoom_start=4, tiles="CartoDB positron")

    for u, v in G.edges():
        folium.PolyLine(
            locations=[
                (G.nodes[u]["lat"], G.nodes[u]["lon"]),
                (G.nodes[v]["lat"], G.nodes[v]["lon"])
            ],
            color="blue",
            weight=2,
            opacity=0.4
        ).add_to(m)

    for n, d in G.nodes(data=True):
        folium.CircleMarker(
            location=(d["lat"], d["lon"]),
            radius=6,
            color="white",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.8,
            popup=f"""
            <b>{d['label']}</b><br>
            {d['name']}<br>
            {d['city']}<br>
            Conexiones: {G.degree(n)}
            """
        ).add_to(m)

    return m

# --------------------------------------------------
# MAIN VIEW
# --------------------------------------------------
G = GRAPHS[layer_id]

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🌍 Red de Aeropuertos")
    st_folium(create_map(G), width=1100, height=650)

with col2:
    st.subheader("📊 Estadísticas")
    st.metric("Aeropuertos", G.number_of_nodes())
    st.metric("Conexiones", G.number_of_edges())
    st.metric("Densidad", round(nx.density(G), 4))

# --------------------------------------------------
# PAGE RANK
# --------------------------------------------------
st.header("📈 PageRank")

if network_type == "Por aerolínea":
    pr = nx.pagerank(G)
    pr_df = (
        pd.DataFrame(pr.items(), columns=["nodeID", "pagerank"])
        .merge(Nodes, on="nodeID")
        .sort_values("pagerank", ascending=False)
    )

    st.dataframe(
        pr_df[["nodeLabel", "name", "municipality", "pagerank"]]
        .head(15),
        use_container_width=True
    )
else:
    st.info("Selecciona una aerolínea para ver PageRank")

