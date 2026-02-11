import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

st.set_page_config(page_title="Visualizador de Grafo", layout="wide")

# --------------------------------------------------
# Carga de datos (cacheada)
# --------------------------------------------------
@st.cache_data
def load_data(path_edges: str):
    # Leer archivo .edges SIN cabecera
    edges = pd.read_csv(
        path_edges,
        sep=r"\s+",
        header=None
    )

    # Asignar nombres de columnas de forma segura
    edges.columns = ["src", "dst", "attr", "weight"][:len(edges.columns)]

    # Crear nodos únicos a partir de src y dst
    nodes = pd.DataFrame(
        pd.unique(edges[["src", "dst"]].values.ravel()),
        columns=["node"]
    )

    # Asignar IDs consecutivos
    nodes["nodeID"] = np.arange(len(nodes))

    # Mapeo nodo → ID
    node_map = dict(zip(nodes["node"], nodes["nodeID"]))

    # Reemplazar valores por nodeID
    edges["src_id"] = edges["src"].map(node_map)
    edges["dst_id"] = edges["dst"].map(node_map)

    # Filtrar posibles NaN (por seguridad)
    edges = edges.dropna(subset=["src_id", "dst_id"])

    return nodes, edges


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("📊 Visualización de Grafo (.edges)")

path_edges = st.text_input(
    "Ruta del archivo .edges",
    value="archivo.edges"
)

if path_edges:
    try:
        nodes, edges = load_data(path_edges)

        st.subheader("Nodos")
        st.dataframe(nodes.head())

        st.subheader("Aristas")
        st.dataframe(edges.head())

        # Crear grafo
        G = nx.from_pandas_edgelist(
            edges,
            source="src_id",
            target="dst_id",
            edge_attr=True
        )

        st.success(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    except Exception as e:
        st.error("Error al cargar el archivo")
        st.exception(e)
