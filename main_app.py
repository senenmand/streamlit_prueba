"""
Dashboard Interactivo de Red de Aeropuertos Europeos
Implementado con Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Verificar e importar dependencias críticas
try:
    import networkx as nx
except ImportError:
    st.error("❌ NetworkX no está instalado. Ejecuta: pip install networkx")
    st.stop()

try:
    from scipy.sparse import csr_matrix
except ImportError:
    st.error("❌ SciPy no está instalado. Ejecuta: pip install scipy")
    st.stop()

try:
    import plotly.graph_objects as go
except ImportError:
    st.error("❌ Plotly no está instalado. Ejecuta: pip install plotly")
    st.stop()

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Dashboard de Aeropuertos Europeos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

@st.cache_data
def load_data():
    """Carga y prepara todos los datos necesarios"""
    # Cargar datos
    layers = pd.read_csv("data/EUAirTransportation_layers.txt", sep=r'\s+')
    nodes = pd.read_csv("data/EUAirTransportation_nodes.txt", sep=r'\s+')
    edges = pd.read_csv("data/EUAirTransportation_multiplex.edges", sep=r'\s+')
    airports = pd.read_csv("data/airports.csv")
    
    # Hacer merge con información de aeropuertos
    nodes = nodes.merge(
        airports[['ident', 'name', 'type', 'municipality']],
        left_on='nodeLabel',
        right_on='ident',
        how='inner'
    )
    
    # Renombrar columnas
    nodes.rename(columns={
        'name': 'airportName',
        'type': 'airportType',
        'municipality': 'city'
    }, inplace=True)
    
    # Filtrar solo aeropuertos grandes
    nodes = nodes[nodes['airportType'] == 'large_airport'].reset_index(drop=True)
    large_airport_ids = set(nodes['nodeID'])
    
    # Filtrar edges para incluir solo aeropuertos grandes
    edges = edges[
        edges.iloc[:, 1].isin(large_airport_ids) & 
        edges.iloc[:, 2].isin(large_airport_ids)
    ].copy()
    
    # Reindexar nodos
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(nodes['nodeID'])}
    nodes['nodeID'] = range(len(nodes))
    edges.iloc[:, 1] = edges.iloc[:, 1].map(node_id_map)
    edges.iloc[:, 2] = edges.iloc[:, 2].map(node_id_map)
    
    # Renombrar columnas de edges para mayor claridad
    edges.columns = ['layer', 'weight', 'node1', 'node2']
    
    return layers, nodes, edges


@st.cache_resource
def create_layer_graphs(_nodes, _edges, num_layers):
    """Crea un grafo para cada capa (aerolínea)"""
    graphs = {}
    adjacency_matrices = {}
    num_nodes = len(_nodes)
    
    for layer_id in range(1, num_layers + 1):
        # Filtrar aristas de esta capa
        layer_edges = _edges[_edges['layer'] == layer_id]
        
        # Crear grafo de NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        for _, row in layer_edges.iterrows():
            G.add_edge(int(row['node1']), int(row['node2']), weight=row['weight'])
        
        # Añadir atributos de nodos
        for idx, node_row in _nodes.iterrows():
            G.nodes[idx]['lon'] = node_row['nodeLong']
            G.nodes[idx]['lat'] = node_row['nodeLat']
            G.nodes[idx]['label'] = node_row['nodeLabel']
            G.nodes[idx]['airport_name'] = node_row['airportName']
            G.nodes[idx]['city'] = node_row['city']
        
        # Crear matriz de adyacencia
        adj_matrix = nx.adjacency_matrix(G)
        
        graphs[layer_id] = G
        adjacency_matrices[layer_id] = adj_matrix
    
    return graphs, adjacency_matrices


@st.cache_resource
def create_aggregate_graph(_graphs, _nodes):
    """Crea el grafo agregado combinando todas las capas"""
    G_agg = nx.Graph()
    num_nodes = len(_nodes)
    G_agg.add_nodes_from(range(num_nodes))
    
    # Combinar aristas de todos los grafos
    for G in _graphs.values():
        for u, v, data in G.edges(data=True):
            if G_agg.has_edge(u, v):
                G_agg[u][v]['weight'] += data.get('weight', 1)
            else:
                G_agg.add_edge(u, v, weight=data.get('weight', 1))
    
    # Añadir atributos de nodos
    for idx, node_row in _nodes.iterrows():
        G_agg.nodes[idx]['lon'] = node_row['nodeLong']
        G_agg.nodes[idx]['lat'] = node_row['nodeLat']
        G_agg.nodes[idx]['label'] = node_row['nodeLabel']
        G_agg.nodes[idx]['airport_name'] = node_row['airportName']
        G_agg.nodes[idx]['city'] = node_row['city']
    
    return G_agg


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_pagerank(adj_matrix):
    """Calcula PageRank usando la matriz de adyacencia"""
    G = nx.from_scipy_sparse_array(adj_matrix)
    pr = nx.pagerank(G, alpha=0.85)
    pr_array = np.array([pr.get(i, 0) for i in range(adj_matrix.shape[0])])
    return pr_array


def get_path_layers(path_nodes, edges_df):
    """Obtiene las capas (aerolíneas) usadas en un camino"""
    layers = []
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        edge_row = edges_df[
            ((edges_df['node1'] == n1) & (edges_df['node2'] == n2)) |
            ((edges_df['node1'] == n2) & (edges_df['node2'] == n1))
        ]
        if not edge_row.empty:
            layers.append(int(edge_row.iloc[0]['layer']))
        else:
            layers.append(None)
    return layers


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def create_map_figure(G, nodes_subset=None, edges_to_draw=None, 
                     origin_node=None, dest_node=None, path_edges=None,
                     pagerank_values=None, mapbox_style='open-street-map'):
    """Crea una figura de mapa con Plotly"""
    
    # Mapeo de estilos
    style_map = {
        'CartoDB.Positron': 'carto-positron',
        'Esri.WorldImagery': 'satellite-streets',
        'OpenStreetMap': 'open-street-map',
        'Stadia.AlidadeSmooth': 'stamen-terrain'
    }
    
    mapbox_style = style_map.get(mapbox_style, 'open-street-map')
    
    fig = go.Figure()
    
    # Determinar qué nodos dibujar
    if nodes_subset is None:
        nodes_subset = list(G.nodes())
    
    # Filtrar nodos que tienen grado > 0
    nodes_with_edges = [n for n in nodes_subset if G.degree(n) > 0]
    
    # Dibujar aristas normales
    if edges_to_draw is not None:
        for u, v in edges_to_draw:
            if u in G.nodes() and v in G.nodes():
                x0, y0 = G.nodes[u]['lon'], G.nodes[u]['lat']
                x1, y1 = G.nodes[v]['lon'], G.nodes[v]['lat']
                
                fig.add_trace(go.Scattermapbox(
                    lon=[x0, x1, None],
                    lat=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='rgba(51, 136, 255, 0.3)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
    else:
        # Dibujar todas las aristas del subgrafo
        for u, v in G.edges():
            if u in nodes_with_edges and v in nodes_with_edges:
                x0, y0 = G.nodes[u]['lon'], G.nodes[u]['lat']
                x1, y1 = G.nodes[v]['lon'], G.nodes[v]['lat']
                
                fig.add_trace(go.Scattermapbox(
                    lon=[x0, x1, None],
                    lat=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='rgba(51, 136, 255, 0.3)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    # Aristas del camino (si existen)
    if path_edges is not None:
        for i, (u, v) in enumerate(path_edges):
            x0, y0 = G.nodes[u]['lon'], G.nodes[u]['lat']
            x1, y1 = G.nodes[v]['lon'], G.nodes[v]['lat']
            
            fig.add_trace(go.Scattermapbox(
                lon=[x0, x1, None],
                lat=[y0, y1, None],
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='text',
                text=f"Tramo {i+1}",
                showlegend=False
            ))
    
    # Preparar datos de nodos
    node_lons = []
    node_lats = []
    node_texts = []
    node_colors = []
    node_sizes = []
    
    for node in nodes_with_edges:
        if node == origin_node or node == dest_node:
            continue
            
        node_lons.append(G.nodes[node]['lon'])
        node_lats.append(G.nodes[node]['lat'])
        
        text = (f"<b>{G.nodes[node]['label']}</b><br>"
                f"{G.nodes[node]['airport_name']}<br>"
                f"{G.nodes[node]['city']}<br>"
                f"Conexiones: {G.degree(node)}")
        
        if pagerank_values is not None:
            text += f"<br>PageRank: {pagerank_values[node]:.5f}"
        
        node_texts.append(text)
        
        # Color y tamaño
        if pagerank_values is not None:
            pr_norm = (pagerank_values[node] - pagerank_values.min()) / \
                     (pagerank_values.max() - pagerank_values.min() + 1e-10)
            node_colors.append(f'rgba(255, 0, 0, {0.3 + 0.7 * pr_norm})')
            node_sizes.append(5 + 15 * pr_norm)
        else:
            node_colors.append('rgba(51, 136, 255, 0.7)')
            node_sizes.append(8)
    
    # Nodos normales
    if node_lons:
        fig.add_trace(go.Scattermapbox(
            lon=node_lons,
            lat=node_lats,
            mode='markers+text',
            marker=dict(
                size=node_sizes if pagerank_values is not None else 8,
                color=node_colors,
                opacity=1,
            ),
            text=[G.nodes[n]['label'] for n in nodes_with_edges 
                  if n != origin_node and n != dest_node],
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hoverinfo='text',
            hovertext=node_texts,
            showlegend=False
        ))
    
    # Nodo de origen (rojo)
    if origin_node is not None and origin_node in G.nodes():
        fig.add_trace(go.Scattermapbox(
            lon=[G.nodes[origin_node]['lon']],
            lat=[G.nodes[origin_node]['lat']],
            mode='markers+text',
            marker=dict(size=15, color='red', opacity=0.9),
            text=[G.nodes[origin_node]['label']],
            textposition='top center',
            textfont=dict(size=11, color='red', family='Arial Black'),
            hoverinfo='text',
            hovertext=(f"<b>ORIGEN: {G.nodes[origin_node]['label']}</b><br>"
                      f"{G.nodes[origin_node]['airport_name']}<br>"
                      f"{G.nodes[origin_node]['city']}"),
            showlegend=False
        ))
    
    # Nodo de destino (amarillo)
    if dest_node is not None and dest_node in G.nodes():
        fig.add_trace(go.Scattermapbox(
            lon=[G.nodes[dest_node]['lon']],
            lat=[G.nodes[dest_node]['lat']],
            mode='markers+text',
            marker=dict(size=15, color='yellow', opacity=0.9, 
                       line=dict(width=2, color='orange')),
            text=[G.nodes[dest_node]['label']],
            textposition='top center',
            textfont=dict(size=11, color='#cc9900', family='Arial Black'),
            hoverinfo='text',
            hovertext=(f"<b>DESTINO: {G.nodes[dest_node]['label']}</b><br>"
                      f"{G.nodes[dest_node]['airport_name']}<br>"
                      f"{G.nodes[dest_node]['city']}"),
            showlegend=False
        ))
    
    # Configuración del mapa
    fig.update_layout(
        mapbox=dict(
            style=mapbox_style,
            center=dict(lat=50, lon=10),
            zoom=3.5
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        hovermode='closest'
    )
    
    return fig


# ============================================================================
# CARGAR DATOS
# ============================================================================

# Mostrar spinner mientras carga
with st.spinner('Cargando datos...'):
    layers, nodes, edges = load_data()
    num_layers = len(layers)
    num_nodes = len(nodes)
    
    layer_graphs, adjacency_matrices = create_layer_graphs(nodes, edges, num_layers)
    G_agg = create_aggregate_graph(layer_graphs, nodes)

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Título principal
st.title("✈️ Dashboard de Red de Aeropuertos Europeos")
st.markdown("---")

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Vista General", 
    "📊 Análisis de Red", 
    "🛫 Conexiones", 
    "⭐ PageRank"
])

# ============================================================================
# PESTAÑA 1: VISTA GENERAL
# ============================================================================

with tab1:
    st.header("Red Completa de Aeropuertos Europeos")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuración")
        
        layer_type_general = st.radio(
            "Tipo de red:",
            options=['aggregated', 'layer'],
            format_func=lambda x: 'Agregada (todas las aerolíneas)' if x == 'aggregated' else 'Por aerolínea individual',
            key='layer_type_general'
        )
        
        if layer_type_general == 'layer':
            layer_select_general = st.selectbox(
                "Seleccionar aerolínea:",
                options=range(1, num_layers + 1),
                format_func=lambda x: layers.iloc[x-1]['nodeLabel'],
                key='layer_select_general'
            )
        
        st.markdown("---")
        st.subheader("Estilo de Mapa")
        map_style_general = st.selectbox(
            "Seleccionar estilo:",
            options=['CartoDB.Positron', 'Esri.WorldImagery', 'OpenStreetMap', 'Stadia.AlidadeSmooth'],
            format_func=lambda x: {
                'CartoDB.Positron': 'Minimalista',
                'Esri.WorldImagery': 'Satélite',
                'OpenStreetMap': 'Callejero',
                'Stadia.AlidadeSmooth': 'Suave'
            }[x],
            key='map_style_general'
        )
    
    with col2:
        # Seleccionar grafo
        if layer_type_general == 'aggregated':
            G = G_agg
        else:
            G = layer_graphs[layer_select_general]
            # Filtrar nodos sin conexiones
            nodes_with_edges = [n for n in G.nodes() if G.degree(n) > 0]
            G = G.subgraph(nodes_with_edges).copy()
        
        # Crear y mostrar mapa
        fig = create_map_figure(G, mapbox_style=map_style_general)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Nodos totales", G.number_of_nodes())
        with col_b:
            st.metric("Conexiones totales", G.number_of_edges())
        with col_c:
            st.metric("Densidad", f"{nx.density(G):.4f}")
        with col_d:
            st.metric("Componentes", nx.number_connected_components(G))

# ============================================================================
# PESTAÑA 2: ANÁLISIS DE RED
# ============================================================================

with tab2:
    st.header("Visualización de la Red de Aeropuertos")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuración")
        
        layer_type_analisis = st.radio(
            "Tipo de red:",
            options=['aggregated', 'layer'],
            format_func=lambda x: 'Agregada (todas las aerolíneas)' if x == 'aggregated' else 'Por aerolínea individual',
            key='layer_type_analisis'
        )
        
        if layer_type_analisis == 'layer':
            layer_select_analisis = st.selectbox(
                "Seleccionar aerolínea:",
                options=range(1, num_layers + 1),
                format_func=lambda x: layers.iloc[x-1]['nodeLabel'],
                key='layer_select_analisis'
            )
        
        # Selector de aeropuerto
        airport_options = [(row['nodeLabel'], f"{row['nodeLabel']} - {row['airportName']}") 
                          for _, row in nodes.iterrows()]
        
        airport_select = st.selectbox(
            "Aeropuerto de origen:",
            options=[x[0] for x in airport_options],
            format_func=lambda x: next(y[1] for y in airport_options if y[0] == x),
            key='airport_select_analisis'
        )
        
        st.markdown("---")
        st.subheader("Estadísticas de la Red")
        
        # Seleccionar grafo
        if layer_type_analisis == 'aggregated':
            G_analisis = G_agg
        else:
            G_analisis = layer_graphs[layer_select_analisis]
        
        st.text(f"Nodos totales: {G_analisis.number_of_nodes()}")
        st.text(f"Conexiones totales: {G_analisis.number_of_edges()}")
        st.text(f"Densidad: {nx.density(G_analisis):.4f}")
        st.text(f"Componentes conectados: {nx.number_connected_components(G_analisis)}")
        
        st.markdown("---")
        st.subheader("Info del Aeropuerto")
        
        airport_idx = nodes[nodes['nodeLabel'] == airport_select].index[0]
        
        if airport_idx in G_analisis.nodes():
            st.text(f"Código: {airport_select}")
            st.text(f"Nombre: {G_analisis.nodes[airport_idx]['airport_name']}")
            st.text(f"Ciudad: {G_analisis.nodes[airport_idx]['city']}")
            st.text(f"Conexiones directas: {G_analisis.degree(airport_idx)}")
        else:
            st.warning("Aeropuerto no encontrado en esta red")
        
        st.markdown("---")
        st.subheader("Estilo de Mapa")
        map_style_analisis = st.selectbox(
            "Seleccionar estilo:",
            options=['CartoDB.Positron', 'Esri.WorldImagery', 'OpenStreetMap', 'Stadia.AlidadeSmooth'],
            format_func=lambda x: {
                'CartoDB.Positron': 'Minimalista',
                'Esri.WorldImagery': 'Satélite',
                'OpenStreetMap': 'Callejero',
                'Stadia.AlidadeSmooth': 'Suave'
            }[x],
            key='map_style_analisis'
        )
    
    with col2:
        if airport_idx in G_analisis.nodes():
            # Crear subgrafo con vecinos
            neighbors = list(G_analisis.neighbors(airport_idx))
            subgraph_nodes = [airport_idx] + neighbors
            G_sub = G_analisis.subgraph(subgraph_nodes).copy()
            
            # Crear mapa
            edges_list = list(G_sub.edges())
            fig = create_map_figure(
                G_sub,
                nodes_subset=subgraph_nodes,
                edges_to_draw=edges_list,
                origin_node=airport_idx,
                mapbox_style=map_style_analisis
            )
        else:
            fig = create_map_figure(G_analisis, mapbox_style=map_style_analisis)
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PESTAÑA 3: CONEXIONES
# ============================================================================

with tab3:
    st.header("Mapa de Rutas")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuración")
        
        layer_type_conexiones = st.radio(
            "Tipo de red:",
            options=['aggregated', 'layer'],
            format_func=lambda x: 'Agregada (todas las aerolíneas)' if x == 'aggregated' else 'Por aerolínea individual',
            key='layer_type_conexiones'
        )
        
        if layer_type_conexiones == 'layer':
            layer_select_conexiones = st.selectbox(
                "Seleccionar aerolínea:",
                options=range(1, num_layers + 1),
                format_func=lambda x: layers.iloc[x-1]['nodeLabel'],
                key='layer_select_conexiones'
            )
        
        # Selectores de aeropuertos
        airport_options = [(row['nodeLabel'], f"{row['nodeLabel']} - {row['airportName']}") 
                          for _, row in nodes.iterrows()]
        
        airport_origin = st.selectbox(
            "Aeropuerto de origen:",
            options=[x[0] for x in airport_options],
            format_func=lambda x: next(y[1] for y in airport_options if y[0] == x),
            key='airport_origin_conexiones'
        )
        
        airport_dest = st.selectbox(
            "Aeropuerto de destino:",
            options=[x[0] for x in airport_options],
            format_func=lambda x: next(y[1] for y in airport_options if y[0] == x),
            index=1 if len(airport_options) > 1 else 0,
            key='airport_dest_conexiones'
        )
        
        st.markdown("---")
        st.subheader("Info del Trayecto")
        
        # Seleccionar grafo
        if layer_type_conexiones == 'aggregated':
            G_conexiones = G_agg
        else:
            G_conexiones = layer_graphs[layer_select_conexiones]
        
        # Encontrar índices
        origin_idx = nodes[nodes['nodeLabel'] == airport_origin].index[0]
        dest_idx = nodes[nodes['nodeLabel'] == airport_dest].index[0]
        
        # Calcular camino más corto
        try:
            path = nx.shortest_path(G_conexiones, origin_idx, dest_idx)
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            path_layers = get_path_layers(path, edges)
            
            # Mostrar info
            airport_names = [G_conexiones.nodes[n]['airport_name'] for n in path]
            st.markdown("**Ruta:**")
            st.write(" → ".join(airport_names))
            
            st.markdown("---")
            st.markdown("**Trayectos por aerolínea:**")
            for i, layer in enumerate(path_layers):
                if layer is not None:
                    layer_name = layers.iloc[layer-1]['nodeLabel']
                    st.text(f"Trayecto {i+1}: {layer_name}")
            
            path_found = True
        except nx.NetworkXNoPath:
            st.warning("No hay ruta disponible entre estos aeropuertos")
            path_found = False
        
        st.markdown("---")
        st.subheader("Estilo de Mapa")
        map_style_conexiones = st.selectbox(
            "Seleccionar estilo:",
            options=['CartoDB.Positron', 'Esri.WorldImagery', 'OpenStreetMap', 'Stadia.AlidadeSmooth'],
            format_func=lambda x: {
                'CartoDB.Positron': 'Minimalista',
                'Esri.WorldImagery': 'Satélite',
                'OpenStreetMap': 'Callejero',
                'Stadia.AlidadeSmooth': 'Suave'
            }[x],
            key='map_style_conexiones'
        )
    
    with col2:
        if path_found:
            fig = create_map_figure(
                G_conexiones,
                nodes_subset=path,
                path_edges=path_edges,
                origin_node=origin_idx,
                dest_node=dest_idx,
                mapbox_style=map_style_conexiones
            )
        else:
            fig = create_map_figure(
                G_conexiones,
                nodes_subset=[origin_idx, dest_idx],
                origin_node=origin_idx,
                dest_node=dest_idx,
                mapbox_style=map_style_conexiones
            )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PESTAÑA 4: PAGERANK
# ============================================================================

with tab4:
    st.header("Visualización de PageRank por Aerolínea")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuración")
        
        layer_select_pagerank = st.selectbox(
            "Seleccionar aerolínea:",
            options=range(1, num_layers + 1),
            format_func=lambda x: layers.iloc[x-1]['nodeLabel'],
            key='layer_select_pagerank'
        )
        
        st.markdown("---")
        st.subheader("Estadísticas de PageRank")
        
        # Obtener grafo y calcular PageRank
        G_pagerank = layer_graphs[layer_select_pagerank]
        adj_matrix = adjacency_matrices[layer_select_pagerank]
        
        # Filtrar nodos sin conexiones
        nodes_with_edges = [n for n in G_pagerank.nodes() if G_pagerank.degree(n) > 0]
        G_sub = G_pagerank.subgraph(nodes_with_edges).copy()
        
        # Calcular PageRank
        pr = calculate_pagerank(adj_matrix)
        pr_filtered = pr[nodes_with_edges]
        
        st.text(f"Nodos con conexiones: {len(nodes_with_edges)}")
        st.text(f"Conexiones totales: {G_sub.number_of_edges()}")
        st.text(f"PageRank promedio: {pr_filtered.mean():.5f}")
        st.text(f"PageRank máximo: {pr_filtered.max():.5f}")
        
        max_pr_node = nodes_with_edges[np.argmax(pr_filtered)]
        st.text(f"Nodo más importante: {G_sub.nodes[max_pr_node]['label']}")
        
        st.markdown("---")
        st.subheader("Estilo de Mapa")
        map_style_pagerank = st.selectbox(
            "Seleccionar estilo:",
            options=['CartoDB.Positron', 'Esri.WorldImagery', 'OpenStreetMap', 'Stadia.AlidadeSmooth'],
            format_func=lambda x: {
                'CartoDB.Positron': 'Minimalista',
                'Esri.WorldImagery': 'Satélite',
                'OpenStreetMap': 'Callejero',
                'Stadia.AlidadeSmooth': 'Suave'
            }[x],
            key='map_style_pagerank'
        )
    
    with col2:
        # Crear mapa
        fig = create_map_figure(
            G_sub,
            nodes_subset=nodes_with_edges,
            edges_to_draw=list(G_sub.edges()),
            pagerank_values=pr,
            mapbox_style=map_style_pagerank
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dashboard de Red de Aeropuertos Europeos | Desarrollado con Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
