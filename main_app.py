"""
Dashboard Interactivo de Red de Aeropuertos Europeos
Migrado de R/Shiny a Python/Dash
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Inicializar la aplicación Dash con tema Bootstrap
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    suppress_callback_exceptions=True
)

app.title = "Dashboard de Aeropuertos Europeos"

# ============================================================================
# FUNCIONES AUXILIARES PARA GRAFOS
# ============================================================================

def load_data():
    """Carga y prepara todos los datos necesarios"""
    # Cargar datos
    layers = pd.read_csv("dataset/EUAirTransportation_layers.txt", sep=r'\s+')
    nodes = pd.read_csv("dataset/EUAirTransportation_nodes.txt", sep=r'\s+')
    edges = pd.read_csv("dataset/EUAirTransportation_multiplex.edges", sep=r'\s+')
    airports = pd.read_csv("dataset/airports.csv")
    
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


def create_layer_graphs(nodes, edges, num_layers):
    """Crea un grafo para cada capa (aerolínea)"""
    graphs = {}
    adjacency_matrices = {}
    num_nodes = len(nodes)
    
    for layer_id in range(1, num_layers + 1):
        # Filtrar aristas de esta capa
        layer_edges = edges[edges['layer'] == layer_id]
        
        # Crear grafo de NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        for _, row in layer_edges.iterrows():
            G.add_edge(int(row['node1']), int(row['node2']), weight=row['weight'])
        
        # Añadir atributos de nodos
        for idx, node_row in nodes.iterrows():
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


def create_aggregate_graph(graphs, nodes):
    """Crea el grafo agregado combinando todas las capas"""
    G_agg = nx.Graph()
    num_nodes = len(nodes)
    G_agg.add_nodes_from(range(num_nodes))
    
    # Combinar aristas de todos los grafos
    for G in graphs.values():
        for u, v, data in G.edges(data=True):
            if G_agg.has_edge(u, v):
                G_agg[u][v]['weight'] += data.get('weight', 1)
            else:
                G_agg.add_edge(u, v, weight=data.get('weight', 1))
    
    # Añadir atributos de nodos
    for idx, node_row in nodes.iterrows():
        G_agg.nodes[idx]['lon'] = node_row['nodeLong']
        G_agg.nodes[idx]['lat'] = node_row['nodeLat']
        G_agg.nodes[idx]['label'] = node_row['nodeLabel']
        G_agg.nodes[idx]['airport_name'] = node_row['airportName']
        G_agg.nodes[idx]['city'] = node_row['city']
    
    return G_agg


def calculate_pagerank(adj_matrix):
    """Calcula PageRank usando la matriz de adyacencia"""
    # Convertir a NetworkX graph
    G = nx.from_scipy_sparse_array(adj_matrix)
    
    # Calcular PageRank
    pr = nx.pagerank(G, alpha=0.85)
    
    # Convertir a array
    pr_array = np.array([pr.get(i, 0) for i in range(adj_matrix.shape[0])])
    
    return pr_array


def get_path_layers(path_nodes, edges):
    """Obtiene las capas (aerolíneas) usadas en un camino"""
    layers = []
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        # Buscar la arista en el dataframe
        edge_row = edges[
            ((edges['node1'] == n1) & (edges['node2'] == n2)) |
            ((edges['node1'] == n2) & (edges['node2'] == n1))
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
    """
    Crea una figura de mapa con Plotly
    
    Parameters:
    - G: grafo de NetworkX
    - nodes_subset: lista de nodos a dibujar (None = todos)
    - edges_to_draw: lista de tuplas (u, v) para dibujar aristas
    - origin_node: nodo de origen (marcado en rojo)
    - dest_node: nodo de destino (marcado en amarillo)
    - path_edges: aristas del camino más corto (dibujadas en azul)
    - pagerank_values: valores de PageRank para dimensionar nodos
    - mapbox_style: estilo del mapa
    """
    
    # Mapeo de estilos
    style_map = {
        'CartoDB.Positron': 'carto-positron',
        'Esri.WorldImagery': 'satellite',
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
    
    # Dibujar aristas
    edge_traces = []
    
    # Aristas normales
    if edges_to_draw is not None:
        for u, v in edges_to_draw:
            if u in G.nodes() and v in G.nodes():
                x0, y0 = G.nodes[u]['lon'], G.nodes[u]['lat']
                x1, y1 = G.nodes[v]['lon'], G.nodes[v]['lat']
                
                edge_trace = go.Scattermapbox(
                    lon=[x0, x1, None],
                    lat=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='rgba(51, 136, 255, 0.3)'),
                    hoverinfo='skip',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
    else:
        # Dibujar todas las aristas del subgrafo
        for u, v in G.edges():
            if u in nodes_with_edges and v in nodes_with_edges:
                x0, y0 = G.nodes[u]['lon'], G.nodes[u]['lat']
                x1, y1 = G.nodes[v]['lon'], G.nodes[v]['lat']
                
                edge_trace = go.Scattermapbox(
                    lon=[x0, x1, None],
                    lat=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='rgba(51, 136, 255, 0.3)'),
                    hoverinfo='skip',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
    
    # Aristas del camino (si existen)
    if path_edges is not None:
        for i, (u, v) in enumerate(path_edges):
            x0, y0 = G.nodes[u]['lon'], G.nodes[u]['lat']
            x1, y1 = G.nodes[v]['lon'], G.nodes[v]['lat']
            
            path_trace = go.Scattermapbox(
                lon=[x0, x1, None],
                lat=[y0, y1, None],
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='text',
                text=f"Tramo {i+1}",
                showlegend=False
            )
            edge_traces.append(path_trace)
    
    # Añadir todas las aristas
    for trace in edge_traces:
        fig.add_trace(trace)
    
    # Preparar datos de nodos
    node_lons = []
    node_lats = []
    node_texts = []
    node_colors = []
    node_sizes = []
    
    for node in nodes_with_edges:
        if node == origin_node:
            continue  # Lo dibujamos después
        if node == dest_node:
            continue  # Lo dibujamos después
            
        node_lons.append(G.nodes[node]['lon'])
        node_lats.append(G.nodes[node]['lat'])
        
        # Texto del hover
        text = (f"<b>{G.nodes[node]['label']}</b><br>"
                f"{G.nodes[node]['airport_name']}<br>"
                f"{G.nodes[node]['city']}<br>"
                f"Conexiones: {G.degree(node)}")
        
        if pagerank_values is not None:
            text += f"<br>PageRank: {pagerank_values[node]:.5f}"
        
        node_texts.append(text)
        
        # Color y tamaño
        if pagerank_values is not None:
            # Color y tamaño basado en PageRank
            pr_norm = (pagerank_values[node] - pagerank_values.min()) / \
                     (pagerank_values.max() - pagerank_values.min() + 1e-10)
            node_colors.append(f'rgba(255, 0, 0, {0.3 + 0.7 * pr_norm})')
            node_sizes.append(5 + 15 * pr_norm)
        else:
            node_colors.append('rgba(51, 136, 255, 0.7)')
            node_sizes.append(8)
    
    # Nodos normales
    if node_lons:
        node_trace = go.Scattermapbox(
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
        )
        fig.add_trace(node_trace)
    
    # Nodo de origen (rojo)
    if origin_node is not None and origin_node in G.nodes():
        origin_trace = go.Scattermapbox(
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
        )
        fig.add_trace(origin_trace)
    
    # Nodo de destino (amarillo)
    if dest_node is not None and dest_node in G.nodes():
        dest_trace = go.Scattermapbox(
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
        )
        fig.add_trace(dest_trace)
    
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
# CARGAR DATOS AL INICIO
# ============================================================================

print("Cargando datos...")
layers, nodes, edges = load_data()
num_layers = len(layers)
num_nodes = len(nodes)

print("Creando grafos por capa...")
layer_graphs, adjacency_matrices = create_layer_graphs(nodes, edges, num_layers)

print("Creando grafo agregado...")
G_agg = create_aggregate_graph(layer_graphs, nodes)

print(f"Datos cargados: {num_nodes} aeropuertos, {num_layers} aerolíneas")

# ============================================================================
# LAYOUT DEL DASHBOARD
# ============================================================================

# Crear opciones para los selectores
airline_options = [{'label': row['nodeLabel'], 'value': i+1} 
                   for i, row in layers.iterrows()]

airport_options = [
    {'label': f"{row['nodeLabel']} - {row['airportName']}", 'value': row['nodeLabel']}
    for _, row in nodes.iterrows()
]

map_style_options = [
    {'label': 'Minimalista', 'value': 'CartoDB.Positron'},
    {'label': 'Satélite', 'value': 'Esri.WorldImagery'},
    {'label': 'Callejero', 'value': 'OpenStreetMap'},
    {'label': 'Suave', 'value': 'Stadia.AlidadeSmooth'}
]

# Layout principal con pestañas
app.layout = dbc.Container([
    html.H1("Dashboard de Red de Aeropuertos Europeos", 
            className="text-center my-4"),
    
    dbc.Tabs([
        # ========== PESTAÑA 1: VISTA GENERAL ==========
        dbc.Tab(label="Vista General", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Configuración", className="mt-3"),
                    html.Hr(),
                    
                    html.Label("Tipo de red:"),
                    dcc.Dropdown(
                        id='layer-type-general',
                        options=[
                            {'label': 'Agregada (todas las aerolíneas)', 'value': 'aggregated'},
                            {'label': 'Por aerolínea individual', 'value': 'layer'}
                        ],
                        value='aggregated',
                        clearable=False
                    ),
                    
                    html.Div(id='airline-selector-general', children=[
                        html.Label("Seleccionar aerolínea:", className="mt-3"),
                        dcc.Dropdown(
                            id='layer-select-general',
                            options=airline_options,
                            value=1,
                            clearable=False
                        )
                    ], style={'display': 'none'}),
                    
                    html.Hr(),
                    html.H4("Estilo de Mapa", className="mt-3"),
                    dcc.Dropdown(
                        id='map-style-general',
                        options=map_style_options,
                        value='CartoDB.Positron',
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.H4("Red Completa de Aeropuertos Europeos", className="mt-3"),
                    dcc.Graph(id='map-general', style={'height': '600px'})
                ], width=9)
            ])
        ]),
        
        # ========== PESTAÑA 2: ANÁLISIS DE RED ==========
        dbc.Tab(label="Análisis de Red", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Configuración", className="mt-3"),
                    html.Hr(),
                    
                    html.Label("Tipo de red:"),
                    dcc.Dropdown(
                        id='layer-type-analisis',
                        options=[
                            {'label': 'Agregada (todas las aerolíneas)', 'value': 'aggregated'},
                            {'label': 'Por aerolínea individual', 'value': 'layer'}
                        ],
                        value='aggregated',
                        clearable=False
                    ),
                    
                    html.Div(id='airline-selector-analisis', children=[
                        html.Label("Seleccionar aerolínea:", className="mt-3"),
                        dcc.Dropdown(
                            id='layer-select-analisis',
                            options=airline_options,
                            value=1,
                            clearable=False
                        )
                    ], style={'display': 'none'}),
                    
                    html.Label("Aeropuerto de origen:", className="mt-3"),
                    dcc.Dropdown(
                        id='airport-select-analisis',
                        options=airport_options,
                        value=nodes.iloc[0]['nodeLabel'],
                        clearable=False
                    ),
                    
                    html.Hr(),
                    html.H4("Estadísticas de la Red"),
                    html.Pre(id='stats-analisis', style={'fontSize': '12px'}),
                    
                    html.Hr(),
                    html.H4("Info del Aeropuerto"),
                    html.Pre(id='airport-info-analisis', style={'fontSize': '12px'}),
                    
                    html.Hr(),
                    html.H4("Estilo de Mapa"),
                    dcc.Dropdown(
                        id='map-style-analisis',
                        options=map_style_options,
                        value='CartoDB.Positron',
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.H4("Visualización de la Red de Aeropuertos", className="mt-3"),
                    dcc.Graph(id='map-analisis', style={'height': '600px'})
                ], width=9)
            ])
        ]),
        
        # ========== PESTAÑA 3: CONEXIONES ==========
        dbc.Tab(label="Conexiones", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Configuración", className="mt-3"),
                    html.Hr(),
                    
                    html.Label("Tipo de red:"),
                    dcc.Dropdown(
                        id='layer-type-conexiones',
                        options=[
                            {'label': 'Agregada (todas las aerolíneas)', 'value': 'aggregated'},
                            {'label': 'Por aerolínea individual', 'value': 'layer'}
                        ],
                        value='aggregated',
                        clearable=False
                    ),
                    
                    html.Div(id='airline-selector-conexiones', children=[
                        html.Label("Seleccionar aerolínea:", className="mt-3"),
                        dcc.Dropdown(
                            id='layer-select-conexiones',
                            options=airline_options,
                            value=1,
                            clearable=False
                        )
                    ], style={'display': 'none'}),
                    
                    html.Label("Aeropuerto de origen:", className="mt-3"),
                    dcc.Dropdown(
                        id='airport-origin-conexiones',
                        options=airport_options,
                        value=nodes.iloc[0]['nodeLabel'],
                        clearable=False
                    ),
                    
                    html.Label("Aeropuerto de destino:", className="mt-3"),
                    dcc.Dropdown(
                        id='airport-dest-conexiones',
                        options=airport_options,
                        value=nodes.iloc[1]['nodeLabel'] if len(nodes) > 1 else nodes.iloc[0]['nodeLabel'],
                        clearable=False
                    ),
                    
                    html.Hr(),
                    html.H4("Info del Trayecto"),
                    html.Div(id='path-info-conexiones', style={'fontSize': '12px'}),
                    
                    html.Hr(),
                    html.H4("Estilo de Mapa"),
                    dcc.Dropdown(
                        id='map-style-conexiones',
                        options=map_style_options,
                        value='CartoDB.Positron',
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.H4("Mapa de Rutas", className="mt-3"),
                    dcc.Graph(id='map-conexiones', style={'height': '600px'})
                ], width=9)
            ])
        ]),
        
        # ========== PESTAÑA 4: PAGERANK ==========
        dbc.Tab(label="PageRank", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Configuración", className="mt-3"),
                    html.Hr(),
                    
                    html.Label("Seleccionar aerolínea:"),
                    dcc.Dropdown(
                        id='layer-select-pagerank',
                        options=airline_options,
                        value=1,
                        clearable=False
                    ),
                    
                    html.Hr(),
                    html.H4("Estadísticas de PageRank"),
                    html.Pre(id='stats-pagerank', style={'fontSize': '12px'}),
                    
                    html.Hr(),
                    html.H4("Estilo de Mapa"),
                    dcc.Dropdown(
                        id='map-style-pagerank',
                        options=map_style_options,
                        value='CartoDB.Positron',
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.H4("Visualización de PageRank por Aerolínea", className="mt-3"),
                    dcc.Graph(id='map-pagerank', style={'height': '600px'})
                ], width=9)
            ])
        ])
    ])
], fluid=True)

# ============================================================================
# CALLBACKS
# ============================================================================

# Callback para mostrar/ocultar selector de aerolínea en Vista General
@app.callback(
    Output('airline-selector-general', 'style'),
    Input('layer-type-general', 'value')
)
def toggle_airline_selector_general(layer_type):
    if layer_type == 'layer':
        return {'display': 'block'}
    return {'display': 'none'}


# Callback para mostrar/ocultar selector de aerolínea en Análisis
@app.callback(
    Output('airline-selector-analisis', 'style'),
    Input('layer-type-analisis', 'value')
)
def toggle_airline_selector_analisis(layer_type):
    if layer_type == 'layer':
        return {'display': 'block'}
    return {'display': 'none'}


# Callback para mostrar/ocultar selector de aerolínea en Conexiones
@app.callback(
    Output('airline-selector-conexiones', 'style'),
    Input('layer-type-conexiones', 'value')
)
def toggle_airline_selector_conexiones(layer_type):
    if layer_type == 'layer':
        return {'display': 'block'}
    return {'display': 'none'}


# Callback para Vista General
@app.callback(
    Output('map-general', 'figure'),
    [Input('layer-type-general', 'value'),
     Input('layer-select-general', 'value'),
     Input('map-style-general', 'value')]
)
def update_general_map(layer_type, layer_id, map_style):
    if layer_type == 'aggregated':
        G = G_agg
    else:
        G = layer_graphs[layer_id]
        # Filtrar nodos sin conexiones
        nodes_with_edges = [n for n in G.nodes() if G.degree(n) > 0]
        G = G.subgraph(nodes_with_edges).copy()
    
    fig = create_map_figure(G, mapbox_style=map_style)
    return fig


# Callback para Análisis de Red
@app.callback(
    [Output('map-analisis', 'figure'),
     Output('stats-analisis', 'children'),
     Output('airport-info-analisis', 'children')],
    [Input('layer-type-analisis', 'value'),
     Input('layer-select-analisis', 'value'),
     Input('airport-select-analisis', 'value'),
     Input('map-style-analisis', 'value')]
)
def update_analisis(layer_type, layer_id, airport_code, map_style):
    # Seleccionar grafo
    if layer_type == 'aggregated':
        G = G_agg
    else:
        G = layer_graphs[layer_id]
    
    # Encontrar índice del aeropuerto
    airport_idx = nodes[nodes['nodeLabel'] == airport_code].index[0]
    
    # Estadísticas del grafo completo
    stats_text = (f"Nodos totales: {G.number_of_nodes()}\n"
                  f"Conexiones totales: {G.number_of_edges()}\n"
                  f"Densidad: {nx.density(G):.4f}\n"
                  f"Componentes conectados: {nx.number_connected_components(G)}")
    
    # Info del aeropuerto
    if airport_idx in G.nodes():
        airport_info = (f"Código: {airport_code}\n"
                       f"Nombre: {G.nodes[airport_idx]['airport_name']}\n"
                       f"Ciudad: {G.nodes[airport_idx]['city']}\n"
                       f"Conexiones directas: {G.degree(airport_idx)}")
        
        # Crear subgrafo con vecinos
        neighbors = list(G.neighbors(airport_idx))
        subgraph_nodes = [airport_idx] + neighbors
        G_sub = G.subgraph(subgraph_nodes).copy()
        
        # Crear mapa
        edges_list = list(G_sub.edges())
        fig = create_map_figure(
            G_sub,
            nodes_subset=subgraph_nodes,
            edges_to_draw=edges_list,
            origin_node=airport_idx,
            mapbox_style=map_style
        )
    else:
        airport_info = "Aeropuerto no encontrado en esta red"
        fig = create_map_figure(G, mapbox_style=map_style)
    
    return fig, stats_text, airport_info


# Callback para Conexiones
@app.callback(
    [Output('map-conexiones', 'figure'),
     Output('path-info-conexiones', 'children')],
    [Input('layer-type-conexiones', 'value'),
     Input('layer-select-conexiones', 'value'),
     Input('airport-origin-conexiones', 'value'),
     Input('airport-dest-conexiones', 'value'),
     Input('map-style-conexiones', 'value')]
)
def update_conexiones(layer_type, layer_id, origin_code, dest_code, map_style):
    # Seleccionar grafo
    if layer_type == 'aggregated':
        G = G_agg
    else:
        G = layer_graphs[layer_id]
    
    # Encontrar índices
    origin_idx = nodes[nodes['nodeLabel'] == origin_code].index[0]
    dest_idx = nodes[nodes['nodeLabel'] == dest_code].index[0]
    
    # Calcular camino más corto
    try:
        path = nx.shortest_path(G, origin_idx, dest_idx)
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Obtener capas usadas
        path_layers = get_path_layers(path, edges)
        
        # Info del trayecto
        airport_names = [G.nodes[n]['airport_name'] for n in path]
        path_info = [
            html.Div([
                html.P(" → ".join(airport_names), style={'fontWeight': 'bold'}),
                html.Hr()
            ])
        ]
        
        for i, layer in enumerate(path_layers):
            if layer is not None:
                layer_name = layers.iloc[layer-1]['nodeLabel']
                path_info.append(
                    html.P(f"Trayecto {i+1}: {layer_name}")
                )
        
        # Crear mapa
        fig = create_map_figure(
            G,
            nodes_subset=path,
            path_edges=path_edges,
            origin_node=origin_idx,
            dest_node=dest_idx,
            mapbox_style=map_style
        )
        
    except nx.NetworkXNoPath:
        path_info = [html.P("No hay ruta disponible entre estos aeropuertos")]
        fig = create_map_figure(
            G,
            nodes_subset=[origin_idx, dest_idx],
            origin_node=origin_idx,
            dest_node=dest_idx,
            mapbox_style=map_style
        )
    
    return fig, path_info


# Callback para PageRank
@app.callback(
    [Output('map-pagerank', 'figure'),
     Output('stats-pagerank', 'children')],
    [Input('layer-select-pagerank', 'value'),
     Input('map-style-pagerank', 'value')]
)
def update_pagerank(layer_id, map_style):
    # Obtener grafo y matriz de adyacencia
    G = layer_graphs[layer_id]
    adj_matrix = adjacency_matrices[layer_id]
    
    # Filtrar nodos sin conexiones
    nodes_with_edges = [n for n in G.nodes() if G.degree(n) > 0]
    G_sub = G.subgraph(nodes_with_edges).copy()
    
    # Calcular PageRank
    pr = calculate_pagerank(adj_matrix)
    pr_filtered = pr[nodes_with_edges]
    
    # Estadísticas
    stats_text = (f"Nodos con conexiones: {len(nodes_with_edges)}\n"
                  f"Conexiones totales: {G_sub.number_of_edges()}\n"
                  f"PageRank promedio: {pr_filtered.mean():.5f}\n"
                  f"PageRank máximo: {pr_filtered.max():.5f}\n"
                  f"Nodo más importante: {G_sub.nodes[nodes_with_edges[np.argmax(pr_filtered)]]['label']}")
    
    # Crear mapa
    fig = create_map_figure(
        G_sub,
        nodes_subset=nodes_with_edges,
        edges_to_draw=list(G_sub.edges()),
        pagerank_values=pr,
        mapbox_style=map_style
    )
    
    return fig, stats_text


# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
