import dash
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
import numpy as np
import json
from pathlib import Path
import torch

# ------------------- PATHS (set these as needed for your interface) -------------------
ATTR_GRAPH_PATH = "/home/fdraye/projects/featflow/example/graph/save/attribution_graph.pt"  # <-- User sets this
DICT_BASE_FOLDER = "/fast/fdraye/data/featflow/cache/dict"           # <-- User sets this

# Configuration
N_layers = 4
d_latent = 49152
prompt_length = 5

def load_attribution_graph(graph_path):
    data = torch.load(graph_path, map_location="cpu")
    adjacency = data["sparse_pruned_adj"].to_dense().numpy()
    feature_indices = data["feature_indices"].cpu().numpy()  # shape: [n_features, 3]
    input_tokens = data["input_tokens"].cpu().numpy() 
    input_str = data["input_string"]
    return adjacency, feature_indices, input_tokens, input_str

def load_feature_dict(layer, feat_idx):
    dict_path = Path(DICT_BASE_FOLDER) / f"layer{layer}" / f"feature_{feat_idx}_complete.json"
    if dict_path.exists():
        with open(dict_path, "r") as f:
            return json.load(f)
    return None

# Load real data (for now, static; in a real interface, reload on user input)
adjacency, feature_indices, tokens, input_str = load_attribution_graph(ATTR_GRAPH_PATH)
input_str = input_str.split()
active_mask = (np.absolute(adjacency).sum(axis=1)) > 0
# Prune feature_indices and adjacency accordingly
feature_indices = feature_indices[active_mask]
feature_indices[:, 0] = feature_indices[:, 0] - 2  # Subtract 1 from ctx_pos (the first column)
adjacency = adjacency[active_mask][:, np.append(active_mask, True)]  # Keep logit column

print(np.diag(adjacency))

print(feature_indices)

input_str = ["Once", "upon", "a", "time", "there", "Il", "était", "une", "fois", "Es", "war", "einmal"]

# Initialize the Dash app
app = dash.Dash(__name__)


def create_active_feature_graph():
    feature_nodes = []
    feature_edges = []

    # Build lookup for (layer, pos): list of feature_idx
    layer_position_features = {}
    for i, (ctx_pos, layer, feat_idx) in enumerate(feature_indices):
        key = (layer, ctx_pos)
        if key not in layer_position_features:
            layer_position_features[key] = []
        layer_position_features[key].append((i, feat_idx))  # i is node_id

    # Fixed column width
    column_width = 5
    frame_width = 21  # Total width of the graph
    last_column_extra_space = 4

    # Calculate token x-axis positions (center of each column)
    token_x_positions = [pos * column_width - frame_width for pos in range(prompt_length - 1)]
    token_x_positions.append((prompt_length - 1) * column_width - frame_width + last_column_extra_space)

    # Place nodes strictly within columns
    for layer in range(N_layers):
        y = layer * 6 + 2  # Vertical position for the layer
        for pos in range(prompt_length):
            features = layer_position_features.get((layer, pos), [])
            n_feat = len(features)
            token_x = token_x_positions[pos]  # Center of the column for this token

            if n_feat == 1:
                # Single node centered around the token's x-axis position
                node_xs = [token_x]
            if pos == prompt_length - 1: 
                # Multiple nodes spaced evenly around the token's x-axis position
                spacing = min(1.5, ((column_width + 7) * 0.9) / max(n_feat - 1, 1))  # Adjust spacing to fit within column
                total_width = spacing * (n_feat - 1)
                node_xs = [token_x - total_width / 2 + i * spacing for i in range(n_feat)]
            else:
                # Multiple nodes spaced evenly around the token's x-axis position
                spacing = min(1.5, (column_width * 0.9) / max(n_feat - 1, 1))  # Adjust spacing to fit within column
                total_width = spacing * (n_feat - 1)
                node_xs = [token_x - total_width / 2 + i * spacing for i in range(n_feat)]

            # Ensure all nodes are strictly within the column boundaries
            col_left = token_x - (column_width / 2)
            if pos == prompt_length - 1:  # Adjust right boundary for the last column
                col_left = token_x - (column_width / 2) -last_column_extra_space
                col_right = token_x + (column_width / 2) + last_column_extra_space
            else:
                col_right = token_x + (column_width / 2)
            node_xs = [max(col_left + 0.5, min(x, col_right - 0.5)) for x in node_xs]

            for i, (node_id, feature_idx) in enumerate(features):
                feature_config = load_feature_dict(layer, feature_idx)
                feature_desc = feature_config["description"] 
                token_str = input_str[pos] if pos < len(input_str) else f"Token{pos}"
                feature_nodes.append({
                    "id": node_id,
                    "x": node_xs[i],
                    "y": y,
                    "layer": layer,
                    "pos": pos,
                    "feature_idx": feature_idx,
                    "token": token_str,
                    "description": feature_desc,
                    "config": feature_config,
                    "node_id_original": node_id
                })

    # Add the last logit node explicitly
    logit_node_id = len(feature_nodes)  # Assign a unique ID for the logit node
    logit_x = token_x_positions[-1]  # Place it in the last column
    logit_y = N_layers * 6 + 1  # Place it below the last layer
    feature_nodes.append({
        "id": logit_node_id,
        "x": logit_x,
        "y": logit_y,
        "layer": "logit",
        "pos": "logit",
        "feature_idx": "logit",
        "token": "logit",
        "description": "logit",  # Bold and descriptive text
        "config": None,
        "node_id_original": logit_node_id
    })

    # Create edges
    for i, node_i in enumerate(feature_nodes):
        for j, node_j in enumerate(feature_nodes):
            if i < j:
                weight = adjacency[node_i["node_id_original"], node_j["node_id_original"]]
                if weight != 0.0:
                    feature_edges.append({
                        "from": i,
                        "to": j,
                        "weight": weight
                    })

    return feature_nodes, feature_edges, token_x_positions

def create_plotly_active_graph(selected_feature_id=None):
    feature_nodes, feature_edges, token_x_positions = create_active_feature_graph()
    
    # Extract coordinates
    node_x = [node["x"] for node in feature_nodes]
    node_y = [node["y"] for node in feature_nodes]
    
    # Create grid traces
    grid_traces = []
    
    # Vertical lines separating tokens
    for pos in range(prompt_length):            
        x_pos = pos * 5 - 23.5  # Shifted left to match content

        grid_traces.append(go.Scatter(
            x=[x_pos, x_pos],
            y=[-6, N_layers * 6 + 2],  # Extended range for taller boxes
            mode='lines',
            line=dict(width=0.5, color='rgba(148, 163, 184, 0.3)'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Horizontal lines separating layers (taller boxes)
    for layer in range(N_layers + 1):
        y_pos = layer * 6 - 2  # Increased spacing from 4 to 6
        grid_traces.append(go.Scatter(
            x=[-26, 12],  # Extended range for layer indicators
            y=[y_pos, y_pos],
            mode='lines',
            line=dict(width=0.5, color='rgba(148, 163, 184, 0.2)'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add layer indicators with "L" prefix (centered in taller boxes)
    for layer in range(N_layers):
        grid_traces.append(go.Scatter(
            x=[-24.5],  # Positioned to the left
            y=[layer * 6 + 1],  # Centered in the taller boxes
            mode='text',
            text=[f"L{layer}"],
            textfont=dict(size=10, color='rgba(100, 116, 139, 0.6)', family="Inter, Arial"),
            textposition="middle center",
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Create edge traces with directional colors
    edge_traces = []
    incoming_edges = set()
    outgoing_edges = set()
    
    if selected_feature_id is not None:
        # Find incoming and outgoing edges for selected feature
        for edge in feature_edges:
            if edge["to"] == selected_feature_id:
                incoming_edges.add((edge["from"], edge["to"]))
            elif edge["from"] == selected_feature_id:
                outgoing_edges.add((edge["from"], edge["to"]))
    
    for edge in feature_edges:
        x0, y0 = feature_nodes[edge["from"]]["x"], feature_nodes[edge["from"]]["y"]
        x1, y1 = feature_nodes[edge["to"]]["x"], feature_nodes[edge["to"]]["y"]
        
        edge_key = (edge["from"], edge["to"])
        
        # Determine edge color and width - made edges lighter
        if edge_key in incoming_edges:
            edge_color = 'rgba(34, 197, 94, 0.4)'  # Light green for incoming - reduced from 0.8 to 0.4
            edge_width = 3
        elif edge_key in outgoing_edges:
            edge_color = 'rgba(59, 130, 246, 0.4)'  # Light blue for outgoing - reduced from 0.8 to 0.4
            edge_width = 3
        else:
            edge_color = 'rgba(0, 0, 0, 0.05)'  # Black for normal edges - reduced from 0.2 to 0.1
            edge_width = 1.5
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge_width,
                color=edge_color
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create feature nodes - only thickness changes, no color changes
    highlighted_nodes = set()
    if selected_feature_id is not None:
        highlighted_nodes.add(selected_feature_id)
        # Add connected nodes
        for edge in feature_edges:
            if edge["from"] == selected_feature_id:
                highlighted_nodes.add(edge["to"])
            elif edge["to"] == selected_feature_id:
                highlighted_nodes.add(edge["from"])
    
    node_colors = []
    node_sizes = []
    node_borders = []
    node_border_widths = []
    
    for i, node in enumerate(feature_nodes):
        # All nodes have same white center and black border
        node_colors.append('#ffffff')  # White center for all
        node_borders.append('#000000')  # Black border for all
        
        if i == selected_feature_id:
            node_sizes.append(28)  # Medium size for selected
            node_border_widths.append(3)  # Medium border for selected
        elif i in highlighted_nodes:
            node_sizes.append(28)  # Medium size for connected
            node_border_widths.append(3)  # Medium border for connected
        else:
            node_sizes.append(24)  # Normal size
            node_border_widths.append(2)  # Normal border
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=node_border_widths, color=node_borders),
            symbol='circle',
            opacity=0.95
        ),
        customdata=[i for i in range(len(feature_nodes))],  # Add this back - needed for clicks
        hovertemplate='<b>%{text}</b><extra></extra>',  # Removed the Layer %{customdata} part
        text=[f"{node['description']}" for node in feature_nodes],  # Removed the (L{node['layer']}) part
        showlegend=False
    )    
    
    # Add arrow and "next token word" text next to the logit node
    logit_node = feature_nodes[-1]  # Assuming the last node is the logit node

    # Create arrow with arrowhead using annotations
    arrow_trace = go.Scatter(
        x=[logit_node["x"] + 1, logit_node["x"] + 2],
        y=[logit_node["y"], logit_node["y"]],
        mode='lines',
        line=dict(width=3, color='black'),
        hoverinfo='none',
        showlegend=False
    )

    # Add arrowhead using annotation
    arrow_annotation = dict(
        x=logit_node["x"] + 2,
        y=logit_node["y"],
        ax=logit_node["x"] + 1,
        ay=logit_node["y"],
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='black',
        showarrow=True
    )

    text_trace = go.Scatter(
        x=[logit_node["x"] + 5.3],
        y=[logit_node["y"]],
        mode='text',
        text=["<b>LOGIT 'WAS'</b>"],
        textfont=dict(size=16, color='black', family="Inter, Arial"),
        textposition="middle left",
        hoverinfo='none',
        showlegend=False
    )
    # Create feature labels
    label_traces = []
    for node in feature_nodes:
        # Split description by words for better formatting
        description = node["description"]
        max_chars_per_line = 9
        
        words = description.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " + word if current_line else word)
            
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Create text traces for each line
        for i, line in enumerate(lines):
            line_y_offset = 1.3 + (i * 0.5)
            label_trace = go.Scatter(
                x=[node["x"]],
                y=[node["y"] - line_y_offset],
                mode='text',
                text=[line],
                textfont=dict(size=9, color='#374151', family="Inter, Arial"),
                textposition="middle center",
                hoverinfo='none',
                showlegend=False
            )
            label_traces.append(label_trace)
        
    
    # Create token labels (at the bottom of the graph)
    token_labels = []
    for i in range(prompt_length):
        if i < len(input_str):
            token = input_str[i]
            x_pos = token_x_positions[i]  # Use the same x positions as your nodes
            
            token_labels.append(go.Scatter(
                x=[x_pos],
                y=[-4.5],  # Position below the graph
                mode='text',
                text=["<b>" + token + "<b>"],
                textfont=dict(size=16, color='black', family="Inter, Arial"),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Then include token_labels in your fig_data:
    fig_data = grid_traces + edge_traces + [node_trace] + label_traces + token_labels + [arrow_trace, text_trace]

    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=5, r=5, t=10),
            annotations=[arrow_annotation],  # Add this line
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-28, 10],
                fixedrange=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-6.5, N_layers * 6 + 2.5],
                fixedrange=True
            ),
            height=850,
            plot_bgcolor='rgba(252, 248, 243, 1)',
            paper_bgcolor='rgba(252, 248, 243, 1)',
            clickmode='event+select',
            font=dict(family="Inter, Arial, sans-serif"),
            dragmode=False
        )    
    )    
    return fig

def create_activation_display(feature_config, feature_info):
    if not feature_config or 'top_20_examples' not in feature_config:
        return html.Div(
            [
                html.Div("🎯 Feature Analysis", style={
                    'fontSize': '20px',
                    'fontWeight': '600',
                    'color': '#1e293b',
                    'marginBottom': '10px'
                }),
                html.Div("Click on a feature node to see activation examples", style={
                    'color': '#64748b',
                    'fontSize': '14px'
                })
            ],
            style={
                'textAlign': 'center',
                'padding': '40px',
                'backgroundColor': '#fcf8f3',
                'borderRadius': '12px',
                'margin': '20px',
                'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
            }
        )
    
    # Create all examples in one continuous list
    all_examples = []
    
    for sentence in feature_config['top_20_examples']:
        words = sentence.split()
        
        colored_words = []
        for word in words:
            if word.startswith("<<") and word.endswith(">>"):
                # Highlight words between << >>
                clean_word = word[2:-2]  # Remove << and >>
                colored_words.append(
                    html.Span(clean_word, style={
                        'backgroundColor': '#fef3c7',  # Light yellow
                        'color': '#92400e',  # Dark brown text
                        'padding': '3px 6px',  # Small padding for boxes
                        'margin': '2px',
                        'borderRadius': '4px',  # Smaller border radius
                        'fontWeight': '500',
                        'display': 'inline-block',
                        'fontSize': '14px',
                        'fontFamily': 'Inter, system-ui, sans-serif',
                        'border': '1px solid #fef3c7'  # Subtle border
                    })
                )
            else:
                # Regular words without highlighting
                colored_words.append(
                    html.Span(word, style={
                        'backgroundColor': '#f8fafc',  # Light gray background
                        'color': '#64748b',  # Gray text
                        'padding': '3px 6px',
                        'margin': '2px',
                        'borderRadius': '4px',
                        'display': 'inline-block',
                        'fontSize': '14px',
                        'fontFamily': 'Inter, system-ui, sans-serif',
                        'border': '1px solid #f1f5f9'
                    })
                )
        
        # Add each sentence as a simple div with minimal spacing
        all_examples.append(
            html.Div(colored_words, style={
                'lineHeight': '1.8',
                'marginBottom': '8px'  # Small spacing between sentences
            })
        )
    
    return html.Div([
        html.Div([
            html.H3([
                html.Span("📊 ", style={'marginRight': '8px'}),
                feature_config['description']
            ], style={
                'margin': '0 0 12px 0',
                'fontSize': '18px',
                'fontWeight': '600',
                'color': '#1f2937',
                'fontFamily': 'Inter, system-ui, sans-serif'
            }),
            html.Div([
                html.Span(f"Layer {feature_info['layer']}", style={
                    'backgroundColor': '#3b82f6',
                    'color': 'white',
                    'padding': '4px 8px',
                    'borderRadius': '12px',
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'marginRight': '6px'
                }),
                html.Span(f"Token: {feature_info['token']}", style={
                    'backgroundColor': '#e5e7eb',
                    'color': '#4b5563',
                    'padding': '4px 8px',
                    'borderRadius': '12px',
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'marginRight': '6px'
                }),
                html.Span(f"Feature #{feature_info['feature_idx']}", style={
                    'backgroundColor': '#e5e7eb',
                    'color': '#4b5563',
                    'padding': '4px 8px',
                    'borderRadius': '12px',
                    'fontSize': '11px',
                    'fontWeight': '500'
                })
            ], style={'marginBottom': '16px'})
        ]),
        # All examples in one scrollable container
        html.Div(all_examples, style={
            'maxHeight': '300px',  # Limit height for many examples
            'overflowY': 'auto',   # Add scroll if needed
            'padding': '12px',
            'backgroundColor': 'white',
            'borderRadius': '6px',
            'border': '1px solid #e5e7eb'
        })
    ], style={
        'padding': '20px',
        'backgroundColor': '#fcf8f3',
        'borderRadius': '8px',
        'margin': '20px',
        'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
        'fontFamily': 'Inter, system-ui, sans-serif',
        'border': '1px solid #e5e7eb',
        'fontSize': '14px'
    })

app.layout = html.Div([
    html.Div([
        html.Div([
            # Max Planck logo on the left
            html.Div([
                html.Img(
                    src="/assets/max_planck_logo.jpg",
                    style={
                        'height': '40px',
                        'width': 'auto',
                        'marginRight': '16px'
                    }
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center'
            }),
            
            # Title
            html.Div([
                html.H1("Causal Feature Graph", style={
                    'fontSize': '28px',
                    'fontWeight': '700',
                    'color': '#1e293b',
                    'margin': '0',
                    'fontFamily': 'Inter, Arial'
                }),
            ], style={
                'flex': '1'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '24px',
            'backgroundColor': 'white',
            'borderBottom': '1px solid #e2e8f0'
        })
    ]),
    
    # Wrapped graph in white box container
    # Wrapped graph in beige box container
    html.Div([
        dcc.Graph(
            id='active-feature-graph',
            figure=create_plotly_active_graph(),
            style={'height': '850px', 'backgroundColor': '#fcf8f3'},  # Match the figure height
            config={
                'scrollZoom': False,
                'displayModeBar': False
            }
        )
    ], style={
        'padding': '24px',  # Match activation display padding
        'backgroundColor': '#fcf8f3',  # Changed to beige for the box container
        'borderRadius': '12px',  # Match activation display border radius
        'margin': '20px',  # Match activation display margin (changed from 8px)
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'fontFamily': 'Inter, Arial, sans-serif',
        'border': '1px solid #e2e8f0'
    }),

    html.Div(id='activation-display'),
    dcc.Store(id='selected-feature-store')
], style={
    'backgroundColor': 'white',  # Changed to white for main background
    'minHeight': '100vh',
    'fontFamily': 'Inter, Arial, sans-serif'
})

@callback(
    [Output('active-feature-graph', 'figure'),
     Output('activation-display', 'children'),
     Output('selected-feature-store', 'data')],
    [Input('active-feature-graph', 'clickData')],
    prevent_initial_call=True
)
def update_on_click(clickData):
    if not clickData or 'points' not in clickData:
        return create_plotly_active_graph(), create_activation_display(None, None), None
    
    point = clickData['points'][0]
    if 'customdata' not in point:
        return create_plotly_active_graph(), create_activation_display(None, None), None
    
    feature_idx = point['customdata']
    feature_nodes, _, _= create_active_feature_graph()
    
    if feature_idx < len(feature_nodes):
        selected_feature = feature_nodes[feature_idx]
        feature_config = selected_feature['config']
        
        # Update graph with highlights
        updated_fig = create_plotly_active_graph(selected_feature_id=feature_idx)
        display = create_activation_display(feature_config, selected_feature)
        
        return updated_fig, display, feature_idx
    
    return create_plotly_active_graph(), create_activation_display(None, None), None

if __name__ == '__main__':
    app.run(debug=True, port=8105, host='0.0.0.0')

# on the local machine: 
# ssh -N -L 8083:localhost:8083 fdraye@login.cluster.is.localnet


#######################

# # --------------------------------- Real APP -----------------------------------

# # Generate active features automatically
# active_features = generate_active_features(N_layers, prompt_length, d_latent)

# # Token information
# tokens = ["The", "capital", "of", "Texas", "is"]

# # Create feature lookup from features.py
# feature_lookup = {}
# for config in feature_configs:
#     feature_lookup[config["ID"]] = config

# # Initialize the Dash app
# app = dash.Dash(__name__)

# def create_active_feature_graph():
#     A = generate_adjacency_matrix(N_layers, d_latent, prompt_length)
    
#     # Group features by layer and position
#     layer_position_features = {}
#     for layer, pos, feature_idx in active_features:
#         key = (layer, pos)
#         if key not in layer_position_features:
#             layer_position_features[key] = []
#         layer_position_features[key].append(feature_idx)
    
#     # Calculate column widths based on max features per position in each layer
#     layer_max_features = {}
#     for layer in range(N_layers):
#         max_features = 0
#         for pos in range(prompt_length):
#             if (layer, pos) in layer_position_features:
#                 max_features = max(max_features, len(layer_position_features[(layer, pos)]))
#         layer_max_features[layer] = max(max_features, 1)
    
#     # Create feature nodes
#     feature_nodes = []
#     feature_edges = []
    
#     # Position calculation - increased vertical spacing and centered nodes
#     for layer in range(N_layers):
#         y = layer * 6 + 2  # Increased from 4 to 6 and added +1 to center in boxes
        
#         for pos in range(prompt_length):
#             if (layer, pos) in layer_position_features:
#                 features = layer_position_features[(layer, pos)]
                
#                 # Calculate horizontal positions for features of this token
#                 start_x = pos * 6 - 21  # Shifted left for better positioning
                
#                 if len(features) == 1:
#                     # Single feature centered
#                     x_positions = [start_x]
#                 else:
#                     # Multiple features spread horizontally with increased spacing
#                     spacing = 1.5
#                     x_positions = [start_x + (i - (len(features)-1)/2) * spacing for i in range(len(features))]
                
#                 # Create feature nodes
#                 for i, feature_idx in enumerate(features):
#                     feature_config = feature_lookup.get((layer, feature_idx))
#                     feature_desc = feature_config["Description"] if feature_config else f"Feature {feature_idx}"
                    
#                     node_id = len(feature_nodes)
#                     feature_nodes.append({
#                         "id": node_id,
#                         "x": x_positions[i],
#                         "y": y,
#                         "layer": layer,
#                         "pos": pos,
#                         "feature_idx": feature_idx,
#                         "token": tokens[pos] if pos < len(tokens) else f"Token{pos}",
#                         "description": feature_desc,
#                         "config": feature_config,
#                         "node_id_original": layer * prompt_length * d_latent + pos * d_latent + feature_idx
#                     })
    
#     # Create edges between features using adjacency matrix
#     for i, node_i in enumerate(feature_nodes):
#         for j, node_j in enumerate(feature_nodes):
#             if i != j:
#                 weight = A[node_i["node_id_original"], node_j["node_id_original"]]
#                 if weight > 0.08:  # Threshold for showing edges
#                     feature_edges.append({
#                         "from": i,
#                         "to": j,
#                         "weight": weight
#                     })
    
#     return feature_nodes, feature_edges, layer_max_features
