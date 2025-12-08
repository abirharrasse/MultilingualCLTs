# Frontend CLAUDE.md

This file provides guidance for working with the FeatFlow frontend visualization dashboard.

## Frontend Architecture

The frontend is a Dash-based web application for interactive visualization of causal attribution graphs. It displays feature relationships, cluster analysis, and intervention results.

### Core Components

**Main Application** (`app.py`):
- Dash app initialization and layout setup
- Integrates all visualization components

**Data Layer** (`data/`):
- `loaders.py`: Loads attribution graphs, clustering data, and intervention results
- `models.py`: Data models for graph nodes, edges, and metadata

**Visualization Layer** (`visualization/`):
- `graph/`: Core graph rendering (nodes, edges, layout, renderer)
- `components/`: Reusable UI components (cluster manager, feature display, etc.)

**Callbacks** (`callbacks/`):
- `graph_callbacks.py`: Interactive graph updates and filtering
- `cluster_callbacks.py`: Cluster analysis and intervention controls
- `annotation_callbacks.py`: Feature annotation and explanation handling

### Key Data Flow

1. **Graph Loading**: `data/loaders.py` loads `attribution_graph.pt` files containing:
   - `sparse_pruned_adj`: Pruned adjacency matrix from attribution analysis
   - Feature indices, token information, and intervention results

2. **Node Processing**: Active nodes determined by adjacency matrix row sums
3. **Edge Processing**: Edges filtered by weight thresholds and active nodes
4. **Layout**: Networkx-based graph layout with clustering support
5. **Rendering**: Plotly-based interactive visualization

### Common Issues

**Nodes with No Outgoing Edges**: 
- Attribution graph correctly prunes these nodes via `feature_mask`
- Frontend should use saved `feature_mask` instead of recalculating from adjacency
- Check `data/loaders.py` lines 104-105 for active mask calculation

**Graph Filtering**:
- Multiple threshold controls: feature influence, edge weights, clustering
- Ensure consistent filtering across all visualization components

### Configuration

**Settings** (`config/settings.py`):
- Graph layout parameters, color schemes, filtering thresholds
- Cluster analysis parameters and intervention settings

### Development Commands

Launch frontend:
```bash
python src/featflow/frontend/launch.py
```

Debug mode:
```bash
python src/featflow/frontend/app.py --debug
```

### Data Requirements

Expected data structure in `attribution_graph.pt`:
- `sparse_pruned_adj`: Pruned adjacency matrix 
- `feature_indices`: Feature location indices
- `input_tokens`, `token_string`: Input text data
- `top_logit_*`: Prediction information
- Optional: `feature_mask` for proper node filtering
