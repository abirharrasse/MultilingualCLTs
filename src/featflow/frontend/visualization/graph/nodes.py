import plotly.graph_objects as go
from typing import List, Set, Optional

from ...data.models import FeatureNode
from ...config.settings import GraphConfig

class NodeRenderer:
    """Handles rendering of graph nodes."""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        # Node sizes
        self.normal_node_size = 10
        self.selected_node_size = 14
        self.highlighted_node_size = 12
        self.description_node_size = 10  # For single-clicked nodes with descriptions
        
        # Get colors with fallbacks
        self.normal_node_color = getattr(config, 'normal_node_color', '#f8f9fa')  # Light gray/white fill
        self.selected_node_color = getattr(config, 'selected_node_color', '#EF4444')
        self.highlighted_node_color = getattr(config, 'highlighted_node_color', '#F59E0B')
        self.description_node_color = '#10B981'  # Green for nodes with descriptions
        self.intersection_node_color = '#DC2626'  # Red for intersection nodes
    
    def create_node_trace(self, nodes: List[FeatureNode], 
                         selected_feature_id: Optional[int] = None,
                         highlighted_nodes: Set[int] = None,
                         nodes_with_descriptions: Set[int] = None,
                         node_to_cluster: dict = None,
                         cluster_highlighted_nodes: Set[int] = None,
                         intersection_nodes: Set[int] = None) -> go.Scatter:
        """Create the main node trace with different states and cluster colors."""
        if not nodes:
            return go.Scatter(x=[], y=[], mode='markers')
        
        highlighted_nodes = highlighted_nodes or set()
        nodes_with_descriptions = nodes_with_descriptions or set()
        node_to_cluster = node_to_cluster or {}
        cluster_highlighted_nodes = cluster_highlighted_nodes or set()
        intersection_nodes = intersection_nodes or set()
        
        # Extract coordinates and properties
        x_coords = [node.x for node in nodes]
        y_coords = [node.y for node in nodes]
        
        # Calculate node sizes and colors
        sizes = []
        colors = []
        
        for i, node in enumerate(nodes):
            if i == selected_feature_id:
                # Double-clicked node (red, largest)
                sizes.append(self.selected_node_size)
                colors.append(self.selected_node_color)
            elif i in intersection_nodes:
                # Intersection node (red, but smaller than selected)
                sizes.append(self.highlighted_node_size)
                colors.append(self.intersection_node_color)
            elif i in highlighted_nodes:
                # Connected to double-clicked node (orange)
                sizes.append(self.highlighted_node_size)
                colors.append(self.highlighted_node_color)
            elif i in nodes_with_descriptions:
                # Single-clicked node with description (green)
                sizes.append(self.description_node_size)
                colors.append(self.description_node_color)
            elif i in cluster_highlighted_nodes:
                # Node in selected cluster - use cluster color with larger size
                sizes.append(self.highlighted_node_size)
                cluster_color = node_to_cluster.get(i, self.highlighted_node_color)
                colors.append(cluster_color)
                # Node in cluster - use cluster color
            else:
                # Normal node - ALWAYS use default blue color, regardless of cluster membership
                sizes.append(self.normal_node_size)
                colors.append(self.normal_node_color)  # Always blue, never cluster color
            
        # Use the previously calculated colors as the base
        final_colors = colors
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=final_colors,
                line=dict(width=2, color='black'),  # Black borders
                opacity=1.0
            ),
            hoverinfo='none',
            showlegend=False
        )
    
    def create_node_labels(self, nodes: List[FeatureNode]) -> List[go.Scatter]:
        """Return empty list - no labels by default."""
        return []
    
    def get_node_style(self, node: FeatureNode, is_selected: bool = False, 
                      is_highlighted: bool = False, has_description: bool = False) -> dict:
        """Get styling for individual nodes."""
        if is_selected:
            return {
                'size': self.selected_node_size,
                'color': self.selected_node_color,
                'line_width': 2
            }
        elif is_highlighted:
            return {
                'size': self.highlighted_node_size,
                'color': self.highlighted_node_color,
                'line_width': 1.5
            }
        elif has_description:
            return {
                'size': self.description_node_size,
                'color': self.description_node_color,
                'line_width': 1.5
            }
        else:
            return {
                'size': self.normal_node_size,
                'color': self.normal_node_color,
                'line_width': 1
            }
