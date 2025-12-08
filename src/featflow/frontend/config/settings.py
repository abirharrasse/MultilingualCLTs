from dataclasses import dataclass, field
import os

@dataclass
class GraphConfig:
    """Configuration for graph layout and appearance."""
    min_tokens: int = 3
    max_tokens: int = 16
    max_layers: int = 12
    
    # Layout parameters
    base_column_width: float = 5.0
    layer_height: float = 6.0
    node_spacing: float = 1.5
    
    # Visual parameters
    normal_node_size: int = 24
    highlighted_node_size: int = 28
    normal_border_width: int = 2
    highlighted_border_width: int = 3
    
    # Grid parameters
    margin_left: float = 3.0
    margin_right: float = 2.0
    margin_top: float = 1.0
    margin_bottom: float = 2.0
    
    # Colors
    background_color: str = '#fcf8f3'
    node_color: str = '#ffffff'
    border_color: str = '#000000'
    
    # Node colors (add these missing attributes)
    normal_node_color: str = '#ffffff'      # White fill
    selected_node_color: str = '#EF4444'    # Red
    highlighted_node_color: str = '#F59E0B' # Orange
    
    # Edge colors
    incoming_edge_color: str = 'rgba(34, 197, 94, 0.4)'
    outgoing_edge_color: str = 'rgba(59, 130, 246, 0.4)'
    normal_edge_color: str = 'rgba(0, 0, 0, 0.05)'
    edge_color: str = '#6B7280'             # Gray
    selected_edge_color: str = '#EF4444'    # Red

@dataclass
class AppConfig:
    """Main application configuration."""
    attr_graph_path: str
    dict_base_folder: str
    clt_checkpoint: str
    model_name: str
    model_class_name: str
    
    # Server config
    host: str = "0.0.0.0"
    port: int = 8105
    debug: bool = False
    
    # Graph config
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        return cls(
            attr_graph_path=os.getenv(
                "ATTR_GRAPH_PATH", 
                "/home/abir19/FeatFlow/example/graph/llama_MT/en/attribution_graph.pt"
            ),
            dict_base_folder=os.getenv(
                "DICT_BASE_FOLDER",
               "/home/abir19/projects/autointerp-llama-multilingual/data" # "/home/abir19/scratch/autointerp-gpt2-multilingual-20/data" #"/home/abir19/projects/def-zhijing/abir19/autointerp-gpt2-multilingual-90/data"
            ),
            clt_checkpoint=os.getenv(
                "CLT_CHECKPOINT",
                "/home/abir19/projects/llama_multilingual_clt"
            ),
            model_name=os.getenv(
                "MODEL_NAME",
                "meta-llama/Llama-3.2-1B"
            ),
            model_class_name=os.getenv(
                "MODEL_CLASS_NAME",
                "HookedTransformer"
            ),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8100")),
            debug=os.getenv("DEBUG", "False").lower() == "true"
        )
