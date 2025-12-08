import torch
from featflow.causal_graph.replacement_model import ReplacementModel

def run_cluster_intervention(
    cluster_features: list,
    manual_features: list,
    clt_checkpoint: str,
    input_string: str,
    cluster_intervention_value: float,
    model_class_name: str = "HookedTransformer",
    model_name: str = "roneneldan/TinyStories-33M",
    top_tokens_count: int = 4,
    device: str = "cuda",
    freeze_attention: bool = False
):
    """
    Creates a ReplacementModel and runs dual intervention on cluster and manual features.
    
    Args:
        cluster_features: List of (ctx_pos, layer, feature_idx) tuples from cluster
        manual_features: List of (ctx_pos, layer, feature_idx, intervention_value) tuples 
        clt_checkpoint: Path to CLT checkpoint
        input_string: Input text to intervene on
        cluster_intervention_value: Intervention multiplier for cluster features
        
    Returns:
        Dict with intervention results
    """
    
    replacement_model = ReplacementModel(
        model_class_name=model_class_name,
        model_name=model_name,
        clt_path=clt_checkpoint,
        device=torch.device(device)
    )
    
    # Add BOS token only if not already present
    if not input_string.startswith(replacement_model.model.tokenizer.bos_token):
        input_string_with_BOS = replacement_model.model.tokenizer.bos_token + input_string
    else:
        input_string_with_BOS = input_string
    input_tokens = replacement_model.model.tokenizer.encode(input_string_with_BOS, return_tensors="pt")[0].to(torch.device(device))
    
    result = replacement_model.compute_cluster_intervention(
        cluster_features,
        manual_features, 
        input_tokens,
        cluster_intervention_value,
        top_tokens_count,
        freeze_attention
    )
    
    return result
