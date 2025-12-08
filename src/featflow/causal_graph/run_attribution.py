import torch

from featflow.causal_graph.replacement_model import ReplacementModel
from featflow.causal_graph.attribution_graph import SimpleAttributionGraph

def run_attribution(
    folder_name: str,
    clt_checkpoint: str,
    input_string: str,
    model_class_name: str = "HookedTransformer",
    model_name: str = "roneneldan/TinyStories-33M",
    feature_threshold: float = 0.9, 
    edge_threshold: float = 0.95,
    device: str = "cuda",
    intervention_values: list = [5, -5.0, -10., -20., -40],
    compute_correlation: bool = True, 
    replace_logit_with_vector: torch.Tensor = None
):
    """        
    Creates a ReplacementModel, loads the relevant data, creates an AttributionGraph,
    and runs it on the given tokens.
    """

    replacement_model = ReplacementModel(
        model_class_name=model_class_name,
        model_name=model_name,
        clt_path=clt_checkpoint,
        device=torch.device(device)
    )
    
    attribution_data = replacement_model.data_for_attribution(input_string)
    
    attribution_graph = SimpleAttributionGraph(
        model=replacement_model.model,
        attribution_data=attribution_data    
    )

    output_dict = attribution_graph.compute_graph(folder_name, feature_threshold, edge_threshold, replace_logit_with_vector)

    adjacency = output_dict["prune_adjacency_matrix"] 
    top_logit_idx = output_dict["top_logit_idx"]
    top_logit_prob = output_dict["top_logit_prob"]
    tokens = output_dict["input_tokens"]

    if compute_correlation : 
        replacement_model.compute_feature_correlation_scores(
        "/home/abir19/FeatFlow/example/graph/save", 
        adjacency, 
        attribution_graph.feature_enc_indices, 
        "/home/abir19/scratch/autointerp-gpt2-multilingual-90/data"
        )
    
    # Compute intervention top tokens for each active feature
    intervention_top_tokens = replacement_model.compute_intervention_top_tokens(folder_name, tokens, intervention_values=intervention_values)
    
    output_dict['intervention_top_tokens'] = intervention_top_tokens
    output_dict['top_token'] = replacement_model.model.tokenizer.decode(top_logit_idx)

        
    return output_dict
