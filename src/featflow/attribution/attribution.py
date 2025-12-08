from circuit_tracer import ReplacementModel, attribute
  # Check where circuit_tracer is being imported from
import circuit_tracer
print(f"circuit_tracer location: {circuit_tracer.__file__}")
from circuit_tracer.graph import prune_graph
from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from featflow.attribution.loading import load_circuit_tracing_clt_from_local, test_clt_performance_on_prompt, compare_reconstruction_with_local_clt_class
from circuit_tracer.utils import create_graph_files
from featflow.attribution.intervention import compute_intervention_top_tokens
from pathlib import Path
import os
import torch
from typing import Optional

def run_attribution(
    folder_name: str,
    clt_checkpoint: str,
    input_string: str,
    model_name: str = "gpt2",
    max_n_logits: int = 10,
    desired_logit_prob = 0.95, 
    max_feature_nodes = 8192, 
    batch_size=256, 
    offload = "cpu",
    verbose = True,
    feature_threshold: float = 0.8,
    edge_threshold: float = 0.95,
    device: str = "cuda", 
    compute_interventions: bool = True, 
    attribution_vector: Optional[torch.Tensor] = None, 
    graph_name: str = "attribution_graph.pt", 
    save_path_cumulative_score: Optional[str] = None
):
    patch_official_model_names()
    patch_convert_hf_model_config()

    clt = load_circuit_tracing_clt_from_local(clt_checkpoint, device=device)

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name = model_name, 
        transcoders = clt
    )
            
    # just testing, can be commented out later
    test_clt_performance_on_prompt(input_string, clt, model)
    compare_reconstruction_with_local_clt_class(
        clt_checkpoint, 
        input_string, 
        clt, 
        model, 
        model_name
    )

    graph = attribute(
        prompt=input_string,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose, 
        attribution_vector=attribution_vector
    )

    prune_result = prune_graph(
        graph=graph,
        node_threshold=feature_threshold,
        edge_threshold=edge_threshold, 
        save_path_cumulative_score=save_path_cumulative_score
    )

    # apply mask to adjacency matrix 
    os.makedirs(folder_name, exist_ok=True)

    sparse_adjacency = prune_result.edge_mask.float()
    n_features = graph.active_features.shape[0]
    print(f"Number of features before pruning: {n_features}")
    print(f"Number of feature after pruning (not counting error nodes): {prune_result.node_mask[:n_features].sum().item()}")

    active_feature = torch.stack([graph.active_features[:,1], graph.active_features[:,0], graph.active_features[:,2]], dim=1) # to pos, layer, feature_idx
    token_string = [model.tokenizer.decode(t) for t in graph.input_tokens]
    logit_token_strings = [model.tokenizer.decode(t) for t in graph.logit_tokens]

    print(logit_token_strings)

    print("Token string: ", token_string)
    print("Corresponding tokens: ", graph.input_tokens)

    # Prepare dictionary
    result = {
        "adjacency_matrix": graph.adjacency_matrix.cpu(),
        "feature_indices": active_feature.cpu(),
        "sparse_pruned_adj": sparse_adjacency.cpu(),
        "feature_mask": prune_result.node_mask.cpu(),
        "edge_mask": prune_result.edge_mask.cpu(),
        "logit_tokens": graph.logit_tokens.cpu(),
        "logit_probabilities": graph.logit_probabilities.cpu(),
        "input_tokens": graph.input_tokens.cpu(),
        "input_string": input_string, 
        "token_string": token_string,
        "logit_token_strings": logit_token_strings,
        "full_adjacency_matrix": graph.adjacency_matrix.cpu(),  # ADDED THIS LINE
        "selected_features": graph.selected_features.cpu() if graph.selected_features is not None else None,
        "activation_values": graph.activation_values.cpu() if graph.activation_values is not None else None,
        "active_features": graph.active_features.cpu()

    }

    torch.save(result, os.path.join(folder_name, graph_name))
    print(f"Saved attribution graph to {os.path.join(folder_name, graph_name)}")

    if compute_interventions: 
        # adds to the graph data the intervention values on each activated feature
        compute_intervention_top_tokens(
            folder_name=folder_name,
            model=model, 
            input_string=input_string, 
            intervention_values=[0, -5.0, -10.],
            top_tokens_count=4
        )

    return result