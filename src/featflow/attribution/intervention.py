import torch
from circuit_tracer import ReplacementModel
from featflow.attribution.loading import load_circuit_tracing_clt_from_local
from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from typing import List, Dict, Any
import os 

def run_cluster_intervention(
    cluster_features: list,
    manual_features: list,
    clt_checkpoint: str,
    input_string: str,
    cluster_intervention_value: float,
    model_name: str = "roneneldan/TinyStories-33M",
    top_tokens_count: int = 4,
    device: str = "cuda",
    freeze_attention: bool = True
): 

    patch_official_model_names()
    patch_convert_hf_model_config()

    clt = load_circuit_tracing_clt_from_local(clt_checkpoint, device=device)

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name = model_name, 
        transcoders = clt
    )

    features_to_intervene = []

    for pos, layer, feature_idx in cluster_features:
        features_to_intervene.append((layer, pos, feature_idx, cluster_intervention_value))
        
    for pos, layer, feature_idx, intervention_value in manual_features:
        features_to_intervene.append((layer, pos, feature_idx, intervention_value))

    input_tokens = model.ensure_tokenized(input_string) # adds a BOS to the sentence
    print(input_tokens)
    intervened_logits, _ = model.feature_intervention(input_tokens, interventions=features_to_intervene, freeze_attention=freeze_attention)
    original_logits = model(input_tokens)

    original_probs = torch.softmax(original_logits[0, -1], dim=-1)
    intervened_probs = torch.softmax(intervened_logits[0, -1], dim=-1)
    
    top_tokens = torch.topk(intervened_probs, top_tokens_count)
    top_token_strings = [model.tokenizer.decode([token_id]) for token_id in top_tokens.indices]
    
    baseline_probs = []
    prob_differences = []
    for token_id in top_tokens.indices:
        baseline_prob = original_probs[token_id].item()
        intervened_prob = intervened_probs[token_id].item()
        baseline_probs.append(baseline_prob)
        prob_differences.append(intervened_prob - baseline_prob)
    
    return {
        'tokens': top_token_strings,
        'probabilities': top_tokens.values.cpu().tolist(),
        'baseline_probabilities': baseline_probs,
        'probability_differences': prob_differences,
        'cluster_feature_count': len(cluster_features),
        'manual_feature_count': len(manual_features)
    }


def compute_intervention_top_tokens(
        folder_name: str,
        model: ReplacementModel, 
        input_string: str, 
        intervention_values: List[float] = [5.0, -5.0, -10.0],
        top_tokens_count: int = 4, 
        freeze_attention: bool = True
): 
    
    input_tokens = model.ensure_tokenized(input_string) # adds a BOS to the sentence
    print(f"Input tokens for intervention: {input_tokens}")

    # Load the saved graph data
    data = torch.load(os.path.join(folder_name, "attribution_graph.pt"))
    feature_indices = data["feature_indices"]
    feature_mask = data["feature_mask"]
    logit_probabilities = data.get("logit_probabilities", None)
    logit_tokens = data.get("logit_tokens", None)
    baseline_top_logit_token_prob = logit_probabilities.max().item()
    baseline_top_logit_token_idx = logit_tokens[logit_probabilities.argmax().item()]

    n_features = len(feature_indices)
    active_feature_indices = feature_indices[feature_mask[:n_features]]
    print(f"Intervening on {len(active_feature_indices)} features")

    intervention_top_tokens: List[Dict[str, Any]] = []

    for feature_data in active_feature_indices:
        layer, ctx_pos, feature_idx = int(feature_data[1]), int(feature_data[0]), int(feature_data[2])
        
        feature_results: Dict[str, Any] = {
            'feature_info': {'layer': layer, 'position': ctx_pos, 'feature_idx': feature_idx},
            'interventions': []
        }
        
        for intervention_value in intervention_values:
            features_to_intervene = [(layer, ctx_pos, feature_idx, intervention_value)]
            
            try:
                intervened_logits, _ = model.feature_intervention(input_tokens, interventions=features_to_intervene, freeze_attention=freeze_attention)
                # Get top tokens after intervention
                intervened_probs = torch.softmax(intervened_logits[0, -1], dim=-1)
                top_tokens = torch.topk(intervened_probs, top_tokens_count)
                
                top_token_strings = []
                for token_id in top_tokens.indices:
                    token_str = model.tokenizer.decode([token_id])
                    top_token_strings.append(token_str)
                
                # Check how much the baseline token probability changed
                baseline_token_new_prob = intervened_probs[baseline_top_logit_token_idx].item()
                prob_change = baseline_token_new_prob - baseline_top_logit_token_prob
                
                feature_results['interventions'].append({
                    'intervention_value': intervention_value,
                    'tokens': top_token_strings,
                    'probabilities': top_tokens.values.cpu().tolist(),
                    'baseline_token': baseline_top_logit_token_idx,
                    'baseline_prob_original': baseline_top_logit_token_prob,
                    'baseline_prob_after_intervention': baseline_token_new_prob,
                    'baseline_prob_change': prob_change
                })

            except Exception as e:
                print(f"Warning: Intervention failed for layer_{layer}_pos_{ctx_pos}_feat_{feature_idx}_val_{intervention_value}: {e}")
                # Add empty intervention result instead of None
                feature_results['interventions'].append({
                    'intervention_value': intervention_value,
                    'tokens': [],
                    'probabilities': [],
                    'baseline_token': '',
                    'baseline_prob_original': 0.0,
                    'baseline_prob_after_intervention': 0.0,
                    'baseline_prob_change': 0.0
                })
        
        intervention_top_tokens.append(feature_results)

    # Add results to saved graph data
    data["intervention_top_tokens"] = intervention_top_tokens
    torch.save(data, os.path.join(folder_name, "attribution_graph.pt"))
    
    return intervention_top_tokens