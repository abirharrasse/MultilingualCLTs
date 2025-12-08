import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import json
import pandas as pd
import numpy as np
from circuit_tracer import ReplacementModel
from featflow.attribution.loading import load_circuit_tracing_clt_from_local
from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config

def compute_feature_alignment_with_unembedding(
    model: 'ReplacementModel',
    layer: int,
    pos: int,
    feature_idx: int,
    target_token: str,
    input_context: str = "",
    activation_strength: float = 1.0
) -> Dict[str, float]:
    """
    Compute alignment between a feature and target token unembedding.
    
    Propagates the feature activation through the model from layer L 
    to unembedding and computes dot product with target token vector.
    
    Args:
        model: ReplacementModel instance
        layer: Layer where feature is located
        pos: Position index (for context-dependent analysis)
        feature_idx: Feature index in transcoder
        target_token: Token to compute alignment with
        input_context: Optional context string for position-aware analysis
        activation_strength: Strength of feature activation
        
    Returns:
        Dict with alignment metrics
    """
    
    # Get target token unembedding vector
    try:
        target_token_id = model.tokenizer.encode(target_token, add_special_tokens=False)[0]
    except:
        target_token_id = model.tokenizer.encode(f" {target_token}", add_special_tokens=False)[0]
    
    target_unembedding = model.W_U[:, target_token_id]  # Shape: (d_model,)
    
    # Get decoder vectors for this feature
    if hasattr(model.transcoders, '__len__'):  
        decoder_vectors = model.transcoders[layer].W_dec[feature_idx]  # Shape: (d_model,)
        n_remaining_layers = 1
    else:  
        decoder_vectors = model.transcoders._get_decoder_vectors(layer, torch.tensor([feature_idx]))
        if decoder_vectors.ndim == 3:  # Shape: (1, n_remaining_layers, d_model)
            decoder_vectors = decoder_vectors.squeeze(0)  # Shape: (n_remaining_layers, d_model)
            n_remaining_layers = decoder_vectors.shape[0]
        else:  # Single layer transcoder
            decoder_vectors = decoder_vectors.squeeze(0).unsqueeze(0)  # Shape: (1, d_model)
            n_remaining_layers = 1
    
    # Scale by activation strength
    if n_remaining_layers == 1:
        feature_outputs = decoder_vectors * activation_strength  # Shape: (d_model,)
        final_output = feature_outputs
    else:
        feature_outputs = decoder_vectors * activation_strength  # Shape: (n_remaining_layers, d_model)
        # Sum contributions across all layers the feature writes to
        final_output = feature_outputs.sum(dim=0)  # Shape: (d_model,)
    
    # For more accurate propagation, we could run through the actual model layers,
    # but for alignment analysis, direct computation is sufficient
    
    # Apply final layer norm approximation (using mean scaling)
    if hasattr(model, 'ln_final') and hasattr(model.ln_final, 'w'):
        ln_scale = model.ln_final.w.mean().item()
        final_output = final_output * ln_scale
    
    # Compute alignment metrics
    alignment_score = torch.dot(final_output, target_unembedding).item()
    
    # Compute cosine similarity
    final_norm = torch.norm(final_output).item()
    target_norm = torch.norm(target_unembedding).item()
    cosine_similarity = alignment_score / (final_norm * target_norm) if (final_norm * target_norm) > 0 else 0.0
    
    # Compute relative alignment (alignment relative to target vector magnitude)
    relative_alignment = alignment_score / target_norm if target_norm > 0 else 0.0
    
    return {
        'layer': layer,
        'pos': pos,
        'feature_idx': feature_idx,
        'target_token': target_token,
        'target_token_id': target_token_id,
        'alignment_score': alignment_score,
        'cosine_similarity': cosine_similarity,
        'relative_alignment': relative_alignment,
        'feature_output_norm': torch.norm(decoder_vectors).item() if n_remaining_layers == 1 else torch.norm(feature_outputs).item(),
        'final_output_norm': final_norm,
        'target_norm': target_norm,
        'n_remaining_layers': n_remaining_layers
    }

def analyze_cluster_alignment_comprehensive(
    model: 'ReplacementModel',
    cluster_features: List[Tuple[int, int, int]],  # (layer, pos, feature_idx)
    target_tokens: List[str],
    cluster_name: str = "unnamed_cluster",
    activation_strength: float = 1.0,
    input_context: str = ""
) -> Dict[str, any]:
    """
    Comprehensive alignment analysis for a cluster of features.
    
    Args:
        model: ReplacementModel instance
        cluster_features: List of (layer, pos, feature_idx) tuples
        target_tokens: List of tokens to compute alignment with
        cluster_name: Name for this cluster
        activation_strength: Strength of feature activations
        input_context: Optional context for position-aware analysis
        
    Returns:
        Dict containing detailed results and summary statistics
    """
    
    print(f"\n=== Analyzing cluster '{cluster_name}' ===")
    print(f"Features: {len(cluster_features)}")
    print(f"Target tokens: {target_tokens}")
    print(f"Activation strength: {activation_strength}")
    
    individual_results = []
    
    # Analyze each feature
    for i, (layer, pos, feature_idx) in enumerate(cluster_features):
        print(f"Processing feature {i+1}/{len(cluster_features)}: L{layer}F{feature_idx} at pos {pos}")
        
        for target_token in target_tokens:
            try:
                result = compute_feature_alignment_with_unembedding(
                    model, layer, pos, feature_idx, target_token, 
                    input_context, activation_strength
                )
                result['cluster_name'] = cluster_name
                individual_results.append(result)
                
                print(f"  {target_token}: align={result['alignment_score']:.4f}, "
                      f"cosine={result['cosine_similarity']:.4f}")
                
            except Exception as e:
                print(f"  Error with {target_token}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(individual_results)
    
    if df.empty:
        return {'cluster_name': cluster_name, 'results_df': df, 'summary': {}, 'error': 'No successful computations'}
    
    # Compute summary statistics
    summary_stats = {}
    
    for token in target_tokens:
        token_data = df[df['target_token'] == token]
        if not token_data.empty:
            summary_stats[token] = {
                'count': len(token_data),
                'mean_alignment': token_data['alignment_score'].mean(),
                'std_alignment': token_data['alignment_score'].std(),
                'max_alignment': token_data['alignment_score'].max(),
                'min_alignment': token_data['alignment_score'].min(),
                'mean_cosine': token_data['cosine_similarity'].mean(),
                'std_cosine': token_data['cosine_similarity'].std(),
                'max_cosine': token_data['cosine_similarity'].max(),
                'positive_alignments': (token_data['alignment_score'] > 0).sum(),
                'strong_alignments': (token_data['alignment_score'] > 0.1).sum(),  # Threshold for "strong"
                'very_strong_alignments': (token_data['alignment_score'] > 0.5).sum()
            }
    
    # Find best and worst aligned features for each token
    best_worst = {}
    for token in target_tokens:
        token_data = df[df['target_token'] == token]
        if not token_data.empty:
            best_idx = token_data['alignment_score'].idxmax()
            worst_idx = token_data['alignment_score'].idxmin()
            best_worst[token] = {
                'best_feature': {
                    'layer': token_data.loc[best_idx, 'layer'],
                    'feature_idx': token_data.loc[best_idx, 'feature_idx'],
                    'alignment': token_data.loc[best_idx, 'alignment_score'],
                    'cosine': token_data.loc[best_idx, 'cosine_similarity']
                },
                'worst_feature': {
                    'layer': token_data.loc[worst_idx, 'layer'],
                    'feature_idx': token_data.loc[worst_idx, 'feature_idx'],
                    'alignment': token_data.loc[worst_idx, 'alignment_score'],
                    'cosine': token_data.loc[worst_idx, 'cosine_similarity']
                }
            }
    
    # Print summary
    print(f"\n--- Summary for cluster '{cluster_name}' ---")
    for token, stats in summary_stats.items():
        print(f"{token}:")
        print(f"  Mean alignment: {stats['mean_alignment']:.4f} ± {stats['std_alignment']:.4f}")
        print(f"  Range: [{stats['min_alignment']:.4f}, {stats['max_alignment']:.4f}]")
        print(f"  Mean cosine: {stats['mean_cosine']:.4f}")
        print(f"  Positive alignments: {stats['positive_alignments']}/{stats['count']}")
        print(f"  Strong alignments (>0.1): {stats['strong_alignments']}")
        if token in best_worst:
            best = best_worst[token]['best_feature']
            print(f"  Best feature: L{best['layer']}F{best['feature_idx']} (align={best['alignment']:.4f})")
    
    return {
        'cluster_name': cluster_name,
        'results_df': df,
        'summary_stats': summary_stats,
        'best_worst_features': best_worst,
        'n_features': len(cluster_features),
        'n_tokens': len(target_tokens)
    }

def load_and_analyze_cluster_from_json(
    json_file: str,
    cluster_description: str,
    target_tokens: List[str],
    clt_checkpoint: str,
    model_name: str = "CausalNLP/gpt2-hf_multilingual-90",
    activation_strength: float = 1.0,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Load cluster from JSON file and perform comprehensive alignment analysis.
    
    Args:
        json_file: Path to JSON file with clusters
        cluster_description: Description of cluster to analyze
        target_tokens: List of tokens to compute alignment with
        clt_checkpoint: Path to CLT checkpoint
        model_name: Name of the model
        activation_strength: Strength of feature activations
        device: Device to run on
        
    Returns:
        Dict with comprehensive analysis results
    """
    
    # Setup model
    patch_official_model_names()
    patch_convert_hf_model_config()
    
    print(f"Loading model {model_name} and CLT from {clt_checkpoint}")
    clt = load_circuit_tracing_clt_from_local(clt_checkpoint, device=device)
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=model_name, 
        transcoders=clt
    )
    
    # Load cluster data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Find target cluster
    target_cluster = None
    cluster_id = None
    for cid, cluster_data in data['clusters'].items():
        if cluster_data['description'] == cluster_description:
            target_cluster = cluster_data
            cluster_id = cid
            break
    
    if target_cluster is None:
        available_clusters = [cluster_data['description'] for cluster_data in data['clusters'].values()]
        raise ValueError(f"Cluster '{cluster_description}' not found. Available clusters: {available_clusters}")
    
    # Extract features as (layer, pos, feature_idx) tuples
    cluster_features = []
    for node in target_cluster['nodes']:
        cluster_features.append((node['layer'], node['pos'], node['feature_idx']))
    
    print(f"Found cluster '{cluster_description}' with {len(cluster_features)} features")
    
    # Run analysis
    results = analyze_cluster_alignment_comprehensive(
        model=model,
        cluster_features=cluster_features,
        target_tokens=target_tokens,
        cluster_name=cluster_description,
        activation_strength=activation_strength
    )
    
    # Add model info to results
    results['model_name'] = model_name
    results['clt_checkpoint'] = clt_checkpoint
    results['json_file'] = json_file
    results['cluster_id'] = cluster_id
    
    return results

def compare_multiple_clusters(
    json_file: str,
    cluster_descriptions: List[str],
    target_tokens: List[str],
    clt_checkpoint: str,
    model_name: str = "CausalNLP/gpt2-hf_multilingual-90",
    activation_strength: float = 1.0,
    device: str = "cuda"
) -> Dict[str, Dict]:
    """
    Compare alignment across multiple clusters.
    
    Returns:
        Dict mapping cluster names to their analysis results
    """
    
    all_results = {}
    
    for cluster_desc in cluster_descriptions:
        print(f"\n{'='*60}")
        try:
            results = load_and_analyze_cluster_from_json(
                json_file=json_file,
                cluster_description=cluster_desc,
                target_tokens=target_tokens,
                clt_checkpoint=clt_checkpoint,
                model_name=model_name,
                activation_strength=activation_strength,
                device=device
            )
            all_results[cluster_desc] = results
        except Exception as e:
            print(f"Error analyzing cluster '{cluster_desc}': {e}")
            all_results[cluster_desc] = {'error': str(e)}
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("CLUSTER COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for token in target_tokens:
        print(f"\nAlignment with '{token}':")
        cluster_scores = []
        for cluster_name, results in all_results.items():
            if 'summary_stats' in results and token in results['summary_stats']:
                mean_align = results['summary_stats'][token]['mean_alignment']
                cluster_scores.append((cluster_name, mean_align))
        
        # Sort by alignment score
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (cluster_name, score) in enumerate(cluster_scores):
            print(f"  {i+1:2d}. {cluster_name:25s}: {score:8.4f}")
    
    return all_results

# Example usage
if __name__ == "__main__":
    # Analyze specific cluster
    result = load_and_analyze_cluster_from_json(
        json_file="/home/abir19/FeatFlow/src/featflow/attribution/antonym_90_fr.json",
        cluster_description="homme", 
        target_tokens=["homme", "man", "Mann"],
        clt_checkpoint="/home/abir19/gpt2_multilingual_90_clt/fsvqfwk0/final_16997376",
        activation_strength=1.0
    )
    
    # Or compare multiple clusters
    comparison = compare_multiple_clusters(
        json_file="/home/abir19/FeatFlow/src/featflow/attribution/antonym_90_de.json",
        cluster_descriptions=["frau und mann", "herr"],
        target_tokens=["frau", "herr", "mann", "woman", "man"],
        clt_checkpoint="/home/abir19/gpt2_multilingual_90_clt/fsvqfwk0/final_16997376"
    )