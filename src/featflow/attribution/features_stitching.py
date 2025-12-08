import torch
from circuit_tracer import ReplacementModel
from featflow.attribution.loading import load_circuit_tracing_clt_from_local
from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from typing import List, Dict, Any, Tuple
import os 
import json
import itertools
import pandas as pd

def run_cluster_intervention(
    cluster_features: list,
    manual_features: list,
    clt_checkpoint: str,
    input_string: str,
    cluster_intervention_value: float,
    model_name: str = "roneneldan/TinyStories-33M",
    top_tokens_count: int = 4,
    device: str = "cuda",
    freeze_attention: bool = True,
    multiplicative: bool = False,
    max_new_tokens: int = 2
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

    input_tokens = model.ensure_tokenized(input_string)
    if not isinstance(input_tokens, torch.Tensor):
        input_tokens = torch.tensor(input_tokens, device=device).unsqueeze(0)
    elif input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0)
    
    print(f"Input tokens shape: {input_tokens.shape}")
    
    with torch.no_grad():
        baseline_output = model.generate(input_tokens, temperature=1e-5, max_new_tokens=max_new_tokens)
    baseline_generated = baseline_output[0, len(input_tokens[0]):]
    
    current_tokens = input_tokens.clone()
    generated_tokens = []
    all_probs = []
    
    for step in range(max_new_tokens):
        intervened_logits, _ = model.feature_intervention(
            current_tokens, 
            interventions=features_to_intervene, 
            freeze_attention=freeze_attention,
            multiplicative=multiplicative
        )
        
        probs = torch.softmax(intervened_logits[0, -1], dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        generated_tokens.append(next_token.item())
        all_probs.append(probs[next_token].item())
        
        current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
    
    generated_text = model.tokenizer.decode(generated_tokens)
    baseline_text = model.tokenizer.decode(baseline_generated)
    
    final_probs = torch.softmax(intervened_logits[0, -1], dim=-1)
    top_tokens = torch.topk(final_probs, top_tokens_count)
    top_token_strings = [model.tokenizer.decode([token_id]) for token_id in top_tokens.indices]
    
    return {
        'generated_text': generated_text,
        'generated_tokens': [model.tokenizer.decode([t]) for t in generated_tokens],
        'generated_probs': all_probs,
        'baseline_text': baseline_text,
        'baseline_tokens': [model.tokenizer.decode([t]) for t in baseline_generated.tolist()],
        'top_candidates': top_token_strings,
        'top_candidate_probs': top_tokens.values.cpu().tolist(),
        'cluster_feature_count': len(cluster_features),
        'manual_feature_count': len(manual_features)
    }

def extract_features_with_values(json_file: str, cluster_interventions: Dict[str, float]) -> List[Tuple[int, int, int, float]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    features = []
    for cluster_id, cluster_data in data['clusters'].items():
        description = cluster_data['description']
        if description in cluster_interventions:
            intervention_value = cluster_interventions[description]
            print(f"Extracting features from cluster {cluster_id} with description: {description}, intervention value: {intervention_value}")
            for node in cluster_data['nodes']:
                features.append((node['pos'], node['layer'], node['feature_idx'], intervention_value))
    
    print(f"Extracted features: {features}")
    return features

def run_intervention_on_clusters_with_values(
    json_file: str,
    cluster_interventions: Dict[str, float], 
    clt_checkpoint: str,
    input_string: str,
    model_name: str = "roneneldan/TinyStories-33M",
    top_tokens_count: int = 10,
    device: str = "cuda",
    freeze_attention: bool = True,
    multiplicative: bool = False,
    max_new_tokens: int = 2,
    additional_features: List[Tuple[int, int, int, float]] = None
):
    features_with_values = extract_features_with_values(json_file, cluster_interventions)
    
    if additional_features is None:
        additional_features = []
    
    all_features = features_with_values + additional_features
    
    return run_cluster_intervention(
        cluster_features=[],
        manual_features=all_features,
        clt_checkpoint=clt_checkpoint,
        input_string=input_string,
        cluster_intervention_value=0.0,
        model_name=model_name,
        top_tokens_count=top_tokens_count,
        device=device,
        freeze_attention=freeze_attention,
        multiplicative=multiplicative,
        max_new_tokens=max_new_tokens
    )

def get_baseline_generation(
    clt_checkpoint: str,
    input_string: str,
    model_name: str = "CausalNLP/gpt2-hf_multilingual-90",
    device: str = "cuda",
    max_new_tokens: int = 2
):
    patch_official_model_names()
    patch_convert_hf_model_config()
    
    clt = load_circuit_tracing_clt_from_local(clt_checkpoint, device=device)
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=model_name, 
        transcoders=clt
    )
    
    input_tokens = model.ensure_tokenized(input_string)
    if not isinstance(input_tokens, torch.Tensor):
        input_tokens = torch.tensor(input_tokens, device=device).unsqueeze(0)
    elif input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0)
    
    print(f"Baseline input tokens shape: {input_tokens.shape}")
    
    with torch.no_grad():
        output = model.generate(input_tokens, temperature=1e-5, max_new_tokens=max_new_tokens)
    
    generated = output[0, len(input_tokens[0]):]
    generated_text = model.tokenizer.decode(generated)
    
    return {
        'generated_text': generated_text,
        'generated_tokens': [model.tokenizer.decode([t]) for t in generated.tolist()],
        'model': model
    }

def sweep_intervention_values(
    json_file: str,
    clt_checkpoint: str,
    input_string: str,
    target_sequence: str,
    clusters_to_test: List[str],
    model_name: str = "CausalNLP/gpt2-hf_multilingual-90",
    device: str = "cuda",
    multiplicative: bool = False,
    max_new_tokens: int = 2,
    values_to_test: List[float] = None,
    suppress_clusters: List[str] = None,
    boost_clusters: List[str] = None,
    suppress_values: List[float] = None,
    boost_values: List[float] = None,
    multi_cluster_configs: List[Dict[str, float]] = None,
    additional_features: List[Tuple[int, int, int, float]] = None
):
    
    if values_to_test is None:
        values_to_test = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    
    print("Getting baseline generation...")
    baseline_info = get_baseline_generation(clt_checkpoint, input_string, model_name, device, max_new_tokens)
    baseline_text = baseline_info['generated_text']
    
    print(f"Baseline generation: '{baseline_text}'")
    
    if additional_features is None:
        additional_features = []
    
    results = []
    
    print("=== Testing individual cluster interventions ===")
    for cluster in clusters_to_test:
        for value in values_to_test:
            cluster_interventions = {cluster: value}
            
            try:
                result = run_intervention_on_clusters_with_values(
                    json_file=json_file,
                    cluster_interventions=cluster_interventions,
                    clt_checkpoint=clt_checkpoint,
                    input_string=input_string,
                    model_name=model_name,
                    top_tokens_count=10,
                    device=device,
                    multiplicative=multiplicative,
                    max_new_tokens=max_new_tokens,
                    additional_features=additional_features)
                

                generated_text = result['generated_text']
                target_match = target_sequence in generated_text
                
                results.append({
                    'strategy': 'single',
                    'cluster': cluster,
                    'value': value,
                    'generated_text': generated_text,
                    'baseline_text': baseline_text,
                    'target_match': target_match,
                    'generated_tokens': result['generated_tokens'],
                    'config': str(cluster_interventions)
                })
                
                match_str = "✓" if target_match else "✗"
                print(f"{cluster}={value}: '{generated_text}' | target '{target_sequence}': {match_str}")
                
            except Exception as e:
                print(f"Error with {cluster}={value}: {e}")
    
    if suppress_clusters and boost_clusters:
        print("\n=== Testing suppress/boost combination strategies ===")
        
        if suppress_values is None:
            suppress_values = [-20, -15, -10, -5, 0]
        if boost_values is None:
            boost_values = [0, 1, 5, 10, 15, 20, 25]
        
        for suppress_cluster in suppress_clusters:
            for boost_cluster in boost_clusters:
                for s_val, b_val in itertools.product(suppress_values, boost_values):
                    cluster_interventions = {
                        suppress_cluster: s_val,
                        boost_cluster: b_val
                    }
                    
                    try:
                        result = run_intervention_on_clusters_with_values(
                            json_file=json_file,
                            cluster_interventions=cluster_interventions,
                            clt_checkpoint=clt_checkpoint,
                            input_string=input_string,
                            model_name=model_name,
                            top_tokens_count=10,
                            device=device,
                            multiplicative=multiplicative,
                            max_new_tokens=max_new_tokens,
                            additional_features=additional_features
                        )
                        
                        generated_text = result['generated_text']
                        target_match = target_sequence in generated_text
                        
                        results.append({
                            'strategy': 'suppress_boost',
                            'cluster': f'{suppress_cluster}+{boost_cluster}',
                            'value': f"{s_val},{b_val}",
                            'generated_text': generated_text,
                            'baseline_text': baseline_text,
                            'target_match': target_match,
                            'generated_tokens': result['generated_tokens'],
                            'config': str(cluster_interventions)
                        })
                        
                        match_str = "✓" if target_match else "✗"
                        print(f"{suppress_cluster}={s_val}, {boost_cluster}={b_val}: '{generated_text}' | target '{target_sequence}': {match_str}")
                        
                    except Exception as e:
                        print(f"Error with {suppress_cluster}={s_val}, {boost_cluster}={b_val}: {e}")
    
    if multi_cluster_configs:
        print("\n=== Testing multi-cluster approach ===")
        
        for i, config in enumerate(multi_cluster_configs):
            try:
                result = run_intervention_on_clusters_with_values(
                    json_file=json_file,
                    cluster_interventions=config,
                    clt_checkpoint=clt_checkpoint,
                    input_string=input_string,
                    model_name=model_name,
                    top_tokens_count=10,
                    device=device,
                    multiplicative=multiplicative,
                    max_new_tokens=max_new_tokens
                )
                
                generated_text = result['generated_text']
                target_match = target_sequence in generated_text
                
                results.append({
                    'strategy': 'multi_cluster',
                    'cluster': f'config_{i}',
                    'value': str(config),
                    'generated_text': generated_text,
                    'baseline_text': baseline_text,
                    'target_match': target_match,
                    'generated_tokens': result['generated_tokens'],
                    'config': str(config)
                })
                
                match_str = "✓" if target_match else "✗"
                print(f"Config {i}: '{generated_text}' | target '{target_sequence}': {match_str} | {config}")
                
            except Exception as e:
                print(f"Error with config {i}: {e}")
    
    df = pd.DataFrame(results)
    
    print("\n=== ANALYSIS ===")
    print(f"Baseline: '{baseline_text}'")
    print(f"Total tests: {len(df)}")
    print(f"Successful matches with '{target_sequence}': {df['target_match'].sum()}")
    
    if df['target_match'].any():
        successful = df[df['target_match'] == True].copy()
        print(f"\nSuccessful configurations:")
        for _, row in successful.iterrows():
            print(f"  {row['config']} -> '{row['generated_text']}'")
    
    print(f"\nGenerated sequences:")
    print(df['generated_text'].value_counts().head(10))
    
    print(f"\nMost common first tokens:")
    first_tokens = [tokens[0] if tokens else '' for tokens in df['generated_tokens']]
    first_token_counts = pd.Series(first_tokens).value_counts()
    print(first_token_counts.head(10))
    
    return df

# Example usage - German with additional manual features
additional_manual_features = [
    (7, 8, 4886, 25.0),  # pos=7, layer=8, feature_idx=4886, value=25.0
]

results_df = sweep_intervention_values(
    json_file="/home/abir19/FeatFlow/src/featflow/attribution/antonym_90_fr.json",
    clt_checkpoint="/home/abir19/gpt2_multilingual_90_clt/fsvqfwk0/final_16997376", 
    input_string='Le contraire de "homme" est "',
    target_sequence="femme",
    clusters_to_test=["homme", "femme", "woman arabic", "french", "homme multiling"],
    suppress_clusters=["homme", "homme multiling"],
    boost_clusters=["femme", "woman arabic", "french"],
    additional_features=additional_manual_features,
    multi_cluster_configs=[
        # {"homme": -20, "femme": 25, "woman arabic": 15},
        # {"homme": -25, "femme": 30, "woman arabic": 20, "homme multiling": -15},
        # {"homme": -30, "femme": 35, "woman arabic": 25, "french": 10},
        # {"homme": -15, "femme": 20, "woman arabic": 15, "french": 5, "homme multiling": -10},
        {"femme": 40, "woman arabic": 30,"homme": -35, "homme multiling": -25, "french": 30}
    ],
    multiplicative=True,
    max_new_tokens=2
)