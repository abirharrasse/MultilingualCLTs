import torch
from featflow.causal_graph.run_attribution import run_attribution
import os
import numpy as np

def main():
    clt_checkpoint = "/home/aharrasse/Multilingual-CLT/y4ck9q08/final_25600000"
    
    # test_strings = [
    #     'Sarah and Tom were at the hospital. Sarah gave a phone to',
    #     'James and Brian were at the restaurant. James gave a book to'
    # ]

    test_strings = [
        'The opposite of "large" is "',
        'The opposite of "good" is "', # Tom and Kate enter the store. Kate gives an apple to
        'The opposite of "sad" is "',
        'The opposite of "woman" is "'
    ]

    # test_strings = [
    #     'L\'opposé de "femme" est "',
    #     'L\'opposé de "grand" est "',
    #     'L\'opposé de "bon" est "',  # Tom et Kate entrent dans le magasin. Kate donne une pomme à
    #     'L\'opposé de "triste" est "'   
    # ]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder_name = "/home/aharrasse/FeatFlow/example/graph/save"
    
    print("🔬 Computing Graphs and Finding Common Features", flush=True)
    print(f"   Checkpoint: {clt_checkpoint}", flush=True)
    print(f"   Device: {device}", flush=True)
    print("=" * 80, flush=True)
    
    results = []
    
    # Compute graphs for each string
    for i, test_string in enumerate(test_strings, 1):
        print(f"\n Processing test string {i}: '{test_string}'", flush=True)
        
        try:
            result = run_attribution(
                folder_name=f"{folder_name}_{i}",
                clt_checkpoint=clt_checkpoint,
                input_string=test_string,
                model_class_name="HookedTransformer",
                model_name="CausalNLP/gpt2-hf_multilingual-70",
                feature_threshold=0.7,
                edge_threshold=0.7,
                device=device,
                compute_correlation=False
            )
            # Process adjacency matrix and feature indices
            pruned_adjacency = result['prune_adjacency_matrix']
            feature_indices = result['feature_indices']
            input_tokens = result['input_tokens']
            n_tokens = len(input_tokens)
            n_layers = int(feature_indices[:, 1].max()) + 1
            n_errors = n_tokens * n_layers
            
            # Remove first n_tokens rows and columns
            pruned_adjacency = pruned_adjacency[n_tokens+n_errors:]
            pruned_adjacency = pruned_adjacency[:, n_tokens+n_errors:]

            # Filter active features - keep only features with outgoing edges
            row_sums = pruned_adjacency.abs().sum(axis=1)
            active_mask = row_sums > 0
            
            # Remove features at token position 0
            token0_mask = feature_indices[:, 0] != 0
            active_mask = active_mask & token0_mask

            feature_indices = feature_indices[active_mask]
            result['processed_feature_indices'] = feature_indices
            
            print(f"Graph computed! Features: {feature_indices.shape[0]}", flush=True)
            results.append(result)
            
        except Exception as e:
            print(f"Attribution computation failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return
    
    # Find common features across all graphs
    print("\n🔍 Finding common features...", flush=True)
    
    if len(results) >= 2:
        # Convert feature indices to sets of tuples (pos, layer, feat_idx)
        feature_sets = []
        for result in results:
            feature_tuples = set(map(tuple, result['processed_feature_indices'].cpu().numpy()))
            feature_sets.append(feature_tuples)
        
        # Find intersection of all feature sets
        common_features = feature_sets[0]
        for feature_set in feature_sets[1:]:
            common_features = common_features.intersection(feature_set)
        
        common_features_list = list(common_features)
        
        print(f"   - Common features found: {len(common_features_list)}", flush=True)
        
        # Save common features to first string's results
        if common_features_list:
            first_folder = f"{folder_name}_1"

            # Load the existing saved results from first string
            results_path = os.path.join(first_folder, "attribution_graph.pt")
            final_path = os.path.join(folder_name, "attribution_graph.pt") 

            if os.path.exists(results_path):
                saved_data = torch.load(results_path)

                # Add common features to the saved data
                saved_data['feature_list_intersection'] = common_features_list
                
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                
                for key, value in saved_data.items():
                    if isinstance(value, np.ndarray):
                        saved_data[key] = torch.from_numpy(value)

                torch.save(saved_data, final_path)
                
                print(f"Common features saved to {final_path}", flush=True)
            else:
                print(f" Could not find saved results at {results_path}", flush=True)
        
    else:
        print("   - Need at least 2 graphs to find common features", flush=True)
    
    print("\n Process complete!", flush=True)
    
    return results

if __name__ == "__main__":
    results = main()