import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def get_cluster_features(json_path, cluster_desc):
    """Extract feature coordinates for a specific cluster"""
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    features = []
    for cluster_id, cluster in json_data["clusters"].items():
        if cluster["description"] == cluster_desc:
            for node in cluster["nodes"]:
                feature_coord = (node["layer"], node["pos"], node["feature_idx"])
                features.append(feature_coord)
    return features

def calculate_activation(feature_coords, coord_to_adj_col, activation_values):
    """Calculate activation for a set of features"""
    found_activations = []
    found_features = 0
    
    for feature_coord in feature_coords:
        if feature_coord in coord_to_adj_col:
            found_features += 1
            adj_col = coord_to_adj_col[feature_coord]
            activation = activation_values[adj_col].item()
            found_activations.append(activation)
    
    if found_activations:
        avg_activation = sum(found_activations) / len(found_activations)
        total_activation = sum(found_activations)
    else:
        avg_activation = 0.0
        total_activation = 0.0
    
    return avg_activation, total_activation, found_features, len(feature_coords)

def analyze_cluster_activations_per_language(languages):
    """
    Measure activation of each cluster in each language.
    For empty cluster lists, use English/German features but measure in target language graph.
    """
    
    results = {}
    
    for lang_name, config in languages.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {lang_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(config['pytorch_path']):
            print(f"  Error: PyTorch file not found")
            continue
        
        data = torch.load(config['pytorch_path'])
        selected_features = data["selected_features"]
        active_features = data["active_features"]
        activation_values = data["activation_values"]
        
        coord_to_adj_col = {}
        for adj_col, active_feat_idx in enumerate(selected_features):
            layer, pos, feat_id = active_features[active_feat_idx]
            coord_to_adj_col[(layer.item(), pos.item(), feat_id.item())] = adj_col
        
        results[lang_name] = {}
        
        for pos, cluster_list in enumerate(config['cluster_lists']):
            is_empty = len(cluster_list) == 0
            
            if is_empty:
                print(f"  Position {pos}: Empty list - using English/German features")
                
                # Get English clusters at this position
                english_clusters = []
                if 'English' in languages and pos < len(languages['English']['cluster_lists']):
                    english_clusters = languages['English']['cluster_lists'][pos]
                
                # Get German clusters at this position
                german_clusters = []
                if 'German' in languages and pos < len(languages['German']['cluster_lists']):
                    german_clusters = languages['German']['cluster_lists'][pos]
                
                # Combine unique clusters
                fallback_clusters = list(set(english_clusters + german_clusters))
                
                for cluster_desc in fallback_clusters:
                    activations_to_max = []
                    
                    # Try English features in current language graph
                    if cluster_desc in english_clusters and 'English' in languages:
                        eng_features = get_cluster_features(languages['English']['json_path'], cluster_desc)
                        avg_act, tot_act, found, total = calculate_activation(
                            eng_features, coord_to_adj_col, activation_values)
                        activations_to_max.append(avg_act)
                        print(f"    '{cluster_desc}' from English features: {avg_act:.6f}")
                    
                    # Try German features in current language graph
                    if cluster_desc in german_clusters and 'German' in languages:
                        ger_features = get_cluster_features(languages['German']['json_path'], cluster_desc)
                        avg_act, tot_act, found, total = calculate_activation(
                            ger_features, coord_to_adj_col, activation_values)
                        activations_to_max.append(avg_act)
                        print(f"    '{cluster_desc}' from German features: {avg_act:.6f}")
                    
                    # Take max
                    if activations_to_max:
                        final_activation = max(activations_to_max)
                        print(f"    Using max: {final_activation:.6f}")
                        
                        results[lang_name][cluster_desc] = {
                            'avg_activation': final_activation,
                            'total_activation': 0.0,
                            'found_features': 0,
                            'total_features': 0,
                            'is_fallback': True
                        }
            else:
                print(f"  Position {pos}: Using own clusters {cluster_list}")
                
                for cluster_desc in cluster_list:
                    features = get_cluster_features(config['json_path'], cluster_desc)
                    avg_act, tot_act, found, total = calculate_activation(
                        features, coord_to_adj_col, activation_values)
                    
                    results[lang_name][cluster_desc] = {
                        'avg_activation': avg_act,
                        'total_activation': tot_act,
                        'found_features': found,
                        'total_features': total,
                        'is_fallback': False
                    }
                    
                    print(f"    '{cluster_desc}': {avg_act:.6f} ({found}/{total} features)")
    
    return results

def print_activation_summary(results):
    """Print comprehensive summary table"""
    print("\n" + "="*80)
    print("ACTIVATION SUMMARY - ALL CLUSTERS ACROSS ALL LANGUAGES (RAW VALUES)")
    print("="*80)
    
    all_cluster_names = set()
    for lang_results in results.values():
        all_cluster_names.update(lang_results.keys())
    
    for cluster_name in sorted(all_cluster_names):
        print(f"\n{cluster_name}:")
        print(f"  {'Language':<15} {'Avg Activation':<15} {'Total Activation':<15} {'Found/Total':<12} {'Type'}")
        print(f"  {'-'*75}")
        
        for lang_name, lang_results in results.items():
            if cluster_name in lang_results:
                data = lang_results[cluster_name]
                type_str = "Fallback" if data.get('is_fallback', False) else "Own"
                if data.get('is_fallback', False):
                    print(f"  {lang_name:<15} {data['avg_activation']:<15.6f} "
                          f"{'N/A':<15} {'N/A':<12} {type_str}")
                else:
                    print(f"  {lang_name:<15} {data['avg_activation']:<15.6f} "
                          f"{data['total_activation']:<15.6f} "
                          f"{data['found_features']}/{data['total_features']:<8} {type_str}")

def plot_cluster_activations_heatmap(results):
    """Create heatmap of raw average activations"""
    plt.rcParams.update({
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.weight": "bold"
    })
    
    all_cluster_names = sorted(set(
        cluster for lang_results in results.values() 
        for cluster in lang_results.keys()
    ))
    languages = list(results.keys())
    
    matrix = np.zeros((len(languages), len(all_cluster_names)))
    
    for i, lang in enumerate(languages):
        for j, cluster in enumerate(all_cluster_names):
            if cluster in results[lang]:
                matrix[i, j] = results[lang][cluster]['avg_activation']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(all_cluster_names)))
    ax.set_yticks(np.arange(len(languages)))
    ax.set_xticklabels(all_cluster_names, rotation=45, ha='right')
    ax.set_yticklabels(languages)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Activation (Raw)', fontweight='bold')
    
    for i in range(len(languages)):
        for j in range(len(all_cluster_names)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Cluster Activations Across Languages (Raw Values)', fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig('cluster_activations_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('cluster_activations_heatmap.pdf', dpi=300, bbox_inches='tight')
    print("\nSaved: cluster_activations_heatmap.png/pdf")
    plt.close()

def plot_cluster_comparison_bars(results):
    """Bar plot comparing each cluster across languages"""
    plt.rcParams.update({
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.weight": "bold"
    })
    
    colors = {
        "English": "#264653",
        "German": "#D46F4D",
        "French": "#FFBF66",
        "Arabic": "#2a9d8f",
        "Chinese": "#8E5572",
    }
    
    all_cluster_names = sorted(set(
        cluster for lang_results in results.values() 
        for cluster in lang_results.keys()
    ))
    
    n_clusters = len(all_cluster_names)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 6))
    
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster_name in enumerate(all_cluster_names):
        ax = axes[idx]
        
        langs = []
        activations = []
        
        for lang_name, lang_results in results.items():
            if cluster_name in lang_results:
                langs.append(lang_name)
                activations.append(lang_results[cluster_name]['avg_activation'])
        
        bars = ax.bar(langs, activations, 
                     color=[colors.get(lang, '#888888') for lang in langs],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9)
        
        ax.set_title(f'"{cluster_name}"', fontweight='bold', fontsize=10)
        ax.set_ylabel('Average Activation (Raw)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('cluster_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig('cluster_comparison_bars.pdf', dpi=300, bbox_inches='tight')
    print("Saved: cluster_comparison_bars.png/pdf")
    plt.close()

if __name__ == "__main__":
    languages = {
        'English': {
            'pytorch_path': "/home/abir19/scratch/FeatFlow/example/graph/antonym_50_edge_log/en/attribution_graph.pt",
            'json_path': "/home/abir19/scratch/FeatFlow/src/featflow/attribution/antonym_50_en.json",
            'cluster_lists': [
                ["say woman (men&women)"],
                ["men"]
            ]
        },
        'German': {
            'pytorch_path': "/home/abir19/scratch/FeatFlow/example/graph/antonym_50_edge_log/de/attribution_graph.pt",
            'json_path': "/home/abir19/scratch/FeatFlow/src/featflow/attribution/antonym_50_de.json",
            'cluster_lists': [
                ["frau&mann"],
                ["man (multiling"]
            ],
        },
        'French': {
            'pytorch_path': "/home/abir19/scratch/FeatFlow/example/graph/antonym_50_edge_log/fr/attribution_graph.pt",
            'json_path': "/home/abir19/scratch/FeatFlow/src/featflow/attribution/antonym_50_fr.json",
            'cluster_lists': [
                ["man&woman"],
                ["man (multiling)"],
            ]
        },
        'Arabic': {
            'pytorch_path': "/home/abir19/scratch/FeatFlow/example/graph/antonym_50_edge_log/ar/attribution_graph.pt", 
            'json_path': "/home/abir19/scratch/FeatFlow/src/featflow/attribution/antonym_50_ar.json",
            'cluster_lists': [
                ["men&women"],
                ["rajol"]
            ]
        },
        'Chinese': {
            'pytorch_path': "/home/abir19/scratch/FeatFlow/example/graph/antonym_50_edge_log/zh/attribution_graph.pt",
            'json_path': "/home/abir19/scratch/FeatFlow/src/featflow/attribution/antonym_90_zh.json",
            'cluster_lists': [
                [],
                []
            ]
        }
    }
    
    results = analyze_cluster_activations_per_language(languages)
    print_activation_summary(results)
    plot_cluster_activations_heatmap(results)
    plot_cluster_comparison_bars(results)
    
    with open('cluster_activation_results_50.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved: cluster_activation_results_50.json")