import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from featflow.causal_graph.replacement_model import ReplacementModel
from featflow.causal_graph.attribution_graph import SimpleAttributionGraph
from featflow.attribution.attribution import run_attribution
import gc
import os
import pickle

def create_multi_model_plot(results_dict, output_path):
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'legend.frameon': False,
        'figure.dpi': 300
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        "GPT2 Multilingual 20%": "#264653",
        "GPT2 Multilingual 50%": "#D46F4D",
        "GPT2 Multilingual 70%": "#FFBF66",
        "GPT2 Multilingual 90%": "#2a9d8f"
    }
    
    model_mapping = {
        '20': "GPT2 Multilingual 20%",
        '50': "GPT2 Multilingual 50%",
        '70': "GPT2 Multilingual 70%",
        '90': "GPT2 Multilingual 90%"
    }
    
    for model_num, model_label in model_mapping.items():
        if model_num in results_dict:
            data = results_dict[model_num]
            layers = data['layers']
            mean_sims = data['mean_similarities']
            
            ax.plot(layers, mean_sims, 
                   marker='o', linewidth=2.5, markersize=8,
                   color=colors[model_label], markerfacecolor=colors[model_label], 
                   markeredgecolor='white', markeredgewidth=1.5,
                   label=model_label)
    
    ax.set_xlabel('Transformer Layer', fontsize=16, fontweight='bold')
    ax.set_ylabel('Jaccard Similarity', fontsize=16, fontweight='bold')
    ax.set_title('Cross-lingual Feature Similarity Across Models and Layers', 
                fontsize=18, fontweight='bold', pad=20)
    
    all_layers = []
    all_similarities = []
    for model_num in results_dict:
        data = results_dict[model_num]
        all_layers.extend(data['layers'])
        all_similarities.extend(data['mean_similarities'])
    
    max_layer = max(all_layers) if all_layers else 11
    max_similarity = max(all_similarities) if all_similarities else 0.6
    
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.set_ylim(0, max_similarity * 1.1)
    ax.set_xticks(range(0, max_layer + 1, 2))
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=14)
    
    plt.tight_layout()
    
    plt.savefig(f'{output_path}_multi_model_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}_multi_model_comparison.png', format='png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return f'{output_path}_multi_model_comparison.pdf'

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def jaccard_similarity(set1, set2, layer=None, force_zero_layer0=False):
    if force_zero_layer0 and layer == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    max_size = max(len(set1), len(set2))
    return intersection / max_size if max_size > 0 else 0

def get_features_per_layer(feature_indices, n_layers):
    features_by_layer = defaultdict(set)
    for pos, layer, feat_idx in feature_indices:
        features_by_layer[layer].add((pos, feat_idx))
    
    for layer in range(n_layers):
        if layer not in features_by_layer:
            features_by_layer[layer] = set()
    
    return features_by_layer

def load_existing_results(folder_path):
    result_file = os.path.join(folder_path, 'attribution_graph.pt')
    if os.path.exists(result_file):
        try:
            data = torch.load(result_file)
            print(f"Loaded existing result from {result_file}")
            return data
        except Exception as e:
            print(f"Failed to load {result_file}: {e}")
    else:
        print(f"No existing result at {result_file}")
    return None

def compute_layer_similarities(result1, result2, max_layers, debug=False, force_zero_layer0=False):
    features1_by_layer = get_features_per_layer(result1['processed_feature_indices'], max_layers)
    features2_by_layer = get_features_per_layer(result2['processed_feature_indices'], max_layers)
    
    layer_sims = []
    for layer in range(max_layers):
        set1 = features1_by_layer[layer]
        set2 = features2_by_layer[layer]
        sim = jaccard_similarity(set1, set2, layer, force_zero_layer0)
        layer_sims.append(sim)
        
        if debug and layer == 0:
            print(f"Layer 0 debug: set1={len(set1)} features, set2={len(set2)} features, similarity={sim:.4f}")
            if len(set1) > 0 or len(set2) > 0:
                print(f"  set1 features: {list(set1)[:5]}...")  
                print(f"  set2 features: {list(set2)[:5]}...")
    
    return layer_sims



def filter_frequent_features(all_results, languages, min_frequency=0, layer0_min_frequency=None, remove_layer0_completely=False):
    if remove_layer0_completely:
        print("Removing ALL layer 0 features")
        layer0_min_frequency = float('inf')
    elif layer0_min_frequency is None:
        layer0_min_frequency = 0
    
    if layer0_min_frequency == 0 and not remove_layer0_completely:
        print("Skipping frequency filtering (layer0 threshold is 0)")
        return all_results, set()
    
    feature_counts = defaultdict(int)
    
    for example_idx in all_results:
        for lang in languages:
            result = all_results[example_idx][lang]
            if result is None:
                continue
                
            feature_indices = result['processed_feature_indices']
            for pos, layer, feat_idx in feature_indices:
                feature_key = (layer, feat_idx)
                feature_counts[feature_key] += 1
    
    total_valid_results = sum(
        1 for example_idx in all_results 
        for lang in languages 
        if all_results[example_idx][lang] is not None
    )
    
    features_to_remove = set()
    layer0_removed = 0
    
    # Only filter layer 0 features
    for (layer, feat_idx), count in feature_counts.items():
        if layer == 0:  # Only process layer 0
            if remove_layer0_completely:
                features_to_remove.add((layer, feat_idx))
                layer0_removed += 1
            elif count >= layer0_min_frequency:
                features_to_remove.add((layer, feat_idx))
                layer0_removed += 1
    
    print(f"Filtering features - layer0_threshold: {layer0_min_frequency}")
    print(f"Total valid results: {total_valid_results}")
    print(f"Features to remove: {len(features_to_remove)} (all from layer0: {layer0_removed})")
    
    layer0_total = sum(1 for (layer, _) in feature_counts.keys() if layer == 0)
    layer0_kept = layer0_total - layer0_removed
    print(f"Layer 0 features: {layer0_total} total, {layer0_kept} kept, {layer0_removed} removed")
    
    filtered_results = {}
    for example_idx in all_results:
        filtered_results[example_idx] = {}
        for lang in languages:
            result = all_results[example_idx][lang]
            if result is None:
                filtered_results[example_idx][lang] = None
                continue
                
            feature_indices = result['processed_feature_indices']
            
            filtered_indices = []
            for pos, layer, feat_idx in feature_indices:
                if (layer, feat_idx) not in features_to_remove:
                    filtered_indices.append([pos, layer, feat_idx])
            
            if filtered_indices:
                filtered_results[example_idx][lang] = {
                    'processed_feature_indices': np.array(filtered_indices),
                    'n_layers': result['n_layers'],
                    'input_tokens': result['input_tokens']
                }
            else:
                print(f"Warning: All features filtered out for example {example_idx}, {lang}")
                filtered_results[example_idx][lang] = None
    
    return filtered_results, features_to_remove




def process_single_model(model_config, min_frequency=0, layer0_min_frequency=None, remove_layer0_completely=False):
    model_num = model_config['model_num']
    folder_name = model_config['folder_name']
    clt_checkpoint = model_config['clt_checkpoint']
    model_name = model_config['model_name']
    
    print(f"Processing Model {model_num}...")
    
    examples = [
        {
            'english': 'The opposite of large is',
            'french': "L'opposé de grand est",
            'german': "Das Gegenteil von groß ist",
            'arabic': "عكس كبير هو",
            'chinese': "大的反义词是"
        },
        {
            'english': 'The opposite of hot is',
            'french': "L'opposé de chaud est",
            'german': "Das Gegenteil von heiß ist",
            'arabic': "عكس حار هو",
            'chinese': "热的反义词是"
        },
        {
            'english': 'John and Mary went to the store. John gave a book to',
            'french': "John et Marie sont allés au magasin. John a donné un livre à",
            'german': "John und Maria gingen zum Laden. John gab ein Buch an",
            'arabic': "جون وماري ذهبا إلى المتجر. جون أعطى كتابًا لـ",
            'chinese': "约翰和玛丽去了商店。约翰把一本书给了"
        },
        {
            'english': 'Alice and Bob went to the restaurant. Alice gave an apple to',
            'french': "Alice et Bob sont allés au restaurant. Alice a donné une pomme à",
            'german': "Alice und Bob gingen ins Restaurant. Alice gab einen Apfel an",
            'chinese': "爱丽丝和鲍勃去了餐馆。爱丽丝把一个苹果给了",
            'arabic': "ذهبت أليس وبوب إلى المطعم. أعطت أليس تفاحة إلى",
        },
        {
            'english': 'Bird, Pig, Horse are all',
            'french': "Oiseau, Cochon, Cheval sont tous",
            'german': "Vogel, Schwein, Pferd sind alle",
            'arabic': "طائر، خنزير، حصان كلها",
            'chinese': "鸟、猪、马 都是"
        },
        {
            'english': 'Yellow, Black, Red are all',
            'french': "Jaune, Noir, Rouge sont tous",
            'german': "Gelb, Schwarz, Rot sind alle",
            'arabic': "أصفر، أسود، أحمر كلها",
            'chinese': "黄色、黑色、红色 都是"
        },
        {
            'english': 'Monday, Tuesday, Wednesday, Thursday,',
            'french': "Lundi, Mardi, Mercredi, Jeudi,",
            'german': "Montag, Dienstag, Mittwoch, Donnerstag,",
            'arabic': "الاثنين، الثلاثاء، الأربعاء، الخميس،",
            'chinese': "星期一、星期二、星期三、星期四、"
        }
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    languages = ["english", "french", "german", "arabic", "chinese"]
    all_results = {}
    
    for example_idx, example in enumerate(examples):
        if example_idx in [0, 1, 2, 3, 4, 5, 6]:
            all_results[example_idx] = {}
            
            for lang in languages:
                sentence = example[lang]
                current_folder = f"{folder_name}_ex{example_idx}_{lang}"
                
                print(f"Processing example {example_idx}, language {lang}")
                print(f"Looking for results in: {current_folder}")
                
                existing_result = load_existing_results(current_folder)
                if existing_result is not None:
                    result = existing_result
                else:
                    print(f"Running new attribution for {lang} example {example_idx}")
                    clear_gpu_memory()
                    try:
                        result = run_attribution(
                            folder_name=current_folder,
                            clt_checkpoint=clt_checkpoint,
                            input_string=sentence, 
                            model_name=model_name, 
                            max_n_logits=5,
                            desired_logit_prob=0.80, 
                            max_feature_nodes=8192, 
                            batch_size=256, 
                            feature_threshold=1.0, 
                            edge_threshold=1.0,
                            device=device,
                            offload=None
                        )
                        print(f"Successfully ran attribution for {lang} example {example_idx}")
                    except Exception as e:
                        print(f"Failed to run attribution for example {example_idx}, {lang}: {e}")
                        all_results[example_idx][lang] = None
                        clear_gpu_memory()
                        continue
                
                try:
                    if 'feature_indices' not in result:
                        print(f"No feature_indices in result for {lang} example {example_idx}")
                        print(f"Available keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                        all_results[example_idx][lang] = None
                        continue
                        
                    feature_indices = result['feature_indices']
                    if hasattr(feature_indices, 'cpu'):
                        feature_indices = feature_indices.cpu()
                        
                    input_tokens = result['input_tokens']
                    n_tokens = len(input_tokens)
                    n_layers = int(feature_indices[:, 1].max()) + 1
                    
                    all_results[example_idx][lang] = {
                        'processed_feature_indices': feature_indices.numpy() if hasattr(feature_indices, 'numpy') else feature_indices,
                        'n_layers': n_layers,
                        'input_tokens': input_tokens
                    }
                    
                    print(f"Processed {lang} example {example_idx}: {n_layers} layers, {len(feature_indices)} features")
                    
                    del result
                    clear_gpu_memory()
                    
                except Exception as e:
                    print(f"Failed to process result for example {example_idx}, {lang}: {e}")
                    if isinstance(result, dict):
                        print(f"Result keys: {list(result.keys())}")
                    else:
                        print(f"Result type: {type(result)}")
                    all_results[example_idx][lang] = None
                    clear_gpu_memory()
    
    filtered_results, _ = filter_frequent_features(all_results, languages, min_frequency, layer0_min_frequency, remove_layer0_completely)
    
    layer_similarities = defaultdict(list)
    max_layers = 0
    
    for example_idx in filtered_results:
        for lang in languages:
            if filtered_results[example_idx][lang] is not None:
                max_layers = max(max_layers, filtered_results[example_idx][lang]['n_layers'])
    
    print(f"Max layers found: {max_layers}")
    
    for example_idx in filtered_results:
        valid_results = {lang: filtered_results[example_idx][lang] 
                        for lang in languages if filtered_results[example_idx][lang] is not None}
        
        print(f"Example {example_idx}: {len(valid_results)} valid languages: {list(valid_results.keys())}")
        
        if len(valid_results) < 2:
            print(f"  Skipping - need at least 2 languages for comparison")
            continue
            
        lang_list = list(valid_results.keys())
        for i in range(len(lang_list)):
            for j in range(i + 1, len(lang_list)):
                lang1, lang2 = lang_list[i], lang_list[j]
                
                try:
                    layer_sims = compute_layer_similarities(
                        valid_results[lang1], valid_results[lang2], max_layers, 
                        debug=(example_idx == 0 and i == 0 and j == 1),  # Debug first comparison only
                        force_zero_layer0=remove_layer0_completely)
                    
                    for layer, jaccard_sim in enumerate(layer_sims):
                        layer_similarities[layer].append(jaccard_sim)
                        
                    print(f"  {lang1}-{lang2}: computed {len(layer_sims)} layer similarities")
                except Exception as e:
                    print(f"  Error computing {lang1}-{lang2}: {e}")
                    continue
    
    print(f"Layer similarities summary:")
    for layer in range(max_layers):
        count = len(layer_similarities.get(layer, []))
        print(f"  Layer {layer}: {count} similarity pairs")
    
    mean_similarities = []
    std_similarities = []
    layers = []
    
    for layer in range(max_layers):
        if layer in layer_similarities and len(layer_similarities[layer]) > 0:
            similarities = layer_similarities[layer]
            mean_similarities.append(np.mean(similarities))
            std_similarities.append(np.std(similarities))
            layers.append(layer)
        else:
            print(f"Warning: No similarities for layer {layer}")
    
    print(f"Final result: {len(layers)} layers with data")
    
    return {
        'layers': layers,
        'mean_similarities': mean_similarities,
        'std_similarities': std_similarities,
        'max_layers': max_layers
    }

def main(layer0_min_frequency=0):
    model_configs = [
        {
            'model_num': '20',
            'folder_name': '/home/abir19/FeatFlow/example/graph/save_20',
            'clt_checkpoint': '/home/abir19/gpt2_multilingual_20_clt/zhb8w33x/final_17478656',
            'model_name': 'CausalNLP/gpt2-hf_multilingual-20'
        },
        {
            'model_num': '50',
            'folder_name': '/home/abir19/FeatFlow/example/graph/save_50',
            'clt_checkpoint': '/home/abir19/gpt2_multilingual_50_clt/iw7j220w/final_17765376',
            'model_name': 'CausalNLP/gpt2-hf_multilingual-50'
        },
        {
            'model_num': '70',
            'folder_name': '/home/abir19/FeatFlow/example/graph/save_70',
            'clt_checkpoint': '/home/abir19/gpt2_multilingual_70_clt/886fq5nf/final_17529856',
            'model_name': 'CausalNLP/gpt2-hf_multilingual-70'
        },
        {
            'model_num': '90',
            'folder_name': '/home/abir19/FeatFlow/example/graph/save_90',
            'clt_checkpoint': '/home/abir19/gpt2_multilingual_90_clt/fsvqfwk0/final_16997376',
            'model_name': 'CausalNLP/gpt2-hf_multilingual-90'
        }
    ]
    
    print("Multi-Model Cross-lingual Feature Analysis")
    print(f"Layer 0 threshold: {layer0_min_frequency}, Other layers threshold: {min_frequency}")
    print(f"Remove layer 0 completely: {remove_layer0_completely}")
    print("=" * 60)
    
    results_dict = {}
    
    for config in model_configs:
        try:
            model_results = process_single_model(config, min_frequency, layer0_min_frequency, remove_layer0_completely)
            results_dict[config['model_num']] = model_results
            print(f"Model {config['model_num']} completed successfully")
        except Exception as e:
            print(f"Model {config['model_num']} failed: {e}")
    
    if results_dict:
        output_file = create_multi_model_plot(results_dict, '/home/abir19/FeatFlow/example/graph')
        print(f"Combined plot saved as: {output_file}")
        
        print("\nSummary:")
        for model_num, data in results_dict.items():
            if data['mean_similarities']:
                mean_overall = np.mean(data['mean_similarities'])
                print(f"Model {model_num}: Average similarity = {mean_overall:.4f}")
            else:
                print(f"Model {model_num}: No similarities computed")
    
    return results_dict

if __name__ == "__main__":
    # Now layer0_min_frequency=2 means: keep only features appearing in 2+ examples
    print("=== Testing with layer0_min_frequency=2 (must appear in 2+ examples) ===")
    results = main(layer0_min_frequency=5)