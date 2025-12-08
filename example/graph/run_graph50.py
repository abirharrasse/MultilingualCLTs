import torch
from featflow.attribution.attribution import run_attribution

from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config

patch_official_model_names()
patch_convert_hf_model_config()

def main():

    clt_checkpoint = "/home/abir19/gpt2_multilingual_50_clt/iw7j220w/final_17765376"
    test_strings = [
        "J'ai bu une tasse"
    ]
    langs = ['ar', 'en', 'fr', 'de', 'zh']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    folder_name = "/home/abir19/FeatFlow/example/graph/multilingual"

    print(" Testing Graph Computation Pipeline", flush=True)
    print(f"   Checkpoint: {clt_checkpoint}", flush=True)
    print(f"   Device: {device}", flush=True)
    print("=" * 80, flush=True)
    
    for i, test_string in enumerate(test_strings, 1):
        print(f"\n Processing test string {i}: '{test_string}'", flush=True)
        
        try:

            result = run_attribution(
                folder_name=f'{folder_name}/preposition_de_50',
                clt_checkpoint=clt_checkpoint,
                input_string=test_string, 
                model_name="CausalNLP/gpt2-hf_multilingual-50", 
                max_n_logits=5,
                desired_logit_prob=0.95, 
                max_feature_nodes=20000, 
                batch_size=256, 
                feature_threshold=0.80, 
                edge_threshold=0.85,
                device=device,
                offload=None, 
                compute_interventions=True
            )

        except Exception as e:
            print(f" Attribution computation failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

        print("\n" + "="*80, flush=True)
    
    print("\n Testing complete!", flush=True)

if __name__ == "__main__":
    main()