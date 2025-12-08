import torch
from featflow.attribution.attribution import run_attribution

from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config

patch_official_model_names()
patch_convert_hf_model_config()

def main():
    # Configuration

    clt_checkpoint = "/home/abir19/projects/llama_multilingual_clt"
    test_strings = [
        'The tastiest dish is "',
       'Das leckerste Gericht ist die "',
    ]

    langs = ['en', 'de']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    folder_name = "/home/abir19/FeatFlow/example/graph/llama_MT"

    print(" Testing Graph Computation Pipeline", flush=True)
    print(f"   Checkpoint: {clt_checkpoint}", flush=True)
    print(f"   Device: {device}", flush=True)
    print("=" * 80, flush=True)
    
    for i, test_string in enumerate(test_strings, 1):
        print(f"\n Processing test string {i}: '{test_string}'", flush=True)
        
        try:

            result = run_attribution(
                folder_name=f'{folder_name}/{langs[i-1]}',
                clt_checkpoint=clt_checkpoint,
                input_string=test_string, 
                model_name="meta-llama/Llama-3.2-1B", 
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
    
    print("\n🎉 Testing complete!", flush=True)

if __name__ == "__main__":
    main()