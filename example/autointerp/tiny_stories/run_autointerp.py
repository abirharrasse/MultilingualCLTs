from featflow.config.autointerp_config import AutoInterpConfig
from featflow.autointerp.pipeline import AutoInterp
import sys
import os
import torch.multiprocessing as mp

# Set environment variables before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing method
mp.set_start_method("spawn", force=True)

def main():
    """Main function to run autointerp with proper multiprocessing support."""
    d_in = 768
    expansion_factor = 32
    d_latent = expansion_factor * d_in

    autointerp_cfg = { 
        "device": "cpu",
        "model_name": "roneneldan/TinyStories-33M", 
        "clt_path": "/home/fdraye/projects/featflow/checkpoints/tiny_stories/z92tu07t/final_10240000",
        "latent_cache_path": "/fast/fdraye/data/featflow/cache",
        "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        "context_size": 16,
        "total_autointerp_tokens": 1_000_000,
        "vllm_model": "meta-llama/Llama-3.1-70B-Instruct",
        "train_batch_size_tokens": 4096,
        "n_batches_in_buffer": 32,
        "store_batch_size_prompts": 32, 
        "d_in": 768
    }

    autointerp_config = AutoInterpConfig(**autointerp_cfg)
    autointerp = AutoInterp(autointerp_config)

    layer = int(sys.argv[1])
    job_id = int(sys.argv[2]) 
    total_jobs = int(sys.argv[3])
    
    # Calculate feature index list for this job
    features_per_job = d_latent // total_jobs
    start_idx = job_id * features_per_job 
    end_idx = start_idx + features_per_job
    index_list = list(range(start_idx, end_idx))
    
    print(f"Job {job_id}/{total_jobs}: Processing layer {layer}, features {start_idx}-{end_idx-1} ({len(index_list)} features)", flush=True)

    print("Running prompt generation", flush=True)
    autointerp.generate_prompts_for_layer(layer=layer, top_k=100, index_list=index_list)

    print("Running explanation generation", flush=True)
    autointerp.generate_explanations_from_prompts(layer=layer, index_list=index_list)
    
    print("Running dictionary generation", flush=True)
    autointerp.generate_feature_dictionaries(layer=layer, index_list=index_list)
                                             
if __name__ == "__main__":
    main()
