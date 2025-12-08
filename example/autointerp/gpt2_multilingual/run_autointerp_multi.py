# Set all environment variables FIRST, before ANY imports
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_USE_MODELSCOPE"] = "0"
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import sys
import torch
import torch.multiprocessing as mp
from featflow.config.autointerp_config import AutoInterpConfig  
from featflow.autointerp.pipeline import AutoInterp

# Check CUDA before any other imports
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Set multiprocessing method
mp.set_start_method("spawn", force=True)

def main():
    """Main function to run autointerp with proper multiprocessing support."""
    d_in = 768
    expansion_factor = 32
    d_latent = expansion_factor * d_in

    autointerp_cfg = { 
        "device": "cpu",
        "model_name": "CausalNLP/gpt2-hf_multilingual-70", 
        "clt_path": "/home/fdraye/projects/featflow/checkpoints/gpt2_multilingual/y4ck9q08/final_25600000",
        "latent_cache_path": "/fast/fdraye/data/featflow/cache/gpt2-hf_multilingual-70-new",
        "dataset_path": "abir-hr196/clt_gpt2_tokenized",
        "context_size": 16,
        "total_autointerp_tokens": 36*(60*4096), #2*(60*4096)
        "vllm_model": "google/gemma-2-9b-it",
        "train_batch_size_tokens": 4096,
        "n_batches_in_buffer": 32,
        "store_batch_size_prompts": 32, 
        "d_in": 768, 
        "n_chunks": 36 #36
    }

    autointerp_config = AutoInterpConfig(**autointerp_cfg)
    autointerp = AutoInterp(autointerp_config)

    layer = int(sys.argv[1])
    job_id = int(sys.argv[2]) 
    total_jobs = int(sys.argv[3])
    # chunk_id = int(sys.argv[4])

    # Calculate feature index list for this job
    features_per_job = d_latent // total_jobs
    start_idx = job_id * features_per_job 
    end_idx = start_idx + features_per_job
    index_list = list(range(start_idx, end_idx))
    
    print(f"Job {job_id}/{total_jobs}: Processing layer {layer}, features {start_idx}-{end_idx-1} ({len(index_list)} features)", flush=True)

    # chunk_list = list(range(3*chunk_id, 3*chunk_id +3))
    # # chunk_list = [chunk_id]
    # print("Generate Cache", flush=True)
    # autointerp.run(chunk_list)

    # Once you have run the cache, you can run the next three functions with queue 96 or less. If you have cuda memory issues when running the explanation, run it seperatly. 
    print("Running prompt generation", flush=True)
    autointerp.generate_prompts_for_layer(layer=layer, top_k=100, index_list=index_list)
 
    print("Running explanation generation", flush=True)
    autointerp.generate_explanations_from_prompts(layer=layer, index_list=index_list)
    
    print("Running dictionary generation", flush=True)
    autointerp.generate_feature_dictionaries(layer=layer, index_list=index_list)
                                             
if __name__ == "__main__":
    main()
