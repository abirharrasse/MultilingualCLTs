import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from featflow.config.clt_training_runner_config import CLTTrainingRunnerConfig
from featflow.clt_training_runner import CLTTrainingRunner

def _ddp_worker(rank: int, world_size: int, cfg: CLTTrainingRunnerConfig):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
        # Add these environment variables for better debugging
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        
        # Initialize with shorter timeout for faster debugging
        from datetime import timedelta
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=timedelta(minutes=10)  # Reduce from default 30 minutes
        )

        cfg = cfg.copy(deep=True)

        trainer = CLTTrainingRunner(cfg, rank=rank, world_size=world_size)
        trainer.run()

    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    assert world_size >= 2, "Need at least 2 GPUs for DDP"

    total_training_steps = 300_000 // world_size # 125k steps at 16k batch size
    train_batch_size_tokens = world_size * 1024
    total_training_tokens = train_batch_size_tokens * total_training_steps # 2B tokens
    lr_decay_steps = (total_training_steps // 20) - 1
    final_lr_scale = 1.
    lr_warm_up_steps = 10
    l0_waiting_steps = 0
    l0_warm_up_steps = int(0.7 * total_training_steps) - l0_waiting_steps - 1 # warm up until the end of training
    decay_stable_steps = total_training_steps - l0_warm_up_steps - lr_decay_steps

    # functional loss 
    functional_loss = "kl"
    fc_coefficient = 0
    fc_warm_up_steps = 1000
    fc_waiting_steps = total_training_steps - fc_warm_up_steps - 1

    cfg = CLTTrainingRunnerConfig(
        device="cuda",  # will be updated in _ddp_worker
        dtype="float32",
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints/gpt2",
        logger_verbose=True,
        model_class_name="HookedTransformer",
        model_name="gpt2",
        context_size=16, # including the bos at the beginning of each sequence
        from_pretrained_path=None,
        d_in=768,
        expansion_factor=32,
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.,
        cached_activations_path="/fast/fdraye/data/featflow/activations_gpt2",
        n_train_batch_per_buffer=100, # should find the optimal value
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=2e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        final_lr_scale=final_lr_scale,
        l0_coefficient=1.,
        dead_penalty_coef=3e-4,
        dead_feature_window=250,
        l0_warm_up_steps=l0_warm_up_steps,
        l0_waiting_steps=l0_waiting_steps,
        decay_stable_steps=decay_stable_steps,
        log_to_wandb=True,
        wandb_project="gpt2-clt",
        wandb_id=None,
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100,
        run_name=None,
        wandb_entity=None,
        fsdp=False, 
        ddp=True,
        functional_loss=functional_loss,
        fc_coefficient=fc_coefficient,
        fc_warm_up_steps=fc_warm_up_steps,
        fc_waiting_steps=fc_waiting_steps
    )

    # ------------------ LAUNCH DDP -------------------
    mp.spawn(_ddp_worker, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()