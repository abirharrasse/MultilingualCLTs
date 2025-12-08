import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from featflow.config.clt_training_runner_config import CLTTrainingRunnerConfig
from featflow.clt_training_runner import CLTTrainingRunner
import traceback
import sys

def _ddp_worker(rank: int, world_size: int, cfg: CLTTrainingRunnerConfig):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        cfg = cfg.copy(deep=True)
        cfg.ddp = True

        trainer = CLTTrainingRunner(cfg, rank=rank, world_size=world_size)
        trainer.run()

        dist.destroy_process_group()

    except Exception:
        error_filename = "/home/fdraye/projects/featflow/example/tiny_stories/logs/error.txt"
        with open(error_filename, "w") as f:
            f.write(f"Exception in rank {rank}:\n")
            f.write(traceback.format_exc())
        print(f"Rank {rank} crashed. See {error_filename} for details.", file=sys.stderr)
        raise  # Re-raise to propagate the error back to mp.spawn

def main():
    world_size = torch.cuda.device_count()
    assert world_size >= 2, "Need at least 2 GPUs for DDP"

    total_training_steps = 20_000 // world_size # 125k steps at 16k batch size
    train_batch_size_tokens = world_size * 2048
    total_training_tokens = train_batch_size_tokens * total_training_steps # 2B tokens
    lr_decay_steps = 1
    final_lr_scale = 1.
    lr_warm_up_steps = 1
    l0_waiting_steps = 0
    l0_warm_up_steps = 1
    decay_stable_steps = 0

    # functional loss 
    functional_loss = "kl"
    fc_coefficient = 100
    fc_warm_up_steps = 1
    fc_waiting_steps = 0

    cfg = CLTTrainingRunnerConfig(
        from_pretrained_path="/home/fdraye/projects/featflow/checkpoints/tiny_stories/z92tu07t/final_10240000",
        device="cuda",  # will be updated in _ddp_worker
        dtype="float32",
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints/tiny_stories",
        logger_verbose=True,
        model_class_name="HookedTransformer",
        model_name="roneneldan/TinyStories-33M",
        context_size=16, # including the bos at the beginning of each sequence
        d_in=768,
        expansion_factor=64,
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.,
        cached_activations_path="/fast/fdraye/data/featflow/activations_tiny_stories_33M",
        n_train_batch_per_buffer=100, # should find the optimal value
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=2e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        final_lr_scale=final_lr_scale,
        l0_coefficient=0.8,
        dead_penalty_coef=1e-4,
        dead_feature_window=250,
        l0_warm_up_steps=l0_warm_up_steps,
        l0_waiting_steps=l0_waiting_steps,
        decay_stable_steps=decay_stable_steps,
        log_to_wandb=True,
        wandb_project="tiny-stories-clt",
        wandb_id=None,
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100,
        run_name=None,
        wandb_entity=None,
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
