import time
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from pathlib import Path
import wandb
from featflow.clt_training_runner import CLTTrainingRunner
from featflow.training.activations_store import ActivationsStore
from featflow.load_model import load_model

from tests.utils import build_clt_training_runner_cfg
import os

def _ddp_worker(rank: int, world_size: int, cfg_base):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfg = copy.deepcopy(cfg_base)
    cfg.ddp = True

    runner = CLTTrainingRunner(cfg, rank=rank, world_size=world_size)

    torch.cuda.synchronize()
    start_time = time.time()
    runner.run()
    torch.cuda.synchronize()
    end_time = time.time()

    if rank == 0:
        print(f"[DDP RANK {rank}] Duration: {end_time - start_time:.2f} s")

    dist.destroy_process_group()

def _single_gpu_run(cfg):
    runner = CLTTrainingRunner(cfg)

    torch.cuda.synchronize()
    start_time = time.time()
    runner.run()
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"[Single GPU] Duration: {end_time - start_time:.2f} s")
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print(f" After cleanup - Allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print(f" After cleanup - Reserved : {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need ≥2 GPUs for DDP test")
def test_compare_single_gpu_vs_ddp(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)

    cfg = build_clt_training_runner_cfg(
        model_name="roneneldan/TinyStories-33M",
        dataset_path=str(Path(__file__).parent / "data/NeelNanda_c4_10k_tokenized"),
        d_in=768,
        device="cuda",  ### need to make it for gpu 
        total_training_tokens=100_000,
        train_batch_size_tokens=4096,
        log_to_wandb=False,
        checkpoint_path=str(tmp_path / "ckpts_single"),
    )

    model = load_model(
        cfg.model_class_name,
        cfg.model_name,
        device=torch.device(cfg.device), 
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

    # generate
    store = ActivationsStore(model, cfg)
    store.generate_and_save_activations(
        path = str(tmp_path), 
        split_count = 8,
        number_of_tokens = 100_000
    )

    # load 
    cfg.cached_activations_path = str(tmp_path)
    cfg.n_train_batch_per_buffer = 2

    print("\n Running on 1 GPU...")
    cfg_single = copy.deepcopy(cfg)
    cfg_single.ddp = False
    _single_gpu_run(cfg_single)

    print("\n Running on 2 GPUs (DDP)...")
    cfg_ddp = copy.deepcopy(cfg)
    cfg_ddp.checkpoint_path = str(tmp_path / "ckpts_ddp")
    mp.spawn(
        _ddp_worker,
        args=(4, cfg_ddp),
        nprocs=4,
        join=True,
    )
    assert False

# def test_activations_store(monkeypatch): 
#     # test that each worker has its own activations 
#     # test that each activation store is independent data, and they correspond to their write part of the data
#     # test the shapes of what is going on 

# def test_training(monkeypatch): 
#     # test that 
