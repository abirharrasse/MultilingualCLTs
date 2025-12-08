import torch
from featflow.config.clt_training_runner_config import CLTTrainingRunnerConfig
from featflow.training.activations_store import ActivationsStore
from featflow.load_model import load_model

total_training_steps = 1000
train_batch_size_tokens = 2048
total_training_tokens = train_batch_size_tokens * total_training_steps
lr_decay_steps = total_training_steps // 10
lr_warm_up_steps = 1000
l0_warm_up_steps = 10000

cfg = CLTTrainingRunnerConfig(
    device="cuda",  # will be updated in _ddp_worker
    dtype="torch.bfloat16",
    seed=42,
    n_checkpoints=4,
    checkpoint_path="checkpoints/gpt2",
    logger_verbose=True,
    model_class_name="HookedTransformer",
    model_name="gpt2",
    dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2", # apollo-research/roneneldan-TinyStories-tokenizer-gpt2
    context_size=16, # changed to 16
    from_pretrained_path=None,
    d_in=768,
    expansion_factor=32,
    jumprelu_init_threshold=0.002,
    jumprelu_bandwidth=0.001,
    n_batches_in_buffer=16,
    store_batch_size_prompts=32, # changed to 64
    total_training_tokens=total_training_tokens,
    train_batch_size_tokens=train_batch_size_tokens,
    adam_beta1=0.0,
    adam_beta2=0.999,
    lr=7e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    l0_coefficient=0.0005,
    l0_warm_up_steps=l0_warm_up_steps,
    log_to_wandb=True,
    wandb_project="tiny-stories-clt",
    wandb_id=None,
    wandb_log_frequency=10,
    eval_every_n_wandb_logs=100,
    run_name=None,
    wandb_entity=None,
    ddp=False, 
    fsdp=False
)

model = load_model(
    cfg.model_class_name,
    cfg.model_name,
    device=torch.device(cfg.device), 
    model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
)

activations_store = ActivationsStore(
    model,
    cfg
)

activations_store.generate_and_save_activations(
    path = "/fast/fdraye/data/featflow/activations_gpt2", 
    split_count = 2048, 
    number_of_tokens = 536870912 # 1024^2 x 512
)

print("Finished activations generation and saving.")
