from featflow.config.clt_training_runner_config import CLTTrainingRunnerConfig
from featflow.clt_training_runner import CLTTrainingRunner

total_training_steps = 100_000
train_batch_size_tokens = 2048
total_training_tokens = train_batch_size_tokens * total_training_steps
lr_decay_steps = total_training_steps // 10
lr_warm_up_steps = 1000
l0_warm_up_steps = 10000
l0_waiting_steps = 0

cfg = CLTTrainingRunnerConfig(
    device="cuda",  # will be updated in _ddp_worker
    dtype="float32",
    seed=42,
    n_checkpoints=4,
    checkpoint_path="checkpoints/tiny_stories",
    logger_verbose=True,
    model_class_name="HookedTransformer",
    model_name="roneneldan/TinyStories-33M",
    context_size=32,
    from_pretrained_path=None,
    d_in=768,
    expansion_factor=16,
    jumprelu_init_threshold=0.03,
    jumprelu_bandwidth=1.,
    cached_activations_path="/fast/fdraye/data/featflow/activations_tiny_stories_33M",
    n_train_batch_per_buffer=50, # should find the optimal value
    total_training_tokens=total_training_tokens,
    train_batch_size_tokens=train_batch_size_tokens,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr=1e-4,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    l0_coefficient=0.0005,
    l0_warm_up_steps=l0_warm_up_steps,
    l0_waiting_steps=l0_waiting_steps,
    log_to_wandb=True,
    wandb_project="tiny-stories-clt",
    wandb_id=None,
    wandb_log_frequency=10,
    eval_every_n_wandb_logs=100,
    run_name=None,
    wandb_entity=None,
    ddp=False
)

trainer = CLTTrainingRunner(cfg)
trainer.run()
