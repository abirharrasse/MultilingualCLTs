import torch
from pathlib import Path
from featflow.training.activations_store import ActivationsStore
from featflow.load_model import load_model
from featflow.config import CLTTrainingRunnerConfig
from llama_clt import CLT
from types import SimpleNamespace

def evaluate_clt_minimal(
    clt_path: str,
    model_name: str, 
    dataset_path: str,
    device: str = "cuda",
    num_batches: int = 100,
    batch_size: int = 32,
    context_size: int = 512
):
    """
    Minimal CLT evaluation: Compare CLT reconstructions with original MLP outputs
    """
    
    print(f"Loading CLT from {clt_path}")
    clt = CLT.load_llama_clt(clt_path, device=device)
    clt.eval()
    
    print(f"Loading model {model_name}")
    model = load_model(
        model_class_name="HookedTransformer", 
        model_name=model_name,
        device=device
    )
    
    # Create minimal config for ActivationsStore
    cfg = CLTTrainingRunnerConfig(
        # Model settings
        model_name=model_name,
        model_class_name="HookedTransformer",
        model_from_pretrained_kwargs={},
        
        # Dataset settings  
        dataset_path=dataset_path,
        cached_activations_path=None,
        is_dataset_tokenized=True,
        
        # Device settings
        device=device,
        dtype="float32",
        context_size=context_size,
        seed=42,
        
        # Batch settings
        store_batch_size_prompts=batch_size,
        train_batch_size_tokens=batch_size * context_size,
        n_batches_in_buffer=4,  # Minimum even number
        
        # CLT parameters
        d_in=clt.d_in,
        d_latent=clt.d_latent,
        cross_layer_decoders=clt.cfg.cross_layer_decoders,
        jumprelu_init_threshold=1e-8,
        jumprelu_bandwidth=0.001,
        normalize_decoder=False,
        
        # Training parameters (required but not used)
        total_training_tokens=1000000,
        lr=1e-3,
        l0_coefficient=0.0,
        dead_penalty_coef=0.0,
        dead_feature_window=1000,
        
        # Checkpoint settings
        checkpoint_path="/tmp/dummy",
        n_checkpoints=1,
        
        # Wandb settings (disabled)
        log_to_wandb=False,
        wandb_project="dummy",
        wandb_entity="dummy", 
        run_name="eval",
        
        # Distributed (disabled)
        ddp=False,
        fsdp=False,
        
        # Optional fields
        from_pretrained_path=None,
    )
    
    print("Setting up ActivationsStore...")
    # Initialize ActivationsStore - this will handle hooks automatically
    activations_store = ActivationsStore(
        model=model,
        cfg=cfg,
        rank=0,
        world_size=1,
        estimated_norm_scaling_factor_in = torch.ones(26, device="cpu"),
        estimated_norm_scaling_factor_out = torch.ones(26, device="cpu") 
    )

    
    print(f"Starting evaluation over {num_batches} batches...")
    print(f"CLT config: {clt.N_layers} layers, d_latent={clt.d_latent}, cross_layer_decoders={clt.cfg.cross_layer_decoders}")
    print(f"Expected input shape: [batch_size, {clt.N_layers}, {clt.d_in}]")
    print(f"Norm scaling factors in: {clt.estimated_norm_scaling_factor_in[:3]}... (showing first 3)")
    print(f"Norm scaling factors out: {clt.estimated_norm_scaling_factor_out[:3]}... (showing first 3)")
    print("-" * 80)
    
    total_mse = 0.0
    total_samples = 0
    layer_mse = torch.zeros(clt.N_layers, device=device)
    
    with torch.no_grad():
        batch_count = 0
        for act_in, act_out in activations_store:
            if batch_count >= num_batches:
                break
            
            # Debug first batch
            if batch_count == 0:
                print(f"First batch shapes:")
                print(f"  act_in:  {act_in.shape} ({act_in.dtype})")
                print(f"  act_out: {act_out.shape} ({act_out.dtype})")
                print(f"  act_in range: [{act_in.min():.3f}, {act_in.max():.3f}]")
                print(f"  act_out range: [{act_out.min():.3f}, {act_out.max():.3f}]")
                print("-" * 40)
                
            # Move to device
            act_in = act_in.to(device)  # [batch_size, n_layers, d_model]
            print(f"  act_in MEANS: {act_in.norm(dim=(1, 2))}")
            act_out = act_out.to(device)  # [batch_size, n_layers, d_model]
            print(f"  act_out MEANS: {act_out.norm(dim=(1, 2))}")
            batch_size_actual = act_in.shape[0]
            
            # Forward through CLT (matching training pattern)
            feat_acts, hidden_pre = clt.encode(act_in)  # [batch_size, n_layers, d_latent]
            act_pred = clt.decode(feat_acts)  # CLT reconstruction
            # act_pred = activations_store.remove_norm_scaling_factor_out(act_pred.to('cpu')).to(device)  # Normalize target
            print(f"  act_pred MEANS: {act_pred.norm(dim=(1, 2))}")

            out_variance = act_out.var().item()
            #Overall MSE (single scalar for the entire batch)
            batch_mse = torch.nn.functional.mse_loss(act_out, act_pred) / out_variance # Simple MSE average
            
            # Option 2: Per-layer MSE for analysis (but not summed for total!)
            mse_loss_tensor = torch.nn.functional.mse_loss(act_out, act_pred, reduction="none") / out_variance
            # mse_loss_tensor shape: [batch_size, n_layers, d_model]
            
            # Per-layer MSE: average over batch and feature dimensions for each layer
            mse_loss_per_layer = mse_loss_tensor.mean(dim=(0, 2))  # [n_layers]
            
            # CORRECT ACCUMULATION
            # For overall MSE: use the simple batch MSE (not sum of layer MSEs!)
            total_mse += batch_mse.item() * batch_size_actual
            
            # For per-layer analysis: accumulate layer-wise MSE separately
            layer_mse += mse_loss_per_layer * batch_size_actual
            
            total_samples += batch_size_actual
            
            batch_count += 1
            
            # Debug first few batches
            if batch_count <= 3:
                print(f"Batch {batch_count} debug:")
                print(f"  Batch MSE (correct): {batch_mse:.6f}")
                print(f"  Sum of layer MSEs (wrong): {mse_loss_per_layer.sum():.6f}")
                print(f"  Ratio (sum/correct): {mse_loss_per_layer.sum()/batch_mse:.2f}x")
                print(f"  act_pred range: [{act_pred.min():.3f}, {act_pred.max():.3f}]")
                
            # Print progress more frequently for debugging
            if batch_count % 5 == 0 or batch_count <= 10:
                # CORRECT explained variance calculation
                current_avg_mse = total_mse / total_samples
                
                # For explained variance, we need the variance of the target
                target_variance = act_out.var().item() if batch_count == 1 else 1.0  # Approximate
                current_explained_var = max(0, 1 - current_avg_mse / target_variance)
                
                print(f"Batch {batch_count:3d}/{num_batches} | "
                      f"Current MSE: {current_avg_mse:.6f} | "
                      f"Explained Var: {current_explained_var:.4f} ({current_explained_var*100:.2f}%) | "
                      f"Samples: {total_samples:,}")
                
                # Print per-layer MSE every 20 batches for detailed debugging
                if batch_count % 20 == 0 and batch_count > 0:
                    current_layer_mse = layer_mse / total_samples
                    print(f"    Per-layer MSE (first 8 layers): {[f'{mse:.4f}' for mse in current_layer_mse[:8].cpu().tolist()]}")
                    if clt.N_layers > 8:
                        print(f"    Per-layer MSE (last 8 layers):  {[f'{mse:.4f}' for mse in current_layer_mse[-8:].cpu().tolist()]}")
                    print(f"    Best/Worst layers: {current_layer_mse.argmin().item()}/{current_layer_mse.argmax().item()} "
                          f"(MSE: {current_layer_mse.min():.4f}/{current_layer_mse.max():.4f})")
                    
                    # Verify: sum of layer MSEs vs overall MSE
                    layer_sum = current_layer_mse.sum().item()
                    print(f"    Layer MSE sum: {layer_sum:.4f} | Overall MSE: {current_avg_mse:.6f} | Ratio: {layer_sum/current_avg_mse:.1f}x")
                    print("    " + "-"*50)
    
    # Calculate final metrics CORRECTLY
    avg_mse = total_mse / total_samples  # This is the CORRECT overall MSE
    avg_layer_mse = layer_mse / total_samples  # Per-layer MSE for analysis
    
    # For explained variance, compute target variance properly
    # Note: This is approximate since we don't store all targets
    explained_variance = max(0, 1 - avg_mse)  # Assuming unit variance target
    
    print("\n" + "="*60)
    print("CLT EVALUATION RESULTS (CORRECTED)")
    print("="*60)
    print(f"Total samples: {total_samples:,}")
    print(f"Mean Squared Error (CORRECT): {avg_mse:.6f}")
    print(f"Root Mean Squared Error: {avg_mse**0.5:.6f}")
    print(f"Explained Variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # Verification
    layer_sum = avg_layer_mse.sum().item()
    print(f"\nVerification:")
    print(f"Sum of per-layer MSEs: {layer_sum:.6f}")
    print(f"Overall MSE: {avg_mse:.6f}")
    print(f"Ratio (sum/overall): {layer_sum/avg_mse:.1f}x")
    print("(Ratio should be ~16 for 16 layers if layers have similar variance)")
    
    print(f"\nPer-layer MSE:")
    for i, mse in enumerate(avg_layer_mse):
        print(f"  Layer {i:2d}: {mse:.6f}")
    print("="*60)
    
    return {
        "mse": avg_mse,
        "layer_mse": avg_layer_mse.cpu().tolist(),
        "explained_variance": explained_variance,
        "total_samples": total_samples
    }


if __name__ == "__main__":
    results = evaluate_clt_minimal(
        clt_path="/home/abir19/clt-gemma-2-2b-426k",
        model_name="google/gemma-2-2b", 
        dataset_path="chanind/openwebtext-gemma", #"chanind/openwebtext-llama3",  # or path to your tokenized dataset
        device="cuda",
        num_batches=10,
        batch_size=1,  # Adjust based on GPU memory
        context_size=32
    )
    
    print(f"\nFinal MSE: {results['mse']:.6f}")
    print(f"Explained Variance: {results['explained_variance']:.4f}")