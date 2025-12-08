# clt_stack_batched.py

import math
import torch
import torch.nn as nn
from jaxtyping import Float
from pathlib import Path

from featflow.config import CLTConfig
from featflow.utils import DTYPE_MAP, CLT_WEIGHTS_FILENAME, CLT_CFG_FILENAME
from featflow.training.optim import JumpReLU
from safetensors.torch import save_file, load_file
import json
from pydantic import BaseModel, ConfigDict
from typing import Union
from featflow.transformer_lens.hooked_transformer_wrapper import patch_transformer_lens
from typing import Optional, Dict
from featflow import logger
from featflow.load_model import load_model
from contextlib import contextmanager
from transformer_lens.hook_points import HookedRootModule
import yaml 

C_l0_COEF = 4

class LossMetrics(BaseModel):
    act_in: torch.Tensor
    act_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    mse_loss: torch.Tensor 
    l0_loss: torch.Tensor
    dead_feature_loss: torch.Tensor
    mse_loss_accross_layers: torch.Tensor
    l0_loss_accross_layers: torch.Tensor
    l0_loss_replacement: torch.Tensor = torch.tensor(float('-inf'))
    l0_accross_layers_replacement: Optional[torch.Tensor] = None
    act_pred: torch.Tensor
    hybrid_loss: Optional[torch.Tensor] = torch.tensor(float('-inf')) # for wandb
    pred_per: Optional[float] = torch.zeros(32)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class CLT(nn.Module):
    """
    * pytorch module for a cross layer transcoder
    * can take an LLM as attribute and compute replacement model forward pass
    """

    def __init__(self, cfg: CLTConfig):
        super().__init__()

        self.cfg = cfg
        self.N_layers = cfg.n_layers
        self.d_in = cfg.d_in
        self.d_latent = cfg.d_latent
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)

        init_device = self.device if not cfg.fsdp else torch.device("cpu")

        self.N_layers_out = torch.tensor(
            [cfg.n_layers - (i + 1) for i in range(self.N_layers)],
            dtype=torch.long,
            device=self.device,
        )
        self.max_layers_out = int(self.N_layers_out.max().item())

        self.W_enc = nn.Parameter(torch.empty(self.N_layers, self.d_in, self.d_latent, dtype=self.dtype, device=init_device))
        self.b_enc = nn.Parameter(torch.zeros(self.N_layers, self.d_latent, dtype=self.dtype, device=init_device))

        if cfg.cross_layer_decoders:
            self.N_dec = self.N_layers * (self.N_layers + 1) // 2
            self.W_dec = nn.Parameter(torch.empty(self.N_dec, self.d_latent, self.d_in, dtype=self.dtype, device=init_device))
            self.b_dec = nn.Parameter(torch.zeros(self.N_dec, self.d_in, dtype=self.dtype, device=init_device))

            l_idx, k_idx = torch.triu_indices(self.N_layers, self.N_layers, offset=0,
                                            device=init_device)
            self.register_buffer('l_idx', l_idx, persistent=False)   # [K]
            self.register_buffer('k_idx', k_idx, persistent=False)   # [K]

            layer_mask = torch.zeros(self.N_layers, self.N_dec, device=init_device, dtype=self.dtype)
            for layer in range(self.N_layers):
                layer_mask[layer, l_idx == layer] = 1
            self.register_buffer('layer_mask', layer_mask)

        else: 
            self.W_dec = nn.Parameter(torch.empty(self.N_layers, self.d_latent, self.d_in, dtype=self.dtype, device=init_device))
            self.b_dec = nn.Parameter(torch.zeros(self.N_layers, self.d_in, dtype=self.dtype, device=init_device))

        self.log_threshold = nn.Parameter(
            torch.full((self.N_layers, self.d_latent), math.log(cfg.jumprelu_init_threshold), dtype=self.dtype, device=init_device)
        )
        self.bandwidth = cfg.jumprelu_bandwidth

        self.register_buffer('feature_count', 
            torch.zeros(
                self.N_layers, 
                self.d_latent, 
                dtype=torch.long, 
                device=init_device
            )
        )

        self.hybrid_forward = lambda input_tokens, return_feat_acts=False, return_sparse=True: (_ for _ in ()).throw(
                NotImplementedError("hybrid_forward should be created by attach_model_for_replacement method.")
        )

        self._initialize()

        self.register_buffer('estimated_norm_scaling_factor_in', torch.ones(self.N_layers, device=self.device))
        self.register_buffer('estimated_norm_scaling_factor_out', torch.ones(self.N_layers, device=self.device))

    def _initialize(self) -> None:
        # Anthropic guidelines
        # encoder:  U(-1/n_features,  +1/n_features)
        enc_lim = 1.0 / self.d_latent**0.5
        for W in self.W_enc:
            nn.init.uniform_(W, -enc_lim, enc_lim)

        # decoder: U(-1/(n_layers*d_model), +1/(n_layers*d_model))
        dec_lim = 1.0 / (self.N_layers * self.d_in)**0.5
        nn.init.uniform_(self.W_dec, -dec_lim, dec_lim)

    def _initialize_b_enc(self, x: Float[torch.Tensor, "..."], rate: float = 0.1) -> None: 
        """
        Initialize b_enc by examining a subset of the data and picking a constant per feature
        such that each feature activates at a certain rate.
        x: [B, N_layers, d_in]
        """
        with torch.no_grad():
            # Compute pre-activations without bias
            hidden_pre = torch.einsum(
                "bnd,ndk->bnk",
                x,
                self.W_enc,
            )  # [B, N_layers, d_latent]
            
            thresh = torch.exp(self.log_threshold) 
            target_activation_rate = rate
            
            # For each layer and feature, find the bias that gives target activation rate
            B = hidden_pre.shape[0]
            bias_values = torch.zeros_like(self.b_enc)
            
            for layer in range(self.N_layers):
                for feature in range(self.d_latent):
                    feature_pre_acts = hidden_pre[:, layer, feature]  # [B]
                    sorted_acts, _ = torch.sort(feature_pre_acts, descending=True)
                    target_idx = int(target_activation_rate * B) + 1
                    threshold_value = sorted_acts[target_idx]
                    required_bias = thresh[layer, feature] - threshold_value
                    
                    bias_values[layer, feature] = required_bias
            
            self.b_enc.data = bias_values
            print(f"Initialized b_enc with target activation rate {target_activation_rate:.6f}")
            
            # # Verify the initialization by computing actual activation rates
            # feat_act, _ = self.encode(x)            
            # activation_rates = (feat_act > 0).bfloat16().mean(dim=0)  # [N_layers, d_latent]
            # avg_activation_rate = activation_rates.mean().item()
            
            # print(f"Actual average activation rate: {avg_activation_rate * self.d_latent:.0f}")
            # print(f"Expected ~{self.d_latent * target_activation_rate:.0f} ")

    def encode(
        self,
        x: Float[torch.Tensor, "..."],
        layer: Optional[int] = None
    ) -> tuple[
        Float[torch.Tensor, "..."],
        Float[torch.Tensor, "..."],
    ]:
        """
        x: [B, N_layers, d_in] if layer is None, else [B, d_in]
        output: tuple([B, N_layers, d_latent], [B, N_layers, d_latent]) if layer is None, else [B, d_latent]
        """

        if layer is None: 
            hidden_pre = torch.einsum(
                "bnd,ndk->bnk",
                x,
                self.W_enc,
            ) + self.b_enc

            thresh = torch.exp(self.log_threshold) #shape [N_layers, d_latent]
        else: 
            assert 0 <= layer < self.N_layers, f"Layer {layer} out of range"
            hidden_pre = x @ self.W_enc[layer] + self.b_enc[layer]
            thresh = torch.exp(self.log_threshold[layer]) 
        
        feat_act = JumpReLU.apply(hidden_pre, thresh, self.bandwidth)
        return feat_act, hidden_pre

    def decode(
        self,
        z: Float[torch.Tensor, "..."],
        layer: Optional[int] = None
    ) -> Float[torch.Tensor, "..."]:
        """
        z: [B, N_layers, d_latent] if layer is None, else [B, d_latent]
        output: [B, N_layers, d_in] if layer is None, else [B, N_layers_out, d_in]
        """

        if layer is None:
            if self.cfg.cross_layer_decoders:
                B = z.shape[0]
                z_sel = z.index_select(1, self.l_idx) # [B, K, d_latent] 

                contrib = torch.einsum(
                    'bkd,kdf->bkf',                 
                    z_sel,
                    self.W_dec
                ) + self.b_dec # [B, K, d_out]
                
                out = torch.zeros(B, self.N_layers, self.d_in,
                                dtype=self.dtype, device=self.device)
                out = out.index_add(1, self.k_idx, contrib)
            else: 
                out = torch.einsum("bnk,nkd->bnd", z, self.W_dec) + self.b_dec
        else: 
            assert 0 <= layer < self.N_layers, f"Layer {layer} out of range"
            if self.cfg.cross_layer_decoders:
                indices = (self.l_idx == layer).nonzero(as_tuple=True)[0]
                
                z_layer = z.unsqueeze(1)  # [B, 1, d_latent]
                z_layer = z_layer.expand(-1, len(indices), -1)  # [B, num_decoders, d_latent]
                
                W_dec_layer = self.W_dec[indices]  # [num_decoders, d_latent, d_in]
                b_dec_layer = self.b_dec[indices]  # [num_decoders, d_in]

                out = torch.einsum(
                    'bkd,kdf->bkf',                 
                    z_layer,
                    W_dec_layer
                ) + b_dec_layer # [B, num_decoders, d_in]

            else: 
                out = z @ self.W_dec[layer] + self.b_dec[layer] # [B, d_out]
        return out
    # def decode(
    #     self,
    #     z: Float[torch.Tensor, "..."],
    #     layer: Optional[int] = None
    # ) -> Float[torch.Tensor, "..."]:
    #     """
    #     z: [B, N_layers, d_latent] if layer is None, else [B, d_latent]
    #     output: [B, N_layers, N_layers, d_in] if layer is None and cross_layer_decoders=True
    #             [B, N_layers, d_in] if layer is None and cross_layer_decoders=False
    #             [B, N_layers_out, d_in] if layer is not None
    #     """

    #     if layer is None:
    #         if self.cfg.cross_layer_decoders:
    #             B = z.shape[0]
    #             z_sel = z.index_select(1, self.l_idx) # [B, K, d_latent] 

    #             contrib = torch.einsum(
    #                 'bkd,kdf->bkf',                 
    #                 z_sel,
    #                 self.W_dec
    #             ) + self.b_dec # [B, K, d_out]
                
    #             # Create 4D output tensor for cross-layer structure
    #             out = torch.zeros(B, self.N_layers, self.N_layers, self.d_in,
    #                             dtype=self.dtype, device=self.device)
                
    #             # Fill the triangular structure properly
    #             for k in range(len(self.l_idx)):
    #                 enc_layer = self.l_idx[k].item()      # Which encoding layer
    #                 target_layer = self.k_idx[k].item()   # Which target layer
    #                 out[:, enc_layer, target_layer, :] = contrib[:, k, :]
                    
    #             return out  # [B, N_layers, N_layers, d_in]
    #         else: 
    #             out = torch.einsum("bnk,nkd->bnd", z, self.W_dec) + self.b_dec
    #             return out  # [B, N_layers, d_in]
    #     else: 
    #         assert 0 <= layer < self.N_layers, f"Layer {layer} out of range"
    #         if self.cfg.cross_layer_decoders:
    #             indices = (self.l_idx == layer).nonzero(as_tuple=True)[0]
                
    #             z_layer = z.unsqueeze(1)  # [B, 1, d_latent]
    #             z_layer = z_layer.expand(-1, len(indices), -1)  # [B, num_decoders, d_latent]
                
    #             W_dec_layer = self.W_dec[indices]  # [num_decoders, d_latent, d_in]
    #             b_dec_layer = self.b_dec[indices]  # [num_decoders, d_in]

    #             out = torch.einsum(
    #                 'bkd,kdf->bkf',                 
    #                 z_layer,
    #                 W_dec_layer
    #             ) + b_dec_layer # [B, num_decoders, d_in]

    #         else: 
    #             out = z @ self.W_dec[layer] + self.b_dec[layer] # [B, d_out]
    #     return out

    def forward_eval(
        self,
        x: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
        """
        x: [N, ..., d_in]
        Returns: z and reconstruction
        """
        z, _ = self.encode(x)
        recon = self.decode(z)
        return recon

    def forward(
        self,
        act_in:  torch.Tensor,
        act_out: torch.Tensor,
        l0_coef: float,
        df_coef: float,
        return_metrics: bool = True,
        input_tokens: Optional[torch.Tensor] = None,
        fl_coef: float = 1.0 # functional loss coefficient
    ):
        """
        Wrapper forward function for DDP.
        """

        # renormalize decoder, should normally not be used
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        metrics = self.loss(act_in, act_out, l0_coef, df_coef)
        loss = metrics.mse_loss + metrics.l0_loss + metrics.dead_feature_loss

        if input_tokens is not None: 
            hybrid_loss, pred_per_pos, l0_loss, l0_accross_layers = self.hybrid_loss(input_tokens, l0_coef=l0_coef)
            metrics.hybrid_loss = fl_coef * hybrid_loss
            metrics.pred_per = pred_per_pos
            metrics.l0_loss_replacement = l0_loss
            metrics.l0_accross_layers_replacement = l0_accross_layers
            loss += metrics.hybrid_loss + metrics.l0_loss_replacement

        return (loss, metrics) if return_metrics else loss

    def loss(self, act_in: torch.Tensor, act_out: torch.Tensor, l0_coef: float, df_coef: float) -> LossMetrics:
        feat_act, hidden_pre = self.encode(act_in)
        act_pred = self.decode(feat_act)

        ### MSE loss
        mse_loss_tensor = torch.nn.functional.mse_loss(act_out, act_pred, reduction="none")
        mse_loss_accross_layers = mse_loss_tensor.sum(dim=-1).mean(dim=0)
        mse_loss = mse_loss_accross_layers.sum()

        ### L0 regularization
        if self.cfg.cross_layer_decoders:
            squared_norms = (self.W_dec**2).sum(dim=2)
            feature_norms = torch.sqrt(torch.matmul(self.layer_mask, squared_norms)) # [N_layers, d_latent]
        else: 
            feature_norms = self.W_dec.norm(dim=2) # [N_layers, d_latent]
        
        weighted_activations = feat_act * feature_norms # [batch_size, N_layers, d_latent]
        tanh_weighted_activations = torch.tanh(C_l0_COEF * weighted_activations)  # [batch_size, N_layers, d_latent]
        l0_loss_accross_layers = l0_coef * tanh_weighted_activations.sum(dim=-1).mean(dim=0)  # [N_layers]
        l0_loss = l0_loss_accross_layers.sum()

        ### Dead feature penalty
        dead_feature_loss = df_coef * torch.relu(torch.exp(self.log_threshold)-hidden_pre) * feature_norms
        dead_feature_loss = dead_feature_loss.sum(dim=-1).mean(dim=0).sum()

        ### Dead feature count
        with torch.no_grad(): 
            firing = feat_act.sum(dim=0) > 0 # [N_layers, d_latent]
            self.feature_count += 1
            self.feature_count[firing] = 0

        return LossMetrics(
            act_in=act_in,
            act_out=act_out,
            feature_acts=feat_act,
            hidden_pre=hidden_pre,
            act_pred=act_pred,
            mse_loss=mse_loss,
            l0_loss=l0_loss, 
            dead_feature_loss=dead_feature_loss,
            mse_loss_accross_layers=mse_loss_accross_layers,
            l0_loss_accross_layers=l0_loss_accross_layers
        )
    
    @torch.no_grad()
    def get_dead_features(self) -> torch.Tensor:
        return self.feature_count > self.cfg.dead_feature_window # [N_layers, d_latent]

    def save_model(self, path_str: str, state_dict_: Optional[Dict] = None):
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        
        state_dict = self.state_dict()

        # Remove any keys that start with 'model.' (the attached transformer model)
        clt_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.')}

        save_file(clt_state_dict, path / CLT_WEIGHTS_FILENAME)

        cfg_dict = self.cfg.to_dict()
        
        cfg_path = path / CLT_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(cfg_dict, f)

        return cfg_path
    
    @classmethod
    def load_from_pretrained(cls, path: Union[str, Path], device: str, model_name: Optional[str] = "gpt2") -> "CLT":
        path = Path(path)
        cfg_path = path / CLT_CFG_FILENAME
        weights_path = path / CLT_WEIGHTS_FILENAME

        with cfg_path.open("r") as f:
            cfg_dict = json.load(f)

        layer_dict = {
            "gpt2": 12,
            "roneneldan/TinyStories-33M": 4,
            "CausalNLP/gpt2-hf_multilingual-70": 12, 
            "CausalNLP/gpt2-hf_multilingual-20": 12,
            "tiny-stories-1M": 1
        }

        cfg_dict["n_layers"] = layer_dict[cfg_dict["model_name"]]
        cfg_dict["device"] = device
        cfg = CLTConfig.from_dict(cfg_dict)

        clt = cls(cfg)
        state_dict = load_file(weights_path, device=device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.')}
        missing, unexpected = clt.load_state_dict(state_dict, strict=False)

        if missing or unexpected:
            raise RuntimeError(f"Incompatible checkpoint.\n  missing: {missing}\n  unexpected: {unexpected}")

        clt.to(torch.device(device))
        return clt
    

    @classmethod
    def load_llama_clt(cls, path: Union[str, Path], device: str) -> "CLT":
        """Load Llama-format CLT weights with cross-layer decoders"""
        path = Path(path)

        # Load config
        with open(path / "config.yaml", 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Get dimensions from first file
        sample = load_file(path / "W_enc_0.safetensors")
        d_latent, d_in = sample['W_enc_0'].shape  # [32768, 2048]
        n_layers = len(list(path.glob("W_enc_*.safetensors")))

        print(f"Loading Llama CLT: {n_layers} layers, d_in={d_in}, d_latent={d_latent}")

        # Create CLTConfig with all required fields
        cfg = CLTConfig(
            model_name=yaml_config["model_name"],
            n_layers=n_layers,
            d_in=d_in,
            d_latent=d_latent,
            cross_layer_decoders=True,
            device=device,
            dtype="float32",
            jumprelu_init_threshold=1e-8,  # to check
            jumprelu_bandwidth=0.001,       #to check
            normalize_decoder=False,
            dead_feature_window=1000,
            fsdp=False,
            seed=42,
            context_size=32,
            l0_coefficient=0.0,
            ddp=False,
            functional_loss="kl"
        )

        # Create CLT instance
        clt = cls(cfg)

        # Load the actual weights
        with torch.no_grad():
            for i in range(n_layers):
                print(f"Loading layer {i}...")

                # Load encoder weights
                enc_data = load_file(path / f"W_enc_{i}.safetensors", device=device)
                clt.W_enc.data[i] = enc_data[f'W_enc_{i}'].T  # [32768,2048] -> [2048,32768]
                clt.b_enc.data[i] = enc_data[f'b_enc_{i}']
                print("ENCODER NORM", clt.W_enc.data[i].norm(dim=1).mean().item())
                print('ENCODER BIAS NORM', clt.b_enc.data[i].norm().mean().item())
                # Load thresholds
                clt.log_threshold.data[i] = torch.log(torch.tensor(1e-8)) #torch.log(enc_data[f'threshold_{i}'] + 1e-8)

                # Load decoder weights and map to triangular structure
                dec_data = load_file(path / f"W_dec_{i}.safetensors", device=device)
                dec_weights = dec_data[f'W_dec_{i}']  # [d_latent, N_layers-i, d_in]
                print("DECODER NORM", dec_weights.norm(dim=(1, 2)).mean().item())
                # For layer i, dec_weights covers target layers [i, i+1, ..., N_layers-1]
                # The shape is [d_latent, N_layers-i, d_in]
                print(f"  W_dec_{i} shape: {dec_weights.shape}, expected: [{d_latent}, {n_layers-i}, {d_in}]")

                # Find all decoder indices in the triangular structure for this encoding layer
                layer_decoder_indices = (clt.l_idx == i).nonzero(as_tuple=True)[0]
                print(f'Layer decoder indices for layer {i}: {layer_decoder_indices}')

                for j, decoder_idx in enumerate(layer_decoder_indices):
                    # Get the target layer for this decoder
                    target_layer = clt.k_idx[decoder_idx].item()
                    
                    # Convert absolute target layer to relative index in dec_weights
                    # For encoding layer i, target layers start from i
                    relative_target_layer = target_layer - i
                    
                    # Check bounds and load the weights
                    if 0 <= relative_target_layer < dec_weights.shape[1]:
                        # dec_weights is [d_latent, N_layers-i, d_in]
                        # W_dec expects [d_latent, d_in] for each decoder
                        clt.W_dec.data[decoder_idx] = dec_weights[:, relative_target_layer, :].clone()
                        print(f"    Mapped decoder {decoder_idx}: enc_layer {i} -> target_layer {target_layer} (rel: {relative_target_layer})")
                    else:
                        print(f"    Warning: Skipping decoder {decoder_idx}: enc_layer {i} -> target_layer {target_layer} (rel: {relative_target_layer}) - out of bounds")

                # Set decoder biases
                layer_b_dec = enc_data[f'b_dec_{i}']
                print(f"  b_dec_{i} shape: {layer_b_dec.shape}, expected: [d_in] or [{n_layers-i}, {d_in}]")
                clt.b_dec.data[layer_decoder_indices[0]] = layer_b_dec
                print('DECODER BIAS NORM', layer_b_dec.norm().mean().item())
        # Freeze thresholds since they're pre-trained
        clt.log_threshold.requires_grad = False

        print("Llama CLT loaded successfully!")
        return clt


    # not used in current implementation
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        For each layer, project out the component of the decoder gradient that is parallel to its decoder weights.
        This makes the decoder update orthogonal to its current direction, preventing magnitude growth along same axes.
        """
        assert self.W_dec.grad is not None

        dec_weight = self.W_dec
        dec_grad = self.W_dec.grad

        dot = (dec_grad * dec_weight).sum(dim=2, keepdim=True)  # shape [N_layers, d_latent, 1]
        norm_sq = (dec_weight ** 2).sum(dim=2, keepdim=True) + 1e-8  # shape [N_layers, d_latent, 1], avoid divide-by-zero
        projection = dot / norm_sq * dec_weight # shape [N_layers, d_latent, d_in]

        self.W_dec.grad -= projection
        
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=2, keepdim=True)

    def attach_model_for_replacement(
        self, 
        model_class_name=None, 
        model_name=None, 
        device=None, 
        model_from_pretrained_kwargs=None,
        model=None
    ) -> None:
        """
        Attach a model and create a hybrid forward function that uses this CLT
        instead of the original MLP. You can pass a model instance directly,
        or provide loading parameters.
        """
        if self.hybrid_forward.__name__ == "<lambda>":

            if model is not None:
                object.__setattr__(self, 'model', model)
            else:
                model_instance = load_model(
                    model_class_name,
                    model_name,
                    device, 
                    model_from_pretrained_kwargs,
                )
                object.__setattr__(self, 'model', model_instance)
                
            self.hybrid_forward = self._create_hybrid_forward()

    def _create_hybrid_forward(self):
        if not hasattr(self, "model"):
            raise RuntimeError("Model not attached. Call attach_model_for_replacement first.")
        
        patch_transformer_lens()

        def _forward_func(input_tokens: torch.Tensor, return_feat_acts: bool = False, return_sparse: bool = True):
            return replacement_forward(
                self, 
                self.model, 
                input_tokens, 
                return_feat_acts=return_feat_acts, 
                return_sparse=return_sparse
            )
        
        return _forward_func
    
    def hybrid_loss(self, input_tokens: torch.Tensor, l0_coef: float) -> torch.Tensor: 
        """
        Compute loss between hybrid model output and target logits.
        Args:
            input_tokens: [train_batch_size] - Input token sequence, should be with initial padding
        """
        # Check if input shape is divisible by context_size
        total_tokens = input_tokens.numel()
        if total_tokens % self.cfg.context_size != 0:
            tokens_to_ignore = total_tokens % self.cfg.context_size
            logger.warning(f"Warning: Input has {total_tokens} tokens, not divisible by context_size {self.cfg.context_size}. "
                f"Ignoring last {tokens_to_ignore} tokens.")
            keep_tokens = total_tokens - tokens_to_ignore
            input_tokens = input_tokens.flatten()[:keep_tokens]
        
        input_tokens_reshaped = input_tokens.reshape(-1, self.cfg.context_size) # Reshape to [B, ctx]
        assert torch.all(input_tokens_reshaped[:, 0] == self.model.tokenizer.bos_token_id), "Not all sequences start with BOS token"

        hybrid_logits, feat_acts = self.hybrid_forward(input_tokens_reshaped, return_feat_acts=True, return_sparse=False)
        target_logits = self.model(input_tokens_reshaped, return_type="logits")

        batch_size, ctx, vocab_size = hybrid_logits.shape

        hybrid_logits_flat = hybrid_logits.reshape(batch_size * ctx, vocab_size)
        target_logits_flat = target_logits.reshape(batch_size * ctx, vocab_size)

        # Instead of hardcoding position 8, compute for all positions
        hybrid_pred = torch.argmax(hybrid_logits, dim=-1)  # [B, ctx]
        target_pred = torch.argmax(target_logits, dim=-1)   # [B, ctx]

        # Compute agreement per position (mean across batch dimension)
        pred_per_pos = 100 * (hybrid_pred == target_pred).float().mean(dim=0)  # [ctx]
        
        if self.cfg.functional_loss == "argmax":
            target_tokens = torch.argmax(target_logits_flat, dim=-1)
            loss = torch.nn.functional.cross_entropy(hybrid_logits_flat, target_tokens)
        elif self.cfg.functional_loss == "kl":
            temperature = 0.2  # or even lower
            target_probs = torch.nn.functional.softmax(target_logits_flat / temperature, dim=-1)
            hybrid_log_probs = torch.nn.functional.log_softmax(hybrid_logits_flat / temperature, dim=-1)
            loss = torch.nn.functional.kl_div(hybrid_log_probs, target_probs, reduction='batchmean')

            ### L0 regularization
            if self.cfg.cross_layer_decoders:
                squared_norms = (self.W_dec**2).sum(dim=2)
                feature_norms = torch.sqrt(torch.matmul(self.layer_mask, squared_norms)) # [N_layers, d_latent]
            else: 
                feature_norms = self.W_dec.norm(dim=2) # [N_layers, d_latent]
            
            weighted_activations = feat_acts * feature_norms # [batch_size, N_layers, d_latent]
            tanh_weighted_activations = torch.tanh(C_l0_COEF * weighted_activations)  # [batch_size, N_layers, d_latent]
            l0_loss_accross_layers = l0_coef * tanh_weighted_activations.sum(dim=-1).mean(dim=0)  # [N_layers]
            l0_loss = l0_loss_accross_layers.sum()
            l0_accross_layers = (feat_acts > 0).float().sum(dim=2).mean(dim=0)

        return loss, pred_per_pos, l0_loss, l0_accross_layers
    
# function is also used in ReplacementModel, thus extract it to avoid reference cycles
def replacement_forward(clt: CLT, model: HookedRootModule, input_tokens: torch.Tensor, return_feat_acts: bool = False, return_sparse: bool = True):
    """
    Input shape should be [B, ctx], need to pad to context size otherwise.
    """
    B = input_tokens.shape[0]
    ctx = clt.cfg.context_size
    assert input_tokens.shape[1] == ctx, f"Input tokens shape: {input_tokens.shape[1]} don't match context {ctx}."
    
    with set_attn_only(model, True):
        residual, _, shortformer_pos_embed, attention_mask = model._process_input_to_residual(input_tokens)
        acts_out_loop = torch.zeros((B * ctx, clt.N_layers, clt.d_in), device=clt.device)
        feat_acts_list: list | None = [] if return_feat_acts else None

        for layer_idx in range(clt.N_layers):
            residual_attn = model._run_single_transformer_block(
                residual, layer_idx, shortformer_pos_embed, attention_mask
            )
            # mlp_input = (residual_attn)
            mlp_input = model.blocks[layer_idx].ln2(residual_attn)
            mlp_input_flatten = mlp_input.reshape(B * ctx, clt.d_in)
            mlp_input_flatten_normalized = mlp_input_flatten #* clt.estimated_norm_scaling_factor_in[layer_idx]

            feat_acts = clt.encode(mlp_input_flatten_normalized, layer_idx)[0]
            if return_feat_acts and feat_acts_list is not None:
                feat_acts_list.append(feat_acts)
            clt_outputs = clt.decode(feat_acts, layer=layer_idx)

            if clt.cfg.cross_layer_decoders:
                acts_out_loop[:, layer_idx:, :] += clt_outputs
                residual = residual + (acts_out_loop[:, layer_idx, :].reshape(B, ctx, clt.d_in) / clt.estimated_norm_scaling_factor_out[layer_idx])
            else:
                residual = residual + (clt_outputs / clt.estimated_norm_scaling_factor_out[layer_idx])

        our_logits = model._residual_to_output(residual, return_type="logits")

    if return_feat_acts: 
        feat_acts_stacked = torch.stack(feat_acts_list, dim=1)  # [B*ctx, N_layers, d_latent]
        if return_sparse:
            feat_acts_stacked = feat_acts_stacked.reshape(B, ctx, clt.N_layers, clt.d_latent)
            nonzero = feat_acts_stacked > 0
            indices = nonzero.nonzero(as_tuple=False).T
            values = feat_acts_stacked[nonzero]
            feat_acts_sparse = torch.sparse_coo_tensor(
                indices, values, feat_acts_stacked.shape, device=feat_acts_stacked.device
            )
            return our_logits, feat_acts_sparse
        else: 
            return our_logits, feat_acts_stacked
    else:
        return our_logits        

@contextmanager
def set_attn_only(model, value=True):
    old_values = [block.cfg.attn_only for block in model.blocks]
    try:
        for block in model.blocks:
            block.cfg.attn_only = value
        yield
    finally:
        for block, old in zip(model.blocks, old_values):
            block.cfg.attn_only = old
