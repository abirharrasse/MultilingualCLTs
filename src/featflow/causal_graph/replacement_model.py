from featflow.transformer_lens.hooked_transformer_wrapper import patch_transformer_lens
from featflow.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from transformer_lens.hook_points import HookPoint
from featflow.clt import CLT, replacement_forward
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path
import json
import torch
import os
from featflow.load_model import load_model 
from pydantic import BaseModel
from functools import partial

class AttributionData(BaseModel):
    input_tokens: torch.Tensor
    input_string: str
    encoder_vecs: torch.Tensor
    decoder_vecs: torch.Tensor
    feature_enc_indices: torch.Tensor
    feature_dec_indices: torch.Tensor
    feature_threshold: torch.Tensor
    error_vectors: torch.Tensor
    
    class Config:
        arbitrary_types_allowed = True  # Allow torch.Tensor

# inspired from Circuit-Tracing
class ReplacementUnembed(torch.nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: torch.nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return x

class ReplacementModel:
    """
    Prepares the feature activations for attribution graph. 
    """

    def __init__(
        self, 
        model_class_name: str, 
        model_name: str,
        clt_path: str,
        device: torch.device,
        model_from_pretrained_kwargs: dict = {}
    ) -> None:
        
        patch_official_model_names()
        patch_convert_hf_model_config()

        self.model = load_model(
            model_class_name,
            model_name,
            torch.device(device),
            model_from_pretrained_kwargs
        )
        # important for transformer lens
        self.model.cfg.use_hook_mlp_in = True 

        self.clt = CLT.load_from_pretrained(clt_path, str(device))

        self.model.unembed = ReplacementUnembed(self.model.unembed)
        for block in self.model.blocks: 
            # Configure MLP gradient hook
            self._configure_mlp_gradient_hook(block)
        self.model.setup()

        self.device = device

        self.N_layers = self.clt.N_layers
        self.d_latent = self.clt.d_latent
        self.without_error = False
        
        # Initialize doc_languages for multilingual support
        self.doc_languages = None
        
        # Adds useful methods to transformer lens
        patch_transformer_lens()


    def _configure_mlp_gradient_hook(self, block):
        
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def hook_output_mlp(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            cached_residual = cached.pop("acts")
            skip = cached_residual * 0 # keeps the computational graph
            return grad_hook(skip + (acts - skip).detach())

        # add feature input hook
        subblock = block.ln2.hook_normalized
        subblock.add_hook(cache_activations, is_permanent=True)
                        
        subblock = block.hook_mlp_out
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(hook_output_mlp, grad_hook=subblock.hook_out_grad),
            is_permanent=True,
        )

    def forward(self, input_tokens: torch.Tensor, return_feat_acts: bool = False):
        """
        Replacement forward pass using CLT instead of original MLP.
        Input shape should be [B, ctx].
        """

        return replacement_forward(self.clt, self.model, input_tokens, return_feat_acts)

    @torch.no_grad()

    def feature_activations_with_error(self, input_tokens: torch.Tensor, hook_names_in: List[str], hook_names_out: List[str], return_sparse: bool = True) -> torch.Tensor:
        """
        Get feature activations from the original base model.
        """

        hook_names = hook_names_in + hook_names_out

        cache = self.model.run_with_cache(
            input_tokens,
            names_filter=hook_names,
            prepend_bos=False,
        )[1]

        missing = [n for n in hook_names if n not in cache]
        if missing:
            raise KeyError(f"The following hooks were not found in the cache: {missing}")
            
        acts_list = [cache[n].flatten(2) for n in hook_names_in]
        output_list = [cache[n].flatten(2) for n in hook_names_out]
        output = torch.stack(output_list, dim=0).permute(1, 2, 0, 3)
        acts = torch.stack(acts_list, dim=0).permute(1, 2, 0, 3)
        B, C, N_layers, d = acts.shape
        acts = acts.reshape(B * C, N_layers, d)

        # normalize activations for clt
        acts = acts * self.clt.estimated_norm_scaling_factor_in.view(1, -1, 1)

        # shape is [Batch, N_layers, d_latent]
        feat_acts = self.clt.encode(acts)[0]
        reconstructed_acts = self.clt.decode(feat_acts) / self.clt.estimated_norm_scaling_factor_out.view(1, -1, 1)
        
        feat_acts = feat_acts.reshape(B, C, N_layers, self.d_latent)
        reconstructed_acts = reconstructed_acts.reshape(B, C, N_layers, d)

        error_vectors = output - reconstructed_acts #shape [B, C, N_layers, d]
        print(f"Norm of error vectors: {error_vectors.norm(dim=-1).mean().item():.4f}")
        print(f"Norm of output vectors: {output.norm(dim=-1).mean().item():.4f}")

        if not return_sparse: 
            return feat_acts

        nonzero = feat_acts > 0
        indices = nonzero.nonzero(as_tuple=False).T
        values = feat_acts[nonzero]

        feat_acts_sparse = torch.sparse_coo_tensor(
            indices, values, feat_acts.shape, device=feat_acts.device
        )

        return feat_acts_sparse, error_vectors
    
    def get_feature_activations(self, input_tokens: torch.Tensor, without_error: bool = True) -> list[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Get feature activations from the CLT model or the basemodel, depending on 'without_error'.
        """

        assert input_tokens.shape[1] == self.clt.cfg.context_size, "input_tokens must have a context size equal to clt.context_size"
        
        if without_error: 
            _, feat_acts_sparse = self.forward(input_tokens, return_feat_acts=True)
            error_vectors = None

            return feat_acts_sparse
        else: 
            feat_acts_sparse, error_vectors = self.feature_activations_with_error(
                input_tokens, 
                hook_names_in=[f"blocks.{i}.ln2.hook_normalized" for i in range(self.N_layers)], 
                hook_names_out=[f"blocks.{i}.hook_mlp_out" for i in range(self.N_layers)]
            )

        return feat_acts_sparse, error_vectors
    
    def data_for_attribution(self, input_string: str): 

        # tokenize the sentence and add BOS token
        input_string_with_BOS = self.model.tokenizer.bos_token + input_string
        input_tokens = self.model.tokenizer.encode(input_string_with_BOS, return_tensors="pt")[0].to(self.device)
        
        original_length = input_tokens.shape[0]
        
        # Pad to context size if needed
        context_size = self.clt.cfg.context_size

        if original_length < context_size:
            pad_length = context_size - original_length
            pad_token_id = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else 0
            padding = torch.full((pad_length,), pad_token_id, device=self.device)
            input_tokens_pad = torch.cat([input_tokens, padding], dim=0)
        else:
            input_tokens_pad = input_tokens

        input_tokens_pad = input_tokens_pad.unsqueeze(0)
        feat_acts_sparse, error_vectors = self.get_feature_activations(input_tokens_pad, without_error=self.without_error)

        # remove first dimension 
        feat_acts_sparse = feat_acts_sparse[0].detach()
        error_vectors = error_vectors[0].detach()

        # Convert back to original size
        if feat_acts_sparse.is_sparse:
            feat_acts_dense = feat_acts_sparse.to_dense()
            feat_acts_sparse = feat_acts_dense[:original_length, :, :]
        else:
            feat_acts_sparse = feat_acts_sparse[:original_length, :, :]
        error_vectors = error_vectors[:original_length, :, :]

        # Put 0 error reconstruction for BOS, and remove BOS features
        error_vectors[0,:,:] = 0.0
        feat_acts_sparse[0,:,:] = 0.0 # TODO: is it enough ?
        
        nonzero = feat_acts_sparse > 0
        indices = nonzero.nonzero(as_tuple=False)
        values = feat_acts_sparse[nonzero]

        encoder_vectors_per_layer = [torch.empty(0, self.clt.d_in, device=self.device) for _ in range(self.N_layers)]
        decoder_vectors_per_layer = [torch.empty(0, self.clt.d_in, device=self.device) for _ in range(self.N_layers)]    
        
        feature_enc_indices = []
        feature_dec_indices = [] 

        for layer in range(self.N_layers):
            # get features from that layer
            layer_mask = (indices[:, 1] == layer)
            feats_idx = indices[layer_mask, 2]
            alpha = values[layer_mask]
            layer_indices = indices[layer_mask]  # [N_features_in_layer, 3]
            
            # multiply features by activation coefficient for encoder features
            if feats_idx.numel() > 0:
                feature_enc_indices.append(layer_indices)
                
                w_enc = self.clt.W_enc[layer]
                w_slice = w_enc[:, feats_idx].T
                w_slice = w_slice * alpha.unsqueeze(-1) 
                encoder_vectors_per_layer[layer] = w_slice

            if self.clt.cfg.cross_layer_decoders:
                # Find decoder matrices that target this layer
                decode_rows = (self.clt.k_idx == layer).nonzero(as_tuple=True)[0]

                dec_vectors_all = []
                dec_indices_all = []
                
                for dec_row in decode_rows:
                    # Find the source layer for this decoder matrix
                    source_layer = self.clt.l_idx[dec_row].item()
                    
                    source_mask = (indices[:, 1] == source_layer)
                    source_feats_idx = indices[source_mask, 2]
                    source_alpha = values[source_mask]
                    source_indices = indices[source_mask]
                    
                    if source_feats_idx.numel() > 0:
                        dec_feature_indices = torch.cat([
                            source_indices[:, :1],  # ctx_pos
                            torch.full((len(source_indices), 1), layer, device=self.device),  # target_layer
                            source_indices[:, 2:3],  # feature_idx
                            torch.full((len(source_indices), 1), source_layer, device=self.device),  # source_layer
                        ], dim=1)
                        dec_indices_all.append(dec_feature_indices)
                        
                        w_slice_dec = self.clt.W_dec[dec_row][source_feats_idx, :]
                        # multiply by the activation and divide by the norm scaling factor
                        w_slice_dec = w_slice_dec * source_alpha.unsqueeze(-1) / self.clt.estimated_norm_scaling_factor_out[layer]
                        dec_vectors_all.append(w_slice_dec)
                
                if dec_vectors_all:
                    w_slice_dec_total = torch.cat(dec_vectors_all, dim=0)
                    decoder_vectors_per_layer[layer] = w_slice_dec_total
                    # Store decoder feature indices for this target layer
                    feature_dec_indices.extend(dec_indices_all)
            else:
                # Standard case: decoder for this layer uses activations from this layer
                if feats_idx.numel() > 0:

                    # Add source layer info (same as target layer in standard case)
                    dec_feature_indices = torch.cat([
                        layer_indices[:, :1], 
                        layer_indices[:, 1:2],
                        layer_indices[:, 2:3],
                        layer_indices[:, 1:2],
                    ], dim=1)

                    feature_dec_indices.append(dec_feature_indices)
                    
                    w_dec = self.clt.W_dec[layer]
                    w_slice_dec = w_dec[feats_idx, :]
                    w_slice_dec = w_slice_dec * alpha.unsqueeze(-1)
                    decoder_vectors_per_layer[layer] = w_slice_dec

        feature_enc_indices = torch.cat(feature_enc_indices, dim=0) if feature_enc_indices else torch.empty(0, 3, device=self.device)
        feature_dec_indices = torch.cat(feature_dec_indices, dim=0) if feature_dec_indices else torch.empty(0, 4, device=self.device)

        # Flatten vectors
        encoder_vecs = torch.cat([v for v in encoder_vectors_per_layer if v.numel() > 0], dim=0)
        decoder_vecs = torch.cat([v for v in decoder_vectors_per_layer if v.numel() > 0], dim=0)

        feature_threshold = torch.exp(self.clt.log_threshold) - self.clt.b_enc

        attribution_data = AttributionData(
            input_tokens=input_tokens,
            input_string=input_string,
            encoder_vecs=encoder_vecs,
            decoder_vecs=decoder_vecs,
            feature_enc_indices=feature_enc_indices,
            feature_dec_indices=feature_dec_indices, 
            feature_threshold=feature_threshold, 
            error_vectors=error_vectors
        )
                
        return attribution_data    
    
    def compute_feature_correlation_scores(self, folder_name: str, pruned_adjacency: torch.Tensor, 
                                        feature_enc_indices: torch.Tensor,
                                        dict_base_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        
        active_indices = torch.where((pruned_adjacency != 0).any(dim=1))[0]
        
        features = {}
        for idx in active_indices:
            ctx_pos, layer, feature_idx = feature_enc_indices[idx]
            key = idx.item()  # Use matrix index as key
            
            dict_file = Path(dict_base_dir) / f"layer{int(layer)}" / f"feature_{int(feature_idx)}_complete.json"
            if dict_file.exists():
                with open(dict_file, 'r') as f:
                    features[key] = (int(layer), int(feature_idx), json.load(f))
        
        # Initialize matrices with same shape as attribution matrix
        activation_rate_matrix = torch.zeros_like(pruned_adjacency)
        normalized_score_matrix = torch.zeros_like(pruned_adjacency)
        
        for source_idx, (_, _, source_dict) in features.items():
            examples = source_dict.get('top_examples_tks', [])
            
            if examples:
                # Extract token sequences from the dictionary format
                sequences = [example['tokens'] for example in examples if 'tokens' in example]
                
                if sequences:
                    batch_tokens = torch.tensor(sequences).to(self.device)
                    
                    with torch.no_grad():
                        feat_acts_sparse = self.feature_activations_with_error(
                            batch_tokens, 
                            hook_names=[f"blocks.{i}.ln2.hook_normalized" for i in range(self.N_layers)], 
                            return_sparse=False
                        )
                        
                        for target_idx, (target_layer, target_feat, target_dict) in features.items():
                            activations = feat_acts_sparse[:, :, target_layer, target_feat]
                            max_activations = activations.max(dim=1).values
                            
                            active_sequences = (max_activations > 0).sum().item()
                            total_sequences = len(sequences)
                            activation_rate = active_sequences / total_sequences if total_sequences > 0 else 0.0
                            
                            if active_sequences > 0:
                                avg_activation_when_active = max_activations[max_activations > 0].mean().item()
                            else:
                                avg_activation_when_active = 0.0
                            
                            target_avg_activation = target_dict.get('average_activation', 1.0)
                            normalized_score = avg_activation_when_active / target_avg_activation if target_avg_activation > 0 else 0.0
                            
                            activation_rate_matrix[source_idx, target_idx] = activation_rate
                            normalized_score_matrix[source_idx, target_idx] = normalized_score
        
        # Convert to sparse matrices
        activation_rate_sparse = activation_rate_matrix.to_sparse()
        normalized_score_sparse = normalized_score_matrix.to_sparse()
        
        # Save matrices
        save_data = {
            'activation_rate_matrix': activation_rate_sparse,
            'normalized_score_matrix': normalized_score_sparse,
            'feature_enc_indices': feature_enc_indices
        }
        
        save_path = Path(folder_name) / "feature_correlation_matrices.pt"
        torch.save(save_data, save_path)
        
        return activation_rate_sparse, normalized_score_sparse
    
    def intervention(
            self, 
            features_to_intervene: list, 
            input_tokens: torch.Tensor,
            freeze_attention: bool = True
    ): 
        """
        Intervene on specific feature activations during forward pass.
        
        Args:
            features_to_intervene: List of tuples [(layer, pos, feature_idx, value)]
            input_tokens: Input tokens tensor [batch_size, seq_len]
            
        Returns:
            Tuple of (original_output, intervened_output, steering_vectors)
        """

        
        interventions_by_layer: dict[int, list[tuple[int, int, float]]] = {}
        for layer, pos, feature_idx, value in features_to_intervene:
            if layer not in interventions_by_layer:
                interventions_by_layer[layer] = []
            interventions_by_layer[layer].append((pos, feature_idx, value))
        
        cached_activations = {}
        steering_vectors = {}
        
        # cache attention patterns if requested
        freeze_cache = {}
        cache_hooks = []
        if freeze_attention:
            hookpoints_to_freeze = ["hook_pattern"]
            freeze_cache, cache_hooks, _ = self.model.get_caching_hooks(
                names_filter=lambda name: any(hookpoint in name for hookpoint in hookpoints_to_freeze)
            )
        
        with torch.no_grad():
            original_logits = self.model.run_with_hooks(input_tokens, fwd_hooks=cache_hooks)

        steering_vectors_accumulation = torch.zeros((input_tokens.shape[0], input_tokens.shape[1], self.N_layers, self.clt.d_in), device=self.device)
        
        def make_steering_calculation_hook(layer_idx, interventions):
            def hook_fn(activations, hook):

                if layer_idx not in cached_activations:
                    cached_activations[layer_idx] = activations.clone()
                
                # CLT encode
                B, seq_len, d_model = activations.shape
                acts_flat = activations.reshape(B * seq_len, d_model)
                acts_normalized = acts_flat * self.clt.estimated_norm_scaling_factor_in[layer_idx]
                feat_acts, _ = self.clt.encode(acts_normalized, layer_idx)
                feat_acts = feat_acts.reshape(B, seq_len, -1)
                
                # Intervene on features
                modified_feat_acts = feat_acts.clone()
                for pos, feature_idx, value in interventions:
                    if pos < seq_len and feature_idx < feat_acts.shape[2]:
                        modified_feat_acts[:, pos, feature_idx] = value * modified_feat_acts[:, pos, feature_idx]
                
                feat_acts_flat = feat_acts.reshape(B * seq_len, -1)
                modified_feat_acts_flat = modified_feat_acts.reshape(B * seq_len, -1)
                
                # Decode both original and intervened
                original_decoded = self.clt.decode(feat_acts_flat, layer_idx)
                modified_decoded = self.clt.decode(modified_feat_acts_flat, layer_idx)
                original_decoded = original_decoded.reshape(B, seq_len, self.N_layers - layer_idx, d_model)
                modified_decoded = modified_decoded.reshape(B, seq_len, self.N_layers - layer_idx, d_model)
                
                # Add the difference to steering vectors 
                steering_vectors_accumulation[:, :, layer_idx:] += (original_decoded - modified_decoded)

                # Compute steering vector
                steering_vector = steering_vectors_accumulation[:, :, layer_idx] / self.clt.estimated_norm_scaling_factor_out[layer_idx]
                steering_vectors[layer_idx] = steering_vector
                
                return activations
            return hook_fn
        
        def make_mlp_intervention_hook(layer_idx):
            def hook_fn(mlp_output, hook):
                if layer_idx in steering_vectors:
                    return mlp_output + steering_vectors[layer_idx]
                return mlp_output
            return hook_fn
        
        intervention_hooks = []
        for layer_idx, interventions in interventions_by_layer.items():
        
            ln2_hook_name = f"blocks.{layer_idx}.ln2.hook_normalized"
            ln2_hook_fn = make_steering_calculation_hook(layer_idx, interventions)
            intervention_hooks.append((ln2_hook_name, ln2_hook_fn))
            
            mlp_hook_name = f"blocks.{layer_idx}.hook_mlp_out"
            mlp_hook_fn = make_mlp_intervention_hook(layer_idx)
            intervention_hooks.append((mlp_hook_name, mlp_hook_fn))
        
        def make_freeze_hook(hook_name, cached_value):
            def freeze_hook_fn(activations, hook):
                # Ensure the cached value matches the current activation shape
                if activations.shape == cached_value.shape:
                    return cached_value
                else:
                    # If shapes don't match, return the original activations
                    return activations
            return freeze_hook_fn
        
        freeze_hooks = []
        if freeze_attention:
            for hook_name, cached_value in freeze_cache.items():
                freeze_hook_fn = make_freeze_hook(hook_name, cached_value)
                freeze_hooks.append((hook_name, freeze_hook_fn))
        
        all_hooks = freeze_hooks + intervention_hooks
        
        with self.model.hooks(fwd_hooks=all_hooks):
            intervened_logits = self.model(input_tokens)
        
        return original_logits, intervened_logits
    
    def compute_cluster_intervention(
        self,
        cluster_features: List[Tuple[int, int, int]],
        manual_features: List[Tuple[int, int, int, float]],
        input_tokens: torch.Tensor,
        cluster_intervention_value: float,
        top_tokens_count: int = 4,
        freeze_attention: bool = False
    ) -> dict:
        features_to_intervene = []
        
        for pos, layer, feature_idx in cluster_features:
            features_to_intervene.append((layer, pos, feature_idx, cluster_intervention_value))
            
        for pos, layer, feature_idx, intervention_value in manual_features:
            features_to_intervene.append((layer, pos, feature_idx, intervention_value))

        original_logits, intervened_logits = self.intervention(features_to_intervene, input_tokens.unsqueeze(0), freeze_attention)

        # Get original probabilities
        original_probs = torch.softmax(original_logits[0, -1], dim=-1)
        intervened_probs = torch.softmax(intervened_logits[0, -1], dim=-1)
        
        # Get top tokens from intervened output
        top_tokens = torch.topk(intervened_probs, top_tokens_count)
        top_token_strings = [self.model.tokenizer.decode([token_id]) for token_id in top_tokens.indices]
        
        # Get baseline probabilities for these same tokens
        baseline_probs = []
        prob_differences = []
        for token_id in top_tokens.indices:
            baseline_prob = original_probs[token_id].item()
            intervened_prob = intervened_probs[token_id].item()
            baseline_probs.append(baseline_prob)
            prob_differences.append(intervened_prob - baseline_prob)
        
        return {
            'tokens': top_token_strings,
            'probabilities': top_tokens.values.cpu().tolist(),
            'baseline_probabilities': baseline_probs,
            'probability_differences': prob_differences,
            'cluster_feature_count': len(cluster_features),
            'manual_feature_count': len(manual_features)
        }
    
    def compute_intervention_top_tokens(self, folder_name: str, input_tokens: torch.Tensor, top_tokens_count: int = 4, intervention_values: List[float] = [-10.0], freeze_attention: bool = False) -> List[dict]:
        """
        Compute top tokens when each active feature in the pruned graph is intervened with specified values.
        
        Args:
            folder_name: Path to folder containing attribution_graph.pt
            input_tokens: Input tokens tensor
            intervention_values: List of intervention multiplier values to apply
            
        Returns:
            List of top token strings after intervention for each active feature and intervention value
        """
        
        # Load the saved graph data
        data = torch.load(os.path.join(folder_name, "attribution_graph.pt"))
        pruned_adjacency = data["sparse_pruned_adj"].to_dense()
        feature_indices = data["feature_indices"]
        input_tokens = data["input_tokens"]
        
        # Remove embedding rows/cols to get encoder-only adjacency matrix
        n_tokens = len(input_tokens)
        n_errors = n_tokens * self.clt.N_layers
        pruned_adjacency = pruned_adjacency[n_tokens+n_errors:]  # Remove first n_tokens rows
        pruned_adjacency = pruned_adjacency[:, n_tokens+n_errors:]  # Remove first n_tokens columns
        feature_indices = data["feature_indices"]
        
        # Get baseline prediction info from saved data
        baseline_top_token_id = data["top_logit_idx"].item()
        baseline_top_prob = data["top_logit_prob"].item()
        baseline_top_token = data["top_logit_token"]
        
        # Create active features mask
        active_mask = (torch.abs(pruned_adjacency).sum(axis=1)) > 0
        active_feature_indices = feature_indices[active_mask]
        intervention_top_tokens: List[Dict[str, Any]] = []
        
        for feature_data in active_feature_indices:
            ctx_pos, layer, feature_idx = int(feature_data[0]), int(feature_data[1]), int(feature_data[2])
            
            feature_results: Dict[str, Any] = {
                'feature_info': {'layer': layer, 'position': ctx_pos, 'feature_idx': feature_idx},
                'interventions': []
            }
            
            for intervention_value in intervention_values:
                features_to_intervene = [(layer, ctx_pos, feature_idx, intervention_value)]
                
                try:
                    original_logits, intervened_logits = self.intervention(
                        features_to_intervene, 
                        input_tokens.unsqueeze(0),
                        freeze_attention
                    )
                    
                    # Get top tokens after intervention
                    intervened_probs = torch.softmax(intervened_logits[0, -1], dim=-1)
                    top_tokens = torch.topk(intervened_probs, top_tokens_count)
                    
                    top_token_strings = []
                    for token_id in top_tokens.indices:
                        token_str = self.model.tokenizer.decode([token_id])
                        top_token_strings.append(token_str)
                    
                    # Check how much the baseline token probability changed
                    baseline_token_new_prob = intervened_probs[baseline_top_token_id].item()
                    prob_change = baseline_token_new_prob - baseline_top_prob
                    
                    feature_results['interventions'].append({
                        'intervention_value': intervention_value,
                        'tokens': top_token_strings,
                        'probabilities': top_tokens.values.cpu().tolist(),
                        'baseline_token': baseline_top_token,
                        'baseline_prob_original': baseline_top_prob,
                        'baseline_prob_after_intervention': baseline_token_new_prob,
                        'baseline_prob_change': prob_change
                    })
                    
                except Exception as e:
                    print(f"Warning: Intervention failed for layer_{layer}_pos_{ctx_pos}_feat_{feature_idx}_val_{intervention_value}: {e}")
                    # Add empty intervention result instead of None
                    feature_results['interventions'].append({
                        'intervention_value': intervention_value,
                        'tokens': [],
                        'probabilities': [],
                        'baseline_token': '',
                        'baseline_prob_original': 0.0,
                        'baseline_prob_after_intervention': 0.0,
                        'baseline_prob_change': 0.0
                    })
            
            intervention_top_tokens.append(feature_results)
        
        # Add results to saved graph data
        data["intervention_top_tokens"] = intervention_top_tokens
        torch.save(data, os.path.join(folder_name, "attribution_graph.pt"))
        
        return intervention_top_tokens
    
    def logit_lens_prediction_check(self, input_string: str, target_tokens: List[str] = None) -> Dict[str, Any]:
        """
        Check if the top next token prediction can be predicted in earlier layers using logit lens.
        
        Args:
            input_string: Input sentence to analyze
            target_tokens: Optional list of tokens to track (e.g., ["big", "large", "huge"]). If None, tracks only the final model prediction.
            
        Returns:
            Dictionary containing top token predictions at each layer and analysis
        """
        
        # Tokenize the sentence
        input_string_with_BOS = self.model.tokenizer.bos_token + input_string
        input_tokens = self.model.tokenizer.encode(input_string_with_BOS, return_tensors="pt")[0].to(self.device)
        
        # Pad to context size if needed
        context_size = self.clt.cfg.context_size
        original_length = input_tokens.shape[0]
        
        if original_length < context_size:
            pad_length = context_size - original_length
            pad_token_id = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else 0
            padding = torch.full((pad_length,), pad_token_id, device=self.device)
            input_tokens_pad = torch.cat([input_tokens, padding], dim=0)
        else:
            input_tokens_pad = input_tokens
            
        input_tokens_pad = input_tokens_pad.unsqueeze(0)
        
        # Get residual stream activations at each layer
        hook_names = [f"blocks.{i}.hook_resid_post" for i in range(self.N_layers)]
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                input_tokens_pad,
                names_filter=hook_names,
                prepend_bos=False
            )
        
        # Get final layer prediction for comparison
        final_logits = self.model(input_tokens_pad)
        final_probs = torch.softmax(final_logits[0, original_length-1], dim=-1)
        final_top_token_id = torch.argmax(final_probs).item()
        final_top_token_string = self.model.tokenizer.decode([final_top_token_id])
        final_top_prob = final_probs[final_top_token_id].item()
        
        # Determine which tokens to track
        tracked_tokens_info = []
        if target_tokens is not None:
            # Process each target token
            for target_token in target_tokens:
                token_ids = self.model.tokenizer.encode(target_token, add_special_tokens=False)
                if len(token_ids) == 0:
                    print(f"Warning: Target token '{target_token}' could not be tokenized, skipping")
                    continue
                token_id = token_ids[0]
                token_string = self.model.tokenizer.decode([token_id])
                tracked_tokens_info.append({
                    'token': token_string,
                    'token_id': token_id,
                    'original_input': target_token
                })
                print(f"Tracking token: '{token_string}' (ID: {token_id}) from input '{target_token}'")
        else:
            # Track the final prediction only
            tracked_tokens_info.append({
                'token': final_top_token_string,
                'token_id': final_top_token_id,
                'original_input': 'final_prediction'
            })
            print(f"Tracking final model prediction: '{final_top_token_string}' (ID: {final_top_token_id})")
        
        layer_predictions = []
        
        # Apply logit lens at each layer
        for layer_idx in range(self.N_layers):
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            residual_activations = cache[hook_name][0, original_length-1]  # Last position
            
            # Apply final layer norm before unembed
            normalized_residual = self.model.ln_final(residual_activations)
            
            # Apply unembed to get logits
            logits_at_layer = self.model.unembed(normalized_residual)
            probs_at_layer = torch.softmax(logits_at_layer, dim=-1)
            
            # Get top token prediction at this layer
            top_token_id_at_layer = torch.argmax(probs_at_layer).item()
            top_token_string_at_layer = self.model.tokenizer.decode([top_token_id_at_layer])
            top_prob_at_layer = probs_at_layer[top_token_id_at_layer].item()
            
            # Get probabilities for all tracked tokens at this layer
            tracked_token_probs = {}
            for token_info in tracked_tokens_info:
                token_id = token_info['token_id']
                prob = probs_at_layer[token_id].item()
                tracked_token_probs[token_info['token']] = {
                    'probability': prob,
                    'matches_top': top_token_id_at_layer == token_id
                }
            
            layer_info = {
                'layer': layer_idx,
                'top_token': top_token_string_at_layer,
                'top_token_id': top_token_id_at_layer,
                'top_probability': top_prob_at_layer,
                'tracked_tokens': tracked_token_probs
            }
            
            layer_predictions.append(layer_info)
        
        # Find earliest layer where each tracked token becomes top prediction
        earliest_correct_layers = {}
        for token_info in tracked_tokens_info:
            token = token_info['token']
            earliest_correct_layers[token] = None
            for layer_info in layer_predictions:
                if layer_info['tracked_tokens'][token]['matches_top']:
                    earliest_correct_layers[token] = layer_info['layer']
                    break
        
        return {
            'input_string': input_string,
            'input_tokens': input_tokens.cpu().tolist(),
            'final_prediction': {
                'top_token': final_top_token_string,
                'top_token_id': final_top_token_id,
                'probability': final_top_prob
            },
            'tracked_tokens_info': tracked_tokens_info,
            'layer_predictions': layer_predictions,
            'earliest_correct_layers': earliest_correct_layers
        }
