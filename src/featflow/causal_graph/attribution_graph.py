import torch
from typing import Tuple, Callable, List, Dict, Any, Optional
from featflow.causal_graph.replacement_model import AttributionData
from transformer_lens.hook_points import HookedRootModule
import os
import matplotlib.pyplot as plt

class SimpleAttributionGraph:
    def __init__(
        self,
        model: HookedRootModule,
        attribution_data: AttributionData
    ):
        self.model = model
        self.input_tokens = attribution_data.input_tokens
        self.input_string = attribution_data.input_string
        self.encoder_vecs = attribution_data.encoder_vecs  # [N_enc, d_model]
        self.decoder_vecs = attribution_data.decoder_vecs  # [N_dec, d_model]
        self.feature_enc_indices = attribution_data.feature_enc_indices  # [N_enc, 3]: (ctx_pos, layer, feature_idx)
        self.feature_dec_indices = attribution_data.feature_dec_indices  # [N_dec, 4]: (ctx_pos, target_layer, feature_idx, source_layer)
        self.feature_threshold = attribution_data.feature_threshold
        self.error_vectors = attribution_data.error_vectors
        
        self._attribution_buffer: Optional[torch.Tensor] = None
        self._embedding_attribution_buffer: Optional[torch.Tensor] = None

        # Group decoder features by layer for efficient hook setup
        decoder_layer_groups: Dict[int, Dict[str, List[Any]]] = {}
        for i, (ctx_pos, target_layer, feat_idx, source_layer) in enumerate(self.feature_dec_indices):
            target_layer = int(target_layer)  # Use target_layer for grouping
            if target_layer not in decoder_layer_groups:
                decoder_layer_groups[target_layer] = {'indices': [], 'positions': [], 'decoders': []}
            decoder_layer_groups[target_layer]['indices'].append(i)
            decoder_layer_groups[target_layer]['positions'].append(int(ctx_pos))
            decoder_layer_groups[target_layer]['decoders'].append(self.decoder_vecs[i])

        self.decoder_layer_groups = decoder_layer_groups

        # Add stop gradient to non-linearities
        self._configure_gradient_flow()

    def _configure_gradient_flow(self):

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.model.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)

            # block.mlp.hook_mlp_out.add_hook(stop_gradient, is_permanent=True)

        self.model.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)
            
        for param in self.model.parameters():
            param.requires_grad = False

        # important to enable gradient for input tokens
        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self.model.hook_embed.add_hook(enable_gradient, is_permanent=True)

    def _setup_attribution_hooks(self) -> List[Tuple[str, Callable]]:
        
        def make_hook(layer_idx, target_indices, target_positions, target_decoders):
            target_decoders = torch.stack(target_decoders)
            target_positions = torch.tensor(target_positions, device=target_decoders.device)
            
            def hook_fn(grads, hook):
                if self._attribution_buffer is not None:
                    selected_grads = grads[0, target_positions]
                    scores = torch.einsum('nd,nd->n', target_decoders, selected_grads)
                    self._attribution_buffer[target_indices] = scores
                    
            return f"blocks.{layer_idx}.hook_mlp_out.hook_out_grad", hook_fn

        def make_embedding_hook():
            def embedding_hook_fn(grads, hook):
                if self._embedding_attribution_buffer is not None:
                    token_embeddings = self.model.embed.W_E[self.input_tokens]  # [seq_len, d_model]
                    scores = torch.einsum('sd,sd->s', token_embeddings, grads[0])  # [seq_len]
                    self._embedding_attribution_buffer[:] = scores
            return "hook_embed", embedding_hook_fn
        
        def make_error_hook(layer_idx):
            def error_hook_fn(grads, hook):
                if self._error_attribution_buffer is not None:
                    scores = torch.einsum('sd,sd->s', self.error_vectors[:,layer_idx,:], grads[0])  # [seq_len]
                    self._error_attribution_buffer[layer_idx] = scores
            return f"blocks.{layer_idx}.hook_mlp_out.hook_out_grad", error_hook_fn

        hooks = []
        for layer_idx, group_data in self.decoder_layer_groups.items():
            hook_name, hook_fn = make_hook(
                layer_idx,
                group_data['indices'],
                group_data['positions'], 
                group_data['decoders']
            )
            hooks.append((hook_name, hook_fn))
        
        # Add embedding attribution hook
        embedding_hook_name, embedding_hook_fn = make_embedding_hook()
        hooks.append((embedding_hook_name, embedding_hook_fn))

        # add error attribution hooks
        for l in range(self.model.cfg.n_layers): 
            error_hook_name, error_hook_fn = make_error_hook(l)
            hooks.append((error_hook_name, error_hook_fn))
            
        return hooks

    def forward_and_cache_residuals(self, input_tokens: torch.Tensor):
        cache = {}
        def save_hook(resid, hook, layer):
            cache[layer] = resid
        
        # use TransformerLens hooks context manager
        fwd_hooks = [
            (f"blocks.{l}.ln2.hook_normalized", lambda resid, hook, l=l: save_hook(resid, hook, l))
            for l in range(self.model.cfg.n_layers)
        ]
        
        fwd_hooks.append(("unembed.hook_pre", lambda resid, hook, l=self.model.cfg.n_layers: save_hook(resid, hook, l)))

        bwd_hooks = self._setup_attribution_hooks()

        with self.model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
            logits = self.model(input_tokens.unsqueeze(0))
        
        residuals = [cache[l][0] for l in range(self.model.cfg.n_layers+1)]
        return residuals, logits

    def compute_target_attribution(self, target_encoder: torch.Tensor, target_layer: int, target_pos: int, residuals) -> Tuple[torch.Tensor, torch.Tensor]:
        self._attribution_buffer = torch.zeros(len(self.feature_dec_indices), device=self.input_tokens.device)
        self._embedding_attribution_buffer = torch.zeros(len(self.input_tokens), device=self.input_tokens.device)
        self._error_attribution_buffer = torch.zeros((self.model.cfg.n_layers, len(self.input_tokens)), device=self.input_tokens.device)

        try:
            grad = torch.zeros_like(residuals[target_layer])
            grad[target_pos] = target_encoder
            
            residuals[target_layer].backward(gradient=grad, retain_graph=True)

            decoder_result = self._attribution_buffer.clone() if self._attribution_buffer is not None else torch.zeros(len(self.feature_dec_indices), device=self.input_tokens.device)
            embedding_result = self._embedding_attribution_buffer.clone() if self._embedding_attribution_buffer is not None else torch.zeros(len(self.input_tokens), device=self.input_tokens.device)
            error_result = self._error_attribution_buffer.clone() if self._error_attribution_buffer is not None else torch.zeros((self.model.cfg.n_layers, len(self.input_tokens)), device=self.input_tokens.device)

            return decoder_result, embedding_result, error_result
        
        finally:
            self._attribution_buffer = None
            self._embedding_attribution_buffer = None
            self._error_attribution_buffer = None

    def compute_graph(self, folder_name: str, feature_threshold: float = 0.8, edge_threshold: float = 0.98, replace_logit_with_vector: Optional[torch.tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_encoders = len(self.encoder_vecs)
        n_decoders = len(self.decoder_vecs)
        n_tokens = len(self.input_tokens)
        n_errors = self.model.cfg.n_layers * n_tokens
        
        # Separate adjacency matrices
        decoder_adjacency = torch.zeros((n_decoders, n_encoders + 1), device=self.input_tokens.device)
        embedding_adjacency = torch.zeros((n_tokens, n_encoders + 1), device=self.input_tokens.device)
        error_adjacency = torch.zeros((n_errors, n_encoders + 1), device=self.input_tokens.device)
        
        self.input_tokens = self.input_tokens.to(self.encoder_vecs.device)
        # self.input_tokens.requires_grad_(True)
        
        residuals, logits = self.forward_and_cache_residuals(self.input_tokens)
        for resid in residuals:
            resid.requires_grad_(True)
        
        logits = logits[0, -1]
        
        # Get top 5 logits with their probabilities
        probs = logits.softmax(-1)
        top5_vals, top5_indices = logits.topk(5, dim=-1)
        top5_probs = probs[top5_indices]
        
        # Keep the original top logit for backward compatibility
        top_logit_val, top_logit_idx = logits.max(dim=-1)
        logit_vec = self.model.unembed.W_U[:, top_logit_idx]
        
        # Compute edges from decoders to each encoder feature
        for i in range(n_encoders):
            ctx_pos, layer, feat_idx = self.feature_enc_indices[i]
            target_encoder = self.encoder_vecs[i]

            # if int(layer) == 0:
            #     continue

            decoder_attribution, embedding_attribution, error_attribution = self.compute_target_attribution(
                target_encoder, int(layer), int(ctx_pos), residuals
            )
            
            decoder_adjacency[:, i] = decoder_attribution
            embedding_adjacency[:, i] = embedding_attribution
            error_adjacency[:, i] = error_attribution.flatten()
        
        if replace_logit_with_vector is None: 
            # Compute edges from decoders to final logit
            decoder_logit_attribution, embedding_logit_attribution, error_logit_attribution = self.compute_target_attribution(
                logit_vec, self.model.cfg.n_layers, len(self.input_tokens) - 1, residuals
            )
        else: 
            print("!!! Using replace_logit_with_vector !!!")
            assert replace_logit_with_vector.shape == logit_vec.shape, "replace_logit_with_vector must have the same shape as logit_vec"
            # Compute edges from decoders to final logit
            decoder_logit_attribution, embedding_logit_attribution, error_logit_attribution = self.compute_target_attribution(
                replace_logit_with_vector, self.model.cfg.n_layers, len(self.input_tokens) - 1, residuals
            )
        
        decoder_adjacency[:, n_encoders] = decoder_logit_attribution
        embedding_adjacency[:, n_encoders] = embedding_logit_attribution
        error_adjacency[:, n_encoders] = error_logit_attribution.flatten()
                
        # Aggregate effects for same features across layers
        aggregated_decoder_adjacency = self.aggregate_feature_effects(decoder_adjacency)
        top_logit_prob = logits.softmax(-1)[top_logit_idx]


        # Construct final adjacency matrix by concatenating all adjacency matrices
        final_adjacency_matrix = torch.zeros((n_tokens + n_errors + n_encoders, n_tokens + n_errors + n_encoders + 1), device=self.input_tokens.device)
        
        # Stack all adjacency matrices vertically and assign to appropriate sections
        all_adjacencies = torch.cat([embedding_adjacency, error_adjacency, aggregated_decoder_adjacency], dim=0)
        final_adjacency_matrix[:, -n_encoders-1:] = all_adjacencies

        # Prune the graph
        prune_adjacency_matrix, feature_mask = self.prune_graph(final_adjacency_matrix, feature_threshold=feature_threshold, edge_threshold=edge_threshold, n_tokens=n_tokens, n_errors=n_errors)
        
        # essential_parents = get_essential_parents_mask(
        #     aggregated_adjacency, 
        #     self.feature_dec_indices,
        #     self.feature_threshold
        # )

        token_strings = []
        for token_id in self.input_tokens:
            try:
                token_str = self.model.tokenizer.decode([int(token_id)])
                token_strings.append(token_str)
            except Exception as e:
                print(f"Warning: Could not decode token {token_id}: {e}")
                token_strings.append(f"<unk_{token_id}>")

        # Get top logit token as string
        try:
            top_logit_token = self.model.tokenizer.decode([int(top_logit_idx)])
        except Exception as e:
            print(f"Warning: Could not decode top logit token {top_logit_idx}: {e}")
            top_logit_token = f"<unk_{top_logit_idx}>"
        
        # Get top 5 logit tokens as strings
        top5_logit_tokens = []
        for idx in top5_indices:
            try:
                token_str = self.model.tokenizer.decode([int(idx)])
                top5_logit_tokens.append(token_str)
            except Exception as e:
                print(f"Warning: Could not decode token {idx}: {e}")
                top5_logit_tokens.append(f"<unk_{idx}>")

        output_dict = { 
            "folder":folder_name,  # or pass as argument
            "feature_indices":self.feature_enc_indices,
            "aggregated_adjacency":final_adjacency_matrix,
            "prune_adjacency_matrix":prune_adjacency_matrix,
            "feature_mask":feature_mask,
            "top_logit_idx":top_logit_idx,
            "top_logit_prob":top_logit_prob,
            "top_logit_token":top_logit_token,
            "top5_logit_indices":top5_indices,
            "top5_logit_probs":top5_probs,
            "top5_logit_tokens":top5_logit_tokens,
            "input_tokens":self.input_tokens,
            "input_string":self.input_string,
            "token_string":self.tokens_to_string()
        }

        save_graph_results(**output_dict)

        return output_dict
    
    def tokens_to_string(self) -> List[str]:
        token_strings = []
        for token_id in self.input_tokens:
            try:
                token_str = self.model.tokenizer.decode([int(token_id)])
                token_strings.append(token_str)
            except Exception as e:
                print(f"Warning: Could not decode token {token_id}: {e}")
                token_strings.append(f"<unk_{token_id}>")
        return token_strings
    
    def aggregate_feature_effects(self, adjacency: torch.Tensor) -> torch.Tensor:

        feature_signatures = {}
        unique_features: List[Tuple[int, int, int]] = []
        
        for i, (ctx_pos, target_layer, feat_idx, source_layer) in enumerate(self.feature_dec_indices):
            signature = (int(ctx_pos), int(feat_idx), int(source_layer))
            if signature not in feature_signatures:
                feature_signatures[signature] = len(unique_features)
                unique_features.append(signature)
        
        n_unique_features = len(unique_features)
        n_encoders = adjacency.shape[1] - 1  # Exclude logit column
        
        aggregated_adjacency = torch.zeros((n_unique_features, n_encoders + 1), device=adjacency.device)
        
        # Sum effects for each unique feature
        for i, (ctx_pos, target_layer, feat_idx, source_layer) in enumerate(self.feature_dec_indices):
            signature = (int(ctx_pos), int(feat_idx), int(source_layer))
            unique_idx = feature_signatures[signature]
            aggregated_adjacency[unique_idx, :] += adjacency[i, :]
        
        return aggregated_adjacency
    
    def prune_graph(self, adjacency_matrix, feature_threshold:float = 0.99, edge_threshold: float = 0.98, n_tokens: int = 0, n_errors: int = 0) -> torch.Tensor:
        # Count layer 0 features (they should be first in the matrix)
        layer_0_count = 0
        for ctx_pos, layer, feat_idx in self.feature_enc_indices:
            if int(layer) == 0:
                layer_0_count += 1
        
        return prune_adjacency_matrix(
            adjacency_matrix, 
            feature_threshold=feature_threshold, 
            edge_threshold=edge_threshold,
            layer_0_count=layer_0_count,
            n_tokens=n_tokens,
            n_errors=n_errors,
            feature_dec_indices=self.feature_dec_indices
        )    

# PRUNING ---------------------------------------------------------------------------

def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=0, keepdim=True).clamp(min=1e-10)

def compute_influence(A: torch.Tensor, logit_weight: float = 1.0, max_iter: int = 1000):
    # Start with direct influence to logit (last column)
    current_influence = A[:, -1] * logit_weight  # [n_features]
    influence = current_influence.clone()
    
    A_ff = A[:, :-1]  # [n_features, n_features] - feature to feature only
    
    iterations = 0
    while current_influence.any()  and iterations < max_iter:
        current_influence = A_ff.T @ current_influence  # [n_features]
        influence += current_influence
        iterations += 1
        print(iterations)
    
    if iterations >= max_iter:
        print(f"Warning: Influence computation may not have converged after {iterations} iterations")
    
    return influence

def compute_edge_influence(
    adjacency_matrix: torch.Tensor,  # [n_features, n_features + 1]
    feature_influence: torch.Tensor
) -> torch.Tensor:

    normalized_adj = normalize_matrix(adjacency_matrix)
    # Edge influence: edge_scores[j, i] = normalized_adj[j, i] * feature_influence[j]
    edge_scores = normalized_adj * feature_influence[:, None]
    return edge_scores


def find_threshold(scores: torch.Tensor, threshold: float, plot_name: str = None):
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index = torch.searchsorted(cumulative_score, threshold)
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    
    # Plot cumulative distribution if plot_name provided
    if plot_name is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(cumulative_score)), cumulative_score.detach().cpu().numpy(), 'b-', linewidth=2)
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2%})')
        plt.axvline(x=threshold_index.cpu().item(), color='g', linestyle=':', linewidth=2, label=f'Selected index ({threshold_index.cpu().item()})')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Cumulative Percentage')
        plt.title(f'Cumulative Distribution - Threshold: {threshold:.2%}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cumulative distribution plot to {plot_name}")
    
    return sorted_scores[threshold_index]

def count_layer_0_features(feature_mask: torch.Tensor, feature_dec_indices: torch.Tensor, n_tokens: int) -> int:
    """Count how many layer 0 features are active in the feature mask"""
    if feature_dec_indices is None:
        return 0
    
    # Get unique features from aggregation (same logic as in aggregate_feature_effects)
    feature_signatures = {}
    unique_features = []
    for i, (ctx_pos, target_layer, feat_idx, source_layer) in enumerate(feature_dec_indices):
        signature = (int(ctx_pos), int(feat_idx), int(source_layer))
        if signature not in feature_signatures:
            feature_signatures[signature] = len(unique_features)
            unique_features.append(signature)
    
    # Count layer 0 features that are active
    layer_0_count = 0
    encoder_mask = feature_mask[n_tokens:]  # Skip tokens, get encoder features
    for i, (ctx_pos, feat_idx, source_layer) in enumerate(unique_features):
        if source_layer == 0 and i < len(encoder_mask) and encoder_mask[i]:
            layer_0_count += 1
    
    return layer_0_count

def prune_adjacency_matrix(
    adjacency_matrix: torch.Tensor,  # [n_features, n_features + 1] - A[j,i] means from j to i
    feature_threshold: float = 0.99,
    edge_threshold: float = 0.98,
    layer_0_count: int = 0,
    n_tokens: int = 0,
    n_errors: int =0,
    feature_dec_indices: torch.Tensor = None
) -> torch.Tensor:
    """
    Prune adjacency matrix by removing nodes with low influence
    """

    if feature_threshold > 1.0 or feature_threshold < 0.0:
        raise ValueError("feature_threshold must be between 0.0 and 1.0")
    
    # Compute node influence
    feature_influence = compute_influence(normalize_matrix(adjacency_matrix))
    
    influence_threshold = find_threshold(feature_influence, feature_threshold, "feature_plot.png")
    feature_mask = feature_influence >= influence_threshold
    feature_mask[:n_tokens] = True
    feature_mask[-1] = True

    # Count pruned nodes (all outgoing edges from these nodes will be zeroed)
    num_nodes_pruned = (~feature_mask).sum().item()
    num_edges_before = (adjacency_matrix != 0).sum().item()

    pruned_adjacency = adjacency_matrix.clone()
    pruned_adjacency[~feature_mask, :] = 0 # Removes edges from weak features
    pruned_adjacency[:, :-1][:, ~feature_mask] = 0 # Removes edges to weak features

    num_edges_after_node_pruning = (pruned_adjacency != 0).sum().item()
    print(f"Node pruning: pruned {num_nodes_pruned} nodes, edges reduced from {num_edges_before} to {num_edges_after_node_pruning}")

    # Prune edges based on infuence
    feature_influence = compute_influence(normalize_matrix(pruned_adjacency))
    edge_scores = compute_edge_influence(pruned_adjacency, feature_influence)
    flat_scores = edge_scores.flatten()
    threshold = find_threshold(flat_scores, edge_threshold, "edge_plot.png")
    edge_mask = edge_scores >= threshold

    old_feature_mask = feature_mask.clone()
    # Ensure feature and error nodes have outgoing edges
    feature_mask[n_tokens:] &= edge_mask[n_tokens:,:].any(1)
    # Ensure feature nodes have incoming edges
    feature_mask[n_tokens+n_errors:] &= edge_mask[:, n_tokens+n_errors:-1].any(0)

    iteration = 0
    
    while not torch.all(old_feature_mask == feature_mask):
        print(iteration)
        iteration += 1
        old_feature_mask = feature_mask.clone()
        edge_mask[:, :-1][:,~feature_mask] = False
        edge_mask[~feature_mask] = False

        # Ensure feature and error nodes have outgoing edges
        feature_mask[n_tokens:] &= edge_mask[n_tokens:,:].any(1)
        # Ensure feature nodes have incoming edges
        feature_mask[n_tokens+n_errors:] &= edge_mask[:, n_tokens+n_errors:-1].any(0)

    # Final adjacency update after convergence
    pruned_adjacency[~edge_mask] = 0
    pruned_adjacency[~feature_mask, :] = 0 # Removes edges from weak features
    pruned_adjacency[:, :-1][:, ~feature_mask] = 0 # Removes edges to weak features

    num_nodes_pruned = (~feature_mask).sum().item()

    num_edges_after_node_pruning = (pruned_adjacency != 0).sum().item()
    print(f"Final pruning: pruned {num_nodes_pruned} nodes, edges reduced from {num_edges_before} to {num_edges_after_node_pruning}")

    return pruned_adjacency, feature_mask

def save_graph_results(
    folder: str,
    feature_indices: torch.Tensor,
    aggregated_adjacency: torch.Tensor,
    prune_adjacency_matrix: torch.Tensor,
    feature_mask: torch.Tensor,
    top_logit_idx: torch.Tensor,
    top_logit_prob: torch.Tensor,
    top_logit_token: str,
    top5_logit_indices: torch.Tensor,
    top5_logit_probs: torch.Tensor,
    top5_logit_tokens: List[str],
    input_tokens: torch.Tensor,
    input_string: str, 
    token_string: List[str]
):
    os.makedirs(folder, exist_ok=True)
    # Convert adjacency to sparse
    sparse_adj = aggregated_adjacency.to_sparse()
    sparse_pruned_adj = prune_adjacency_matrix.to_sparse()

    # Prepare dictionary
    result = {
        "feature_indices": feature_indices,
        "adjacency_sparse": sparse_adj,
        "prune_adjacency_matrix": prune_adjacency_matrix,
        "sparse_pruned_adj": sparse_pruned_adj,
        "feature_mask": feature_mask.cpu(),
        "top_logit_idx": top_logit_idx.cpu(),
        "top_logit_prob": top_logit_prob.cpu(),
        "top_logit_token": top_logit_token,
        "top5_logit_indices": top5_logit_indices.cpu(),
        "top5_logit_probs": top5_logit_probs.cpu(),
        "top5_logit_tokens": top5_logit_tokens,
        "input_tokens": input_tokens.cpu(),
        "input_string": input_string, 
        "token_string": token_string
    }
    torch.save(result, os.path.join(folder, "attribution_graph.pt"))
    print(f"Saved attribution graph to {os.path.join(folder, 'attribution_graph.pt')}")

def is_parent_essential(
    adjacency_matrix: torch.Tensor, 
    parent_idx: int, 
    feature_dec_indices: torch.Tensor,
    feature_threshold: torch.Tensor
) -> bool:
    """Check if removing a parent would deactivate any children."""
    children_edges = adjacency_matrix[parent_idx, :-1]
    children_indices = torch.where(children_edges != 0)[0]
    
    if len(children_indices) == 0:
        return False
    
    children_totals = adjacency_matrix[:, children_indices].sum(dim=0)
    children_without_parent = children_totals - children_edges[children_indices]
    
    if feature_threshold.numel() == 1:
        threshold = feature_threshold.item()
        return (children_without_parent < threshold).any().item()
    else:
        for i, child_idx in enumerate(children_indices):
            child_layer = int(feature_dec_indices[child_idx, 1])
            child_feat_idx = int(feature_dec_indices[child_idx, 2])
            
            if (child_layer < feature_threshold.shape[0] and 
                child_feat_idx < feature_threshold.shape[1]):
                threshold = feature_threshold[child_layer, child_feat_idx].item()
            else:
                threshold = feature_threshold.mean().item()
            
            if children_without_parent[i].item() < threshold:
                return True
        return False

def get_essential_parents_mask(
    adjacency_matrix: torch.Tensor, 
    feature_dec_indices: torch.Tensor,
    feature_threshold: torch.Tensor
) -> torch.Tensor:
    """Get boolean mask of essential parents."""
    return torch.tensor([
        is_parent_essential(adjacency_matrix, i, feature_dec_indices, feature_threshold) 
        for i in range(adjacency_matrix.shape[0])
    ], device=adjacency_matrix.device, dtype=torch.bool)
