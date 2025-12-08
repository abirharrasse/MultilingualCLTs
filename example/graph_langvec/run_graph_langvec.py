import torch
from featflow.attribution.attribution import run_attribution
import sys
from pathlib import Path

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from featflow.load_model import load_model
from featflow.transformer_lens.multilingual_patching import (
    patch_official_model_names,
    patch_convert_hf_model_config,
)

def load_model_multilingual(MODEL_NAME, device="cuda"):
    """Load the multilingual model."""
    patch_official_model_names()
    patch_convert_hf_model_config()
    print(f"Loading model: {MODEL_NAME}")
    model = load_model("HookedTransformer", MODEL_NAME, device, {})
    print(f"Model loaded. d_model={model.cfg.d_model}")
    return model

def replace_logit_with_vector(device="cuda"): 
    """Extract last layer residual streams / last token for translated sentences"""
    
    sentences = {
        'english': "Monday, Tuesday, Wednesday, Thursday,",
        'french': "Lundi, Mardi, Mercredi, Jeudi,", 
        'german': "Montag, Dienstag, Mittwoch, Donnerstag,",
    }

    expected_token = {
        'english': " Friday",
        'french': " Vendredi", 
        'german': " Freitag",
    }

    residual_streams = {}
    unembedded_vectors = {}

    MODEL_NAME = "CausalNLP/gpt2-hf_multilingual-90"
    model = load_model_multilingual(MODEL_NAME, device)

    with torch.no_grad():
        for lang, sentence in sentences.items():
            print(f"Processing {lang}: '{sentence}'")
            
            # Tokenize with BOS token
            tokens = model.tokenizer.encode(sentence, add_special_tokens=True)
            input_tokens = torch.tensor([tokens], device=device)
            
            # Get residual stream at final layer, final position (after layer norm)
            _, cache = model.run_with_cache(input_tokens)
            residual_stream = cache['ln_final.hook_normalized'][0, -1, :]
            residual_streams[lang] = residual_stream
            
            # Check if expected token is top predicted
            logits = residual_stream @ model.W_U + model.b_U
            top_token = model.tokenizer.decode([torch.argmax(logits).item()])
            
            if top_token == expected_token[lang]:
                print(f" Expected token '{expected_token[lang]}' matches top prediction")
                # Store unembedded vector for expected token
                token_id = model.tokenizer.encode(expected_token[lang], add_special_tokens=False)[0]
                unembedded_vectors[lang] = model.W_U[:, token_id]
            else:
                print(f" Expected '{expected_token[lang]}', got '{top_token}'")
                assert False

    # compute the translation vector 
    mean_vector = torch.mean(torch.stack(list(unembedded_vectors.values())), dim=0)
    assert mean_vector.shape == unembedded_vectors["english"].shape
    translation_vectors_embed = {k: v - mean_vector for k, v in unembedded_vectors.items()}
    

    translation_vectors_project = {k: torch.dot(residual_streams[k], translation_vectors_embed[k]) / torch.dot(translation_vectors_embed[k], translation_vectors_embed[k]) * translation_vectors_embed[k] for k in unembedded_vectors.keys()}
    print("Residual norm: ", residual_streams["english"].norm().item())
    print("Projection norm: ", translation_vectors_project["english"].norm().item())
    return translation_vectors_project

def main():
    # Configuration
    clt_checkpoint = "/home/abir19/gpt2_multilingual_90_clt/fsvqfwk0/final_16997376"
    test_strings = [
        'Monday, Tuesday, Wednesday, Thursday,',
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    folder_name = "/home/abir19/FeatFlow/example/graph/save"
    
    print(" Testing Graph Computation Pipeline", flush=True)
    print(f"   Checkpoint: {clt_checkpoint}", flush=True)
    print(f"   Device: {device}", flush=True)
    print("=" * 80, flush=True)
    
    # Get the language vector
    lang_vectors = replace_logit_with_vector(device)
    english_vector = lang_vectors["english"]
    
    for i, test_string in enumerate(test_strings, 1):
        print(f"\n Processing test string {i}: '{test_string}'", flush=True)
        
        try:
            result = run_attribution(
                folder_name=folder_name,
                clt_checkpoint=clt_checkpoint,
                input_string=test_string,
                model_name="CausalNLP/gpt2-hf_multilingual-90",
                max_n_logits=5,
                desired_logit_prob=0.80,
                max_feature_nodes=8192,
                batch_size=256,
                feature_threshold=0.80,
                edge_threshold=0.95,
                device=device,
                offload=None,
                replace_logit_with_vector=english_vector
            )
            
            print(" Attribution computation completed successfully!", flush=True)
            print(f"   - Top logit index: {result['logit_tokens'][0].item()}", flush=True)
            print(f"   - Top logit probability: {result['logit_probabilities'][0].item():.4f}", flush=True)
            print(f"   - Input tokens: {result['input_tokens'].tolist()}", flush=True)
            print(f"   - Language vector used: English projection vector", flush=True)
            
        except Exception as e:
            print(f" Attribution computation failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

        print("\n" + "="*80, flush=True)
    
    print("\n Testing complete!", flush=True)

if __name__ == "__main__":
    main()