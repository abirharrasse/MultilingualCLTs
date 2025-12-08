import pytest
from featflow.autointerp.pipeline import AutoInterp
from tests.utils import build_autointerp_cfg, build_clt_training_runner_cfg
from pathlib import Path
from featflow.config import CLTTrainingRunnerConfig, CLTConfig
from featflow.clt import CLT
from featflow.utils import PROMPTS_FOLDERNAME, EXPLANATIONS_FOLDERNAME

# Get the directory of the current file
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

@pytest.fixture(
    params=[
        {
            "model_name": "roneneldan/TinyStories-33M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "cross_layer_decoders": False,
            "d_in": 768,
        }
    ]
)
def autointerp_cfg(request: pytest.FixtureRequest):
    return build_autointerp_cfg(**request.param)

@pytest.fixture(
    params=[
        {
            "model_name": "roneneldan/TinyStories-33M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "cross_layer_decoders": False,
            "d_in": 768,
        }
    ]
)
def clt_training_cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of CLTTrainingRunnerConfig.
    """
    params = request.param
    return build_clt_training_runner_cfg(**params)

@pytest.fixture(params=[4])
def clt_saving(clt_training_cfg: CLTTrainingRunnerConfig, request, tmp_path):
    n_layers = request.param
    clt_config = clt_training_cfg.create_sub_config(
        CLTConfig, 
        n_layers=n_layers
    )
    clt = CLT(clt_config)
    clt.save_model(tmp_path)
    return clt

# def test_model_init(autointerp_cfg, clt_saving, tmp_path):
#     # The clt_saving fixture creates and saves a CLT model to tmp_path
#     autointerp_cfg.clt_path = tmp_path
#     autointerp = AutoInterp(autointerp_cfg)
    
#     # Add some basic assertions to verify it worked
#     assert autointerp is not None
#     assert autointerp.clt is not None

# def test_run(autointerp_cfg, clt_saving, tmp_path):
#     # The clt_saving fixture creates and saves a CLT model to tmp_path
#     autointerp_cfg.clt_path = str(tmp_path)
#     autointerp_cfg.latent_cache_path = str(tmp_path / "latent_cache")
#     autointerp = AutoInterp(autointerp_cfg)
#     autointerp.run()

#     latent_files = sorted(Path(autointerp_cfg.latent_cache_path).glob("*.safetensors"))
#     assert latent_files, "No latent cache files found."

#     first_file = latent_files[0]
#     data = load_file(first_file)
#     activations = data["activations"]

#     # --- CHECK DIMENSIONS ---
#     print(f"Loaded activations shape from {first_file.name}: {activations.shape}")
    
#     # Test the shape structure: [n_seq, ctx, n_layers, d_latent]
#     assert activations.ndim == 4, "Expected activations to have 4 dimensions"
#     n_seq, ctx, d1, d2 = activations.shape
#     print(f"n_seq: {n_seq}, ctx: {ctx}, d1: {d1}, d2: {d2}")
#     assert ctx == autointerp_cfg.context_size, f"Context size mismatch: expected {autointerp_cfg.context_size}, got {ctx}"
#     assert d1 == 4
#     assert d2 == 4

# def test_run_prompts(autointerp_cfg, clt_saving, tmp_path):
#     autointerp_cfg.clt_path = str(tmp_path)
#     autointerp_cfg.latent_cache_path = str(tmp_path / "latent_cache")
#     autointerp_cfg.total_autointerp_tokens = 1000 
#     autointerp = AutoInterp(autointerp_cfg)
#     autointerp.run()
    
#     layer = 0
#     top_k = 5
    
#     autointerp.run_prompts(layer=layer, top_k=top_k)
    
#     # Verify outputs
#     prompt_dir = Path(tmp_path / "latent_cache" / PROMPTS_FOLDERNAME / f"layer{layer}")
#     assert prompt_dir.exists(), "Prompt directory should be created"
    
#     # Check that prompt files were created
#     prompt_files = sorted(prompt_dir.glob("explanation_*.txt"))
#     assert len(prompt_files) > 0, "Should create at least one prompt file"
    
#     print(f"\n=== FOUND {len(prompt_files)} PROMPT FILES ===")
    
#     # Print complete content of all files (should be 4 since d_latent=4)
#     for i, prompt_file in enumerate(prompt_files):
#         content = prompt_file.read_text()
#         print(f"\n{'='*60}")
#         print(f"FILE {i+1}: {prompt_file.name}")
#         print(f"{'='*60}")
#         print(content)
#         print(f"{'='*60}")
        
#         # Basic assertions
#         assert len(content) > 0, f"Prompt file {prompt_file} should not be empty"
    
#     # Should have exactly 4 files (one for each feature)
#     assert len(prompt_files) == 4, f"Expected 4 files for 4 features, got {len(prompt_files)}"
#     assert False

def test_run_explanations(autointerp_cfg, clt_saving, tmp_path):
    autointerp_cfg.clt_path = str(tmp_path)
    autointerp_cfg.latent_cache_path = str(tmp_path / "latent_cache")
    autointerp_cfg.total_autointerp_tokens = 1000 
    autointerp = AutoInterp(autointerp_cfg)
    autointerp.run()
    
    layer = 0
    top_k = 5
    
    autointerp.run_prompts(layer=layer, top_k=top_k)
    autointerp.run_explanations(layer=layer)

    # Verify outputs
    prompt_dir = Path(tmp_path / "latent_cache" / PROMPTS_FOLDERNAME / f"layer{layer}")
    assert prompt_dir.exists(), "Prompt directory should be created"

    # Check that prompt files were created
    prompt_files = sorted(prompt_dir.glob("explanation_*.txt"))
    assert len(prompt_files) > 0, "Should create at least one prompt file"

    print(f"\n=== FOUND {len(prompt_files)} PROMPT FILES ===")

    # Print complete content of all files (should be 4 since d_latent=4)
    for i, prompt_file in enumerate(prompt_files):
        content = prompt_file.read_text()
        print(f"\n{'='*60}")
        print(f"FILE {i+1}: {prompt_file.name}")
        print(f"{'='*60}")
        print(content)
        print(f"{'='*60}")
    
        # Basic assertions
        assert len(content) > 0, f"Prompt file {prompt_file} should not be empty"

    explanations_dir = Path(autointerp.cfg.latent_cache_path) / EXPLANATIONS_FOLDERNAME / f"layer{layer}"
    assert explanations_dir.exists(), "Explanations directory should be created"

    # Check that prompt files were created
    explanations_files = sorted(explanations_dir.glob("explanation_*.txt"))
    assert len(explanations_files) > 0, "Should create at least one prompt file"
    
    print(f"\n=== FOUND {len(explanations_files)} PROMPT FILES ===")
    
    # Print complete content of all files (should be 4 since d_latent=4)
    for i, explanation_file in enumerate(explanations_files):
        content = explanation_file.read_text()
        print(f"\n{'='*60}")
        print(f"FILE {i+1}: {explanation_file.name}")
        print(f"{'='*60}")
        print(content)
        print(f"{'='*60}")
        
        # Basic assertions
        assert len(content) > 0, f"Prompt file {explanation_file} should not be empty"
    
    # Should have exactly 4 files (one for each feature)
    assert len(explanations_files) == 4, f"Expected 4 files for 4 features, got {len(explanations_files)}"
    assert False

# @pytest.mark.parametrize("sentence,high_indices,expected_spans", [
#     ("The cat sat on the mat", [1, 2, 5], ["<<cat sat>>", "<<mat>>"]),
#     ("Jump now, quickly!", [0, 3], ["<<Jump>>", "<<quickly>>"]),
#     ("Wait, stop. Think again.", [0, 4, 5], ["<<Wait>>", "<<Think again>>"]),
#     ("Unbreakable code", [0], ["<<Un>>"])
# ])
# def test_highlight_activations_dynamic(sentence, high_indices, expected_spans):
#     tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#     encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
#     tokens = encoding["input_ids"][0]
#     num_tokens = len(tokens)

#     # high activation for selected tokens
#     activations = torch.rand(num_tokens) * 0.5
#     for idx in high_indices:
#         if idx < num_tokens:
#             activations[idx] = random.uniform(0.8, 1.0)

#     result = highlight_activations(tokens, activations, tokenizer, threshold_ratio=0.6)

#     print(f"\n Sentence: {sentence}")
#     print(f"🔍 Highlighted Output:\n{result}")

#     for span in expected_spans:
#         assert span in result, f"Missing expected span: {span}"

#     if not expected_spans:
#         assert "<<" not in result and ">>" not in result
