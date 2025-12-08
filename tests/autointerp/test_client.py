import pytest
from featflow.autointerp.client import run_client

@pytest.fixture
def test_prompts(tmp_path):
    prompt_paths = []
    for i in range(100):  # Keep small for speed
        path = tmp_path / f"prompt_{i}.txt"
        path.write_text(f"Explain the behavior of neuron {i} in layer 3.")
        prompt_paths.append(path)
    return prompt_paths

def test_run_client_real_model(test_prompts, tmp_path):
    out_dir = tmp_path / "outputs"

    run_client(
        prompts=test_prompts,
        out_dir=out_dir,
        vllm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # or local path
        vllm_max_tokens=64
    )

    # Check all explanations were generated and print them
    for prompt_file in test_prompts:
        output_file = out_dir / prompt_file.name
        assert output_file.exists(), f"{output_file} not generated"

        content = output_file.read_text().strip()
        assert len(content) > 0, "Generated content is empty"

        print(f"\n🔍 Output for {prompt_file.name}:\n{content}\n")
