
# Multilingual Cross-Layer Transcoders (MultilingualCLTs)



![MultilingualCLT Feature Tracing Visualization](assets/gemini_clts.png)


This repository contains the official code for implementing and experimenting with Multilingual Cross-Layer Transcoders (CLTs), the associated Automated Interpretability (AutoInterp) pipeline, designed to study feature composition and communication across transformer layers in multilingual LLMs and the multilingual interface to display these CLTs.

See more:

- 📄 Paper: https://arxiv.org/abs/2511.10840

- 📊 Datasets and Models: https://huggingface.co/collections/CausalNLP/multilingual-gpt2-models and https://huggingface.co/collections/CausalNLP/multilingual-tinystories

- 🧬 Multilingual CLTs: https://huggingface.co/collections/CausalNLP/multilingual-clts

- 🧠 Multilingual Autointerp: https://huggingface.co/collections/CausalNLP/multilingual-autointerp


## Code and Repository Structure Overview

### 1. Core Components & Definitions

These files define the fundamental mechanistic interpretability tools:

* `src/featflow/clt.py`: Defines the core **Cross-Layer Transcoder (CLT)** model architecture.
* `src/featflow/clt_training_runner.py` and `.training/`: Contain the configuration and logic for training the CLT models.
* `src/featflow/causal_graph/`: Defines the **Attribution Graph** structure, which models feature interaction and flow across layers.

### 2. Training Pipelines and Examples

| Path | Purpose |
| :--- | :--- |
| `example/generate_activations/run_activations.py` | Script for generating and storing activations (residual stream) required for CLT training. |
| `example/gpt2_multilingual_20` | Example configuration for launching a full CLT training run on the GPT-2 Multilingual 20M model variant. |
| `example/autointerp/gpt2_multilingual` | Example demonstrating how to run the **Automated Interpretation (AutoInterp)** pipeline on top of a trained CLT. |

### 3. Analysis, Intervention, and Feature Alignment

These scripts are used for deep analysis, feature manipulation, and generating specific experimental data:

* `stitching.py`: Contains logic for **intervening** on the Attribution Graph (e.g., feature suppression/substitution) for causal studies.
* `src/featflow/attribution/alignment_clusters.py`: Implements the feature alignment analysis with English vs Original Language.
* `example/graph/run_aggreg_graph.py`: An example script to generate an **aggregated graph** structure by aggreggating attribution across many prompts.
* `example/graph_langvec`: A focused example to generate an Attribution Graph targeting the **language vector** rather than the full logit space, isolating language-specific decoding features.

### 4. Interactive Interface

* `src/featflow/frontend`: Contains the source code for the multilingual interface.
    * **Execution:** `cd` into the folder, adjust the paths in `src/featflow/frontend/config/settings.py` (pointing to CLT, AutoInterp, and Graph files), and launch using `poetry run python launch.py`.


## 🚀 TODO: Ongoing Work

To further advance this line of research and improve the tools, we plan on pursuing the following improvements:

* **Feature Sharding Optimization:** Improve the current distributed CLT training pipeline by feature sharding.
* **Model Diffing for Low-Resource Languages (LRLs):** Conduct systematic model diffing studies using CLTs trained on models pre-trained on different data distributions to identify which features are missing or misaligned for high-resource languages.
* **Scaling Experiments:** Replicate the CLT training and evaluation protocols on significantly larger multilingual models (e.g., Gemma2-2B, Apertus 8B) to study the scaling behavior of cross-layer feature representation.

### Citation

Please cite our work if you are using our code, datasets, models or CLTs.

```bibtex
@misc{harrasse2025tracingmultilingualrepresentationsllms,
      title={Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders}, 
      author={Abir Harrasse and Florent Draye and Zhijing Jin and Bernhard Schölkopf},
      year={2025},
      eprint={2511.10840},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.10840}, 
}
```
