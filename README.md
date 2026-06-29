<h1 align="center">
    <p> StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold <br> (NeurIPS 2025 Spotlight) </p>
</h1>

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025%20Spotlight-brightgreen)](https://arxiv.org/abs/2510.01938)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)]()

Official implementation of the NeurIPS 2025 Spotlight paper:  
**“StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold”**  
by [Zhizhong Li](https://zhizhong.li/), [Sina Sajadmanesh](https://sisaman.github.io), [Jingtao Li](https://zlijingtao.github.io/), and [Lingjuan Lyu](https://sites.google.com/view/lingjuan-lyu)

[[Paper](https://arxiv.org/abs/2510.01938)] [[BibTex](#citation)]

## Abstract

> Low-rank adaptation (LoRA) has been widely adopted as a parameter-efficient technique for fine-tuning large-scale pre-trained models. However, it still lags behind full fine-tuning in performance, partly due to its insufficient exploitation of the geometric structure underlying low-rank manifolds. In this paper, we propose a geometry-aware extension of LoRA that uses a three-factor decomposition $USV^\top$. Analogous to the structure of singular value decomposition (SVD), it separates the adapter's input and output subspaces, $V$ and $U$, from the scaling factor $S$. Our method constrains $U$ and $V$ to lie on the Stiefel manifold, ensuring their orthonormality throughout the training. To optimize on the Stiefel manifold, we employ a flexible and modular geometric optimization design that converts any Euclidean optimizer to a Riemannian one. It enables efficient subspace learning while remaining compatible with existing fine-tuning pipelines. Empirical results across a wide range of downstream tasks, including commonsense reasoning, math and code generation, image classification, and image generation, demonstrate the superior performance of our approach against the recent state-of-the-art variants of LoRA.

<!-- ## Results -->

## Installation

StelLA extends the Hugging Face PEFT API. Install PEFT and then install this
package from the repository root:

```bash
pip install git+https://github.com/SonyResearch/stella
```

For the commonsense reasoning experiment environment, see
[`experiments/commonsense/README.md`](experiments/commonsense/README.md).

## Usage

StelLA is designed to feel like LoRA in PEFT while replacing the usual two-factor
LoRA adapter with a geometry-aware three-factor adapter, `U S V^T`. The `U` and
`V` factors are constrained to the Stiefel manifold, so training must run the
StelLA optimizer hooks around every optimizer update.

### 1. Import StelLA before creating or loading PEFT adapters

Importing `stella` registers StelLA with PEFT and patches PEFT save/load support:

```python
import stella  # registers the STELLA PEFT type
```

If you import only the classes directly, the patching still happens:

```python
from stella import StellaConfig, StellaTrainer
```

### 2. Create a StelLA adapter

Use `StellaConfig` with PEFT's standard `get_peft_model` function:

```python
from peft import get_peft_model
from stella import StellaConfig

config = StellaConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    stella_retraction="polar",
    stella_diag_s=True,
    init_lora_weights=True,
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

The most important options are:

| Option | Description |
| --- | --- |
| `r` | StelLA rank, analogous to LoRA rank. |
| `lora_alpha` | Adapter scaling factor. |
| `target_modules` | Module names or a regex identifying layers to adapt. Use the same style as PEFT LoRA configs, for example `["q_proj", "v_proj"]` for many decoder-only LMs. |
| `stella_retraction` | Retraction used after optimizer updates. Supported values are `"polar"` and `"exp_map"`. `"polar"` is the default. |
| `stella_diag_s` | If `True`, uses a diagonal `S` factor. |
| `stella_grad_scaling` | If `True`, scales manifold updates using the model hidden size. A float can be provided as a custom scaling factor. |
| `init_lora_weights` | Initialization for the StelLA factors. Use `True` for the default initialization, or one of the explicit initialization modes supported by `StellaConfig`. |

For the complete set of configuration options and accepted values, see
[`stella/tuner/config.py`](stella/tuner/config.py).

### 3. Train with `StellaTrainer`

When using Hugging Face `Trainer`, replace it with `StellaTrainer`. It installs
the optimizer pre-step and post-step hooks that convert Euclidean gradients to
Riemannian gradients and retract `U` and `V` back to the Stiefel manifold after
each update:

```python
from stella import StellaTrainer
from transformers import TrainingArguments

trainer = StellaTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="outputs/stella-run",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        num_train_epochs=3,
        optim="adamw_torch",
        save_safetensors=False,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
```

`StellaTrainer` also excludes the Stiefel-constrained `U` and `V` factors from
weight decay.

### 4. Training with a custom loop

If you do not use `StellaTrainer`, call the model hooks once per optimizer
update:

```python
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()

    model.pre_optimizer_step()
    optimizer.step()
    model.post_optimizer_step()

    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

With gradient accumulation, call the hooks only on steps where `optimizer.step()`
is executed.

### 5. Save, load, and merge adapters

StelLA adapters use the normal PEFT checkpoint APIs after `stella` has been
imported:

```python
model.save_pretrained("outputs/stella-adapter")
```

```python
import stella  # required before loading a STELLA adapter
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "outputs/stella-adapter")
```

For inference, you can merge the adapter into the base model:

```python
merged_model = model.merge_and_unload()
```

## Experiments

Reproducible experiment scripts are available under [`experiments/`](experiments).
For example, the commonsense reasoning setup provides an environment file, data
layout instructions, a fine-tuning script, and evaluation commands.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{li2025stella,
  title={StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold},
  author={Li, Zhizhong and Sajadmanesh, Sina and Li, Jingtao and Lyu, Lingjuan},
  booktitle={Advances in Neural Information Processing Systems},
  publisher = {Curran Associates, Inc.},
  volume = {38},
  year={2025}
}
```
