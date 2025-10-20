# Official code for "StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold" (NeurIPS 2025 Spotlight)

## Abstract

Low-rank adaptation (LoRA) has been widely adopted as a parameter-efficient technique for fine-tuning large-scale pre-trained models. However, it still lags behind full fine-tuning in performance, partly due to its insufficient exploitation of the geometric structure underlying low-rank manifolds. In this paper, we propose a geometry-aware extension of LoRA that uses a three-factor decomposition $USV^\top$. Analogous to the structure of singular value decomposition (SVD), it separates the adapter's input and output subspaces, $V$ and $U$, from the scaling factor $S$. Our method constrains $U$ and $V$ to lie on the Stiefel manifold, ensuring their orthonormality throughout the training. To optimize on the Stiefel manifold, we employ a flexible and modular geometric optimization design that converts any Euclidean optimizer to a Riemannian one. It enables efficient subspace learning while remaining compatible with existing fine-tuning pipelines. Empirical results across a wide range of downstream tasks, including commonsense reasoning, math and code generation, image classification, and image generation, demonstrate the superior performance of our approach against the recent state-of-the-art variants of LoRA. Code is available at https://github.com/SonyResearch/stella.

## Results

## Install

Install the latest `peft` in your project, and then install this package.

```bash
pip install -e .
```

## Usage

Import `stella` in the beginning of your train/eval script.
Stella will be monkey-patched into the `peft` library.

```bash
import stella # the import will monkey-patch peft to support stella
```

Please refer to the examples in the `experiments/` folder for more details.

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
