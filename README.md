<h1 align="center">
    <p> StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold <br> (NeurIPS 2025 Spotlight) </p>
</h1>

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025%20Spotlight-brightgreen)](https://arxiv.org/abs/2510.01938)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)]()

Official implementation of the NeurIPS 2025 Spotlight paper:  
**“StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold”**  
by [Zhizhong Li](https://zhizhong.li/), [Sina Sajadmanesh](https://sisaman.github.io), [Jingtao Li](https://zlijingtao.github.io/), and [Lingjuan Lyu](https://sites.google.com/view/lingjuan-lyu)

[[Paper](https://arxiv.org/abs/2510.01938)] [[BibTex](#citation)]

## Abstract

> Low-rank adaptation (LoRA) has been widely adopted as a parameter-efficient technique for fine-tuning large-scale pre-trained models. However, it still lags behind full fine-tuning in performance, partly due to its insufficient exploitation of the geometric structure underlying low-rank manifolds. In this paper, we propose a geometry-aware extension of LoRA that uses a three-factor decomposition $USV^\top$. Analogous to the structure of singular value decomposition (SVD), it separates the adapter's input and output subspaces, $V$ and $U$, from the scaling factor $S$. Our method constrains $U$ and $V$ to lie on the Stiefel manifold, ensuring their orthonormality throughout the training. To optimize on the Stiefel manifold, we employ a flexible and modular geometric optimization design that converts any Euclidean optimizer to a Riemannian one. It enables efficient subspace learning while remaining compatible with existing fine-tuning pipelines. Empirical results across a wide range of downstream tasks, including commonsense reasoning, math and code generation, image classification, and image generation, demonstrate the superior performance of our approach against the recent state-of-the-art variants of LoRA.

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
