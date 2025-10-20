# Commonsense Reasoning Experiments

## Installation

```bash
mamba env create -f environment.yml
conda activate commonsense
```

## Prepare Data

Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows.

```bash
.
└── dataset
    ├── AQuA
    ├── ARC-Challenge
    ├── ARC-Easy
    ├── AddSub
    ├── MultiArith
    ├── SVAMP
    ├── SingleEq
    ├── boolq
    ├── gsm8k
    ├── hellaswag
    ├── mathqa
    ├── mawps
    ├── openbookqa
    ├── piqa
    ├── social_i_qa
    ├── winogrande
    └── commonsense_170k.json
```


## Running Experiments

```bash
./llama3_8b_stella.sh
```

## Acknowledgement

This experiment is modified based on the code in [DoRA](https://github.com/NVlabs/DoRA).
