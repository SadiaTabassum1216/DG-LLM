# DG-LLM: Dynamic Graph-aware Large Language Model for Traffic Prediction

A spatio-temporal traffic prediction model that combines **Variational Mode Decomposition (VMD)** with a **GPT-2 backbone** enhanced by **dynamic graph attention**.

## Architecture Overview

```
Input Data → VMD Decomposition → [Mode 1, Mode 2, Mode 3]
                                         ↓
                              Each mode processed by:
                              ┌──────────────────────────────────┐
                              │  SingleMode_Dynamic_STLLM        │
                              │  ├─ Temporal Embeddings          │
                              │  ├─ Multi-scale Conv             │
                              │  ├─ Dynamic Graph Learning (GAT) │
                              │  └─ GPT-2 with LoRA (PFA)        │
                              └──────────────────────────────────┘
                                         ↓
                              Attention Fusion → Prediction
```

## Key Features

- **VMD**: Per-sample decomposition (no data leakage between train/val/test)
- **Dynamic Graph Learning**: Learned adjacency via multi-head GAT with curriculum learning
- **GPT-2 Backbone**: Pre-trained LLM adapted for time series with LoRA fine-tuning
- **Gradient Checkpointing**: Memory-efficient training for large models

## Installation

```bash
pip install torch transformers peft vmdpy tqdm numpy
```

## Project Structure

```
DG-LLM/
├── DGLLM.ipynb          # Main training notebook
├── Dataset/
├── log/
│   └── best_model.pth   # Saved model checkpoint
└── vmd_cache/           # Cached VMD decompositions
```

## Dataset Download

The datasets are available on Google Drive:

📥 **[Download Datasets](https://drive.google.com/file/d/19LkZXBCS7E2SCuM2ZQ7YKT7L0-wMXrJa/view?usp=sharing)**

After downloading, extract the contents to the `Dataset/` folder.

## Dataset Format

Each dataset should contain:
- `train.npz`, `val.npz`, `test.npz` with keys `x` and `y`
  - `x`: Input features `[Samples, 12, Nodes, Features]`
  - `y`: Target values `[Samples, 12, Nodes, 1]`
- `adj_mx.pkl`: Adjacency matrix `[Nodes, Nodes]`

## Training

Open `DGLLM.ipynb` and run all cells. Key configurations:

```python
args = Args(
    data="taxi_pick",        # Dataset name
    num_nodes=266,           # Number of nodes
    input_len=12,            # Input sequence length
    output_len=12,           # Prediction horizon
    batch_size=32,
    epochs=100,
    lrate=1e-3,
)
```


## Model Components

| Component | Description |
|-----------|-------------|
| `GATLLM` | Main model combining K VMD mode branches |
| `SingleMode_Dynamic_STLLM` | Single mode processor with dynamic graph |
| `PFA` | GPT-2 with LoRA and graph attention |
| `TemporalEmbedding` | Learnable time-of-day/day-of-week embeddings |
