# DG-LLM: Decomposition-based Dynamic Graph Adaptation of Large Language Models for Spatiotemporal Traffic Forecasting

A spatio-temporal traffic prediction model that combines **Variational Mode Decomposition (VMD)** with a **GPT-2 backbone** enhanced by **dynamic graph attention**.

## Architecture

![DG-LLM Architecture](Images/architecture.png)
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
├── log/
│   └── best_model.pth   # Saved model checkpoint
└── vmd_cache/           # Cached VMD decompositions
```

## Dataset Download

The datasets are available on Google Drive: **[Datasets](https://drive.google.com/file/d/19LkZXBCS7E2SCuM2ZQ7YKT7L0-wMXrJa/view?usp=sharing)**

## Dataset Format

Each dataset should contain:
- `train.npz`, `val.npz`, `test.npz` with keys `x` and `y`
  - `x`: Input features `[Samples, 12, Nodes, Features]`
  - `y`: Target values `[Samples, 12, Nodes, 1]`
- `adj_mx.pkl`: Adjacency matrix `[Nodes, Nodes]`

## Run Model

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

## Acknowledgements

This project is built upon the work of [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM). We thank the authors for providing the base code and datasets.

## Citation

Paper is currently under review. Citation information will be updated upon publication.
