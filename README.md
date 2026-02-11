# DG-LLM: Decomposition-based Dynamic Graph Adaptation of Large Language Models for Spatiotemporal Traffic Forecasting

## Abstract

Traffic forecasting is a crucial part of urban planning, and it is critical to understand the evolution of urban traffic patterns. However, the current state-of-the-art models struggle to capture the spatiotemporal dependency of traffic data, and struggle with long-range dependency capturing because of its inherent entangled multi-scale nature. This paper proposes a novel spatiotemporal forecasting framework named DG-LLM that bridges the gap between signal decomposition, dynamic graph learning, and the reasoning capabilities of pretrained Large Language Models (LLMs). Our approach explicitly decomposes traffic signals into intrinsic temporal modes, learns mode-dependent dynamic graphs, and integrates these structures into a pretrained LLM using spatially constrained attention and efficient fine-tuning strategies. We conducted comprehensive experiments on six real world datasets, covering both grid-based and graph-structured traffic networks, and performed both short term and long term forecasting analysis. Experimental results demonstrate that our approach consistently outperforms benchmark models across diverse datasets and forecasting horizons. Our model achieves an overall average improvement of **14.09%** in MAE and **20.88%** in RMSE across all datasets with the benchmark models. Comprehensive ablation studies further validate the effectiveness of each component, highlighting the benefits of multi-scale temporal decomposition, dynamic spatial modeling, and parameter-efficient LLM adaptation. Furthermore, by utilizing Low-Rank Adaptation (LoRA), we demonstrate that the expansive knowledge within LLM backbones can be harnessed efficiently, reducing the parameter overhead associated with fine tuning large-scale transformers.

## Architecture

![DG-LLM Architecture](Images/architecture.png)

## Key Features

- **VMD**: Per-sample decomposition (no data leakage between train/val/test)
- **Dynamic Graph Learning**: Learned adjacency via multi-head GAT with curriculum learning
- **GPT-2 Backbone**: Pre-trained LLM adapted for time series with LoRA fine-tuning
- **Gradient Checkpointing**: Memory-efficient training for large models

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**: 
   - Download datasets from the link below
   - Extract to `./Dataset/<dataset_name>/`
   - Each dataset folder should contain:
     - `adj_mx.pkl` - Adjacency matrix
     - `processed/` folder with `train.npz`, `val.npz`, `test.npz`

3. **Run Training**:
   ```bash
   python main.py
   ```

## Dataset Download

The datasets are available on Google Drive: **[Datasets](https://drive.google.com/file/d/19LkZXBCS7E2SCuM2ZQ7YKT7L0-wMXrJa/view?usp=sharing)**

## Run on Kaggle

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sadia1216/dgllm)

## Dataset Format

```
Dataset/
├── PEMSD04/
│   ├── adj_mx.pkl          # Adjacency matrix [307, 307]
│   └── processed/
│       ├── train.npz       # x: [Samples, 12, 307, F], y: [Samples, 12, 307, 1]
│       ├── val.npz
│       └── test.npz
├── PEMSD08/                # 170 nodes
├── bike_drop/              # 250 nodes
├── bike_pick/              # 250 nodes
├── taxi_drop/              # 266 nodes
└── taxi_pick/              # 266 nodes
```

## Model Components

| Component | Description |
|-----------|-------------|
| `DGLLM` | Main model combining K VMD mode branches |
| `TemporalEmbedding` | Learnable time-of-day/day-of-week embeddings |
| `ModeProcessor` | Single mode processor with dynamic graph |
| `PFA` | GPT-2 with LoRA and graph attention |


## Acknowledgements

This project is built upon the work of [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM). We thank the authors for providing the base code and datasets.

## Citation

Paper is currently under review. Citation information will be updated upon publication.
