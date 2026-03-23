# DG-LLM Configurations



## 1. Data Preprocessing Pipeline

To ensure consistent and leakage-free evaluation, we follow a rigorous preprocessing protocol:

- **Sliding Window Generation**: Raw traffic time series are transformed into overlapping samples using a sliding window with a step size of 1. Each sample consists of an input window of length $T_{in}$ and a target window of length $T_{out}$.
- **Normalization**: We utilize a `StandardScaler` (Z-score normalization) computed exclusively on the training set to prevent data leakage. The same mean and standard deviation are applied to validate and test sets.
- **VMD Decomposition**: Variational Mode Decomposition is performed **per-sample** on the input flow sequences. This is a critical design choice to ensure that the decomposition only uses information available at the time of prediction, preventing temporal leakage from future data points in the decomposition process.
- **Temporal Encoding**: Two additional features are generated to provide periodic context:
  - **Time-of-Day**: Normalized $[0, 1]$ based on the index within a 24-hour cycle (e.g., $t/288$ for 5-min intervals).
  - **Day-of-Week**: Integer encoding (0-6) mapped to learned embedding vectors.

## 2. Training Configurations

- **Optimization**: We employ the **Ranger** optimizer, which integrates **RAdam** (Rectified Adam) for robust initial convergence and **Lookahead** (k=5, alpha=0.5) to improve stability and prevent local minima.
- **Mixed Precision (AMP)**: The framework automatically detects GPU capabilities to enable `torch.amp`. It uses `BF16` on Ampere+ architectures (A100, L4) and `FP16` with a `GradScaler` on older architectures (T4, V100), ensuring efficient training without loss of numerical stability.
- **Gradient Accumulation**: To maintain consistent effective batch sizes across different hardware, we support `grad_accum_steps`. The effective batch size is $Batch\_Size \times Accumulation\_Steps$.
- **Loss Calculation**: While the model optimizes normalized values, the reported metrics (MAE, RMSE, MAPE) and the validation loss are calculated on the **descaled** (original scale) traffic values for physical interpretability.


## 3. Hyperparameters

### 1. Training Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Optimizer | Ranger | Combination of RAdam and Lookahead |
| Learning Rate | $1 \times 10^{-3}$ | Initial learning rate |
| Weight Decay | $1 \times 10^{-5}$ | L2 regularization factor |
| Batch Size | 8 | Multiplied by gradient accumulation steps |
| Gradient Accumulation | 1 | Steps to accumulate gradients |
| Epochs | 100 | Maximum training iterations |
| Loss Function | MAE | Mean Absolute Error on unscaled values |
| Gradient Clipping | 5.0 | Max norm for gradient clipping |
| Seed | 42 | Default random seed |

### 2. Dynamic Graph Learning

| Parameter | Value | Description |
| :--- | :--- | :--- |
| GAT Heads | 4 | Number of attention heads for adjacency learning |
| Warmup Steps | 500 | Union with fixed graph during initial steps |
| Edge Dropout | 0.1 | Randomly drop edges during training |
| Head Dropout | 0.1 | Mask entire GAT heads for robustness |
| Symmetrize | True | Ensure $A = \max(A, A^T)$ |
| Hysteresis Ratio | 0.8 | Threshold for retaining previous graph edges |
| EMA Decay ($\beta$) | 0.99 | Smoothing factor for learned adjacency |
| Adaptive Density ($p$) | 0.15 | Target sparsity ratio for binary graph |

### 3. LLM Backbone

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Backbone Model | GPT-2 (base) | Small variant (12 layers, 768 dim, 12 heads) |
| LoRA Rank ($r$) | 16 | Rank of Low-Rank Adaptation matrices |
| LoRA Alpha ($\alpha$) | 32 | Scaling factor for LoRA updates |
