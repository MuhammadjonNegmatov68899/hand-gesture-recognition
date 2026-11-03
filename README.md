# Hand Gesture Recognition on CPU-Only Devices

Real-time hand gesture recognition on CPU-only devices: accuracy–latency trade-offs of landmark-based vs. CNN models with quantization.

This component implements **Person 2: Landmark-Based Approach**.

---

## 1. Landmark Pipeline Architecture

The landmark-based recognition pipeline operates in four deterministic stages:

```
[Webcam Frame / Video]
        │
        ▼ (BGR image, 720p/1080p)
┌───────────────────────────────────────────────┐
│ MediaPipe Tasks Hand Landmarker (CPU)        │
│  - 21 3D hand landmarks (x, y, z)             │
│  - Handedness classification & selfie flip    │
└───────────────────────────────────────────────┘
        │
        ▼ (21 landmarks, shape (21, 3))
┌───────────────────────────────────────────────┐
│ Landmark Coordinate Normalizer                │
│  - Wrist-relative translation (p_i - p_0)     │
│  - Anatomical reference scaling (||p_12||)    │
│  - Output: 63-D float32 feature vector        │
└───────────────────────────────────────────────┘
        │
        ▼ (63-dimensional normalized features)
┌───────────────────────────────────────────────┐
│ Lightweight Classifiers                       │
│  - Random Forest (RandomForestClassifier)     │
│  - MLP (MLPClassifier, 2 hidden layers)       │
└───────────────────────────────────────────────┘
        │
        ▼
[Predicted Gesture Class Label]
```

### Latency Decomposition
In order to accurately evaluate CPU latency trade-offs against quantized CNN models (Person 3), latency is decomposed into individual stages:

$$T_{\text{total}} = T_{\text{MediaPipe}} + T_{\text{normalization}} + T_{\text{classifier}}$$

- $T_{\text{MediaPipe}}$: Frame inference through Google MediaPipe Tasks Vision HandLandmarker.
- $T_{\text{normalization}}$: Wrist translation and scale reference normalization.
- $T_{\text{classifier}}$: Feature inference time through Random Forest or MLP.

---

## 2. Coordinate Normalization Methodology

To ensure robust classification invariant to camera distance, hand positioning, and subject hand size variations, coordinates undergo a two-step normalization:

### 1. Translation Normalization (Wrist-Centric)
Given the 21 landmarks $P = [p_0, p_1, \dots, p_{20}]^T$ where $p_i = (x_i, y_i, z_i)$, translation invariance is achieved by subtracting the wrist position ($p_0$, landmark index 0):

$$p'_i = p_i - p_0 \quad \forall i \in \{0, \dots, 20\}$$

After translation, the wrist coordinate is identically centered at the origin: $p'_0 = (0, 0, 0)$.

### 2. Scale Normalization (Anatomical Reference Distance)
To achieve scale invariance across varying distances to the camera and physical differences in hand size, coordinates are scaled by the Euclidean distance between the wrist ($p_0$) and the middle fingertip ($p_{12}$, landmark index 12):

$$d_{\text{ref}} = \|p'_{12}\|_2 = \sqrt{(x'_{12})^2 + (y'_{12})^2 + (z'_{12})^2}$$

Each centered coordinate is normalized by this reference distance:

$$p''_i = \frac{p'_i}{d_{\text{ref}} + \epsilon}$$

where $\epsilon = 10^{-7}$ prevents division by zero. If $d_{\text{ref}} < \epsilon$ (degenerate or collapsed hand input), scaling defaults to unit scale ($1.0$) to prevent numerical divergence.

### 3. Feature Vector Output
The normalized landmarks $P'' \in \mathbb{R}^{21 \times 3}$ are flattened into a deterministic 63-dimensional vector of 32-bit floating point values (`np.float32`):

$$\mathbf{x} = [x''_0, y''_0, z''_0, x''_1, y''_1, z''_1, \dots, x''_{20}, y''_{20}, z''_{20}] \in \mathbb{R}^{63}$$

---

## 3. Dataset Format Expected from Person 1

> [!IMPORTANT]
> **Dataset Status: PENDING.**
> The official shared dataset and splits have not yet been provided by Person 1.
> Both Random Forest and MLP must use **the exact same dataset and train/validation/test split** as Person 3's CNN models to guarantee fair, scientifically valid comparison.

The `DatasetLoader` (`src/data/dataset.py`) autodetects and supports any of the following formats placed in the `data/` directory:

### Option A (Recommended): NumPy NPZ Archive (`data/dataset.npz`)
A compressed `.npz` file containing:
- `X_train`: shape `(N_train, 63)` or `(N_train, 21, 3)`
- `y_train`: shape `(N_train,)` containing class label strings or integer IDs
- `X_test`: shape `(N_test, 63)` or `(N_test, 21, 3)`
- `y_test`: shape `(N_test,)`
- Optional `X_val`, `y_val`: validation split
- Optional `classes`: list of class names in order

### Option B: Separate CSV Files (`data/train.csv`, `data/test.csv`, `data/val.csv`)
CSV files with:
- 63 landmark feature columns (e.g. `x0, y0, z0, ..., x20, y20, z20`)
- One `label` column containing gesture class names

### Option C: Single CSV with Split Column (`data/dataset.csv`)
A unified CSV containing:
- A `split` column with values `train`, `val` (or `validation`), `test`
- A `label` column
- 63 landmark feature columns

---

## 4. Setup and Installation

### 1. Prerequisites
- Python 3.11
- Virtual environment (`.venv`)

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## 5. Model Training

Once Person 1 provides the shared dataset in `data/`, train the models using the commands below.

### Train Random Forest
```bash
python -m src.models.random_forest --data-dir data --output models/random_forest.joblib
```

**Configurable Options:**
- `--n-estimators`: Number of decision trees (default: `100`).
- `--max-depth`: Maximum tree depth (default: `None` for full expansion).
- `--random-state`: Seed for reproducible training (default: `42`).
- `--n-jobs`: CPU threads for training (default: `-1` for all available cores).

### Train MLP (Multilayer Perceptron)
```bash
python -m src.models.mlp --data-dir data --output models/mlp.joblib
```

**Configurable Options:**
- `--hidden-layers`: Architecture layer sizes (default: `64,32`).
- `--max-iter`: Maximum training epochs (default: `300`).
- `--learning-rate`: Initial learning rate (default: `0.001`).
- `--alpha`: L2 regularization penalty (default: `0.0001`).
- `--random-state`: Seed for reproducible weight initialization (default: `42`).

---

## 6. Model Evaluation and Latency Benchmarking

Run the evaluation script to compute classification metrics and latency breakdowns:

### Evaluate Random Forest
```bash
python -m src.evaluation.evaluate --model-path models/random_forest.joblib --model-type rf --data-dir data
```

### Evaluate MLP
```bash
python -m src.evaluation.evaluate --model-path models/mlp.joblib --model-type mlp --data-dir data
```

### Full End-to-End Pipeline Latency Benchmark
Include `--benchmark-e2e` along with `--benchmark-image <path/to/hand_image.jpg>` to measure the complete pipeline latency ($T_{\text{total}} = T_{\text{MediaPipe}} + T_{\text{normalization}} + T_{\text{classifier}}$):

```bash
python -m src.evaluation.evaluate \
  --model-path models/random_forest.joblib \
  --model-type rf \
  --data-dir data \
  --benchmark-e2e \
  --benchmark-image path/to/sample_hand.jpg \
  --iterations 500 \
  --threads 1 \
  --output-json results/rf_benchmark.json
```

> [!NOTE]
> `--benchmark-e2e` requires a real hand image supplied via `--benchmark-image` to ensure MediaPipe executes both palm detection and landmark regression neural networks. Blank or synthetic frames are rejected.

### Output Metrics Reported
- **Classification Performance**:
  - Accuracy
  - Macro F1-score
  - Macro Precision and Recall
  - Confusion Matrix
- **Resource Footprint**:
  - Model file size (in KB)
  - Training duration (in seconds)
- **Latency Breakdown**:
  - Classifier inference latency: Mean, P95, P99, and FPS
  - End-to-end pipeline latency: MediaPipe vs. Normalization vs. Classifier Mean, P95, and FPS

---

## 7. Hand Detector Verification

Run the detector synthetic smoke test (does not require a webcam):
```bash
python -m src.landmarks.hand_detector --test-synthetic
```

To run real-time detection on a live webcam feed:
```bash
python -m src.landmarks.hand_detector --camera 0
```

---

## 8. Running Unit Tests

Execute the complete test suite:
```bash
pytest
```

For verbose output with individual test execution details:
```bash
pytest -v
```
