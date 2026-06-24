# Quantum Kernel Anomaly Detector — Add-on to v9.6d

A quantum-classical hybrid anomaly detection module that extends the Indian Financial Contagion Detection pipeline (v9.6d) with a quantum kernel-based Hypergraph Anomaly Detector (HAD).

---

## What this does

The classical pipeline (v9.6d) detects financial crises using a Gaussian Mixture Model (GMM) applied to hyperedge activity series. It achieves high recall but raises too many false alarms (470 flags over the sample period, precision = 0.011).

This module replaces the GMM anomaly detector with a **quantum kernel one-class SVM**, which maps per-window network features into a 64-dimensional quantum Hilbert space via a ZZ-style feature map. The result is a more selective detector that raises far fewer flags while maintaining meaningful early-warning capability.

### Results vs classical GMM-HAD

| Metric | Quantum Kernel | GMM-HAD (v9.6d) |
|--------|---------------|-----------------|
| Flags raised | 27 | 470 |
| Crises detected | 3 / 5 | 5 / 5 |
| Precision | 0.111 | 0.011 |
| Recall | 0.60 | 1.00 |
| F1 | 0.188 | 0.021 |
| Mean lead time | 38.3 days | 57.0 days |

**Key finding:** The quantum kernel reduces false alarms by 17x and improves precision 10.5x, at the cost of missing 2 crises (COVID Crash 2020 and Banking Stress 2023) which were externally-driven shocks not preceded by domestic cross-sectoral TE network buildup.

---

## Requirements

```bash
pip install pennylane pennylane-lightning scikit-learn numpy
```

| Package | Purpose |
|---------|---------|
| `pennylane` | Quantum circuit definition and execution |
| `pennylane-lightning` | Fast CPU simulator backend |
| `scikit-learn` | One-class SVM, MinMaxScaler |
| `numpy` | Numerical computation |

---

## How to run

**This module is a diagnostic add-on — it cannot run standalone.** It must run in the same Jupyter/Colab session as `indian_financial_contagion_gmm_v9_6d.py` because it reuses variables from that pipeline.

### Step 1 — Run the classical pipeline first

```python
# In your Jupyter or Colab notebook, run v9.6d first
exec(open('indian_financial_contagion_gmm_v9_6d.py').read())
# Wait for: PIPELINE COMPLETE — v9.6d (HAD + RS Fixed)
```

### Step 2 — Verify shared variables exist

```python
required = ['n_windows','all_he','TE_mats','RVS','TE_dates',
            'te_train_mask','CRISIS_PERIODS','N',
            'hyperedge_activity','had_flag_gmm','metrics_gmm']
missing = [v for v in required if v not in dir()]
print('Ready' if not missing else f'Missing: {missing}')
```

### Step 3 — Run the quantum module

```python
exec(open('quantum_kernel_real.py').read())
```

Or paste the code directly into the next notebook cell after v9.6d.

---

## How it works

### 1. Feature extraction
For each of the 477 time windows, a 6-dimensional feature vector is built from the v9.6d pipeline outputs:

| Feature | Description | Source |
|---------|-------------|--------|
| mean TE activity | Average transfer entropy across hyperedges | `all_he`, `TE_mats` |
| max TE activity | Peak contagion signal in window | `TE_mats` |
| mean cardinality | Average hyperedge clique size | `all_he` |
| n hyperedges | Number of hyperedges in window | `all_he` |
| mean RVS | Mean risk virality (PageRank) across 17 vars | `RVS` |
| max RVS | Most viral variable in window | `RVS` |

> **Note:** Feature 4 (mean RVS) has near-zero variance because PageRank normalises to sum=1, so the mean is always ≈ 1/17 = 0.059. The kernel effectively operates on 5 active features. This is a known property noted in the methodology.

### 2. Quantum feature map (ZZ-style)
Features are normalised to [0, π] and encoded into a 6-qubit quantum circuit:

```
For each qubit i:   RY(x[i])              ← angle encoding
For adjacent pairs: CNOT → RZ(x[i]·x[i+1]) → CNOT   ← ZZ entanglement
```

This maps the 6D classical feature vector into a **64-dimensional Hilbert space** (2^6 = 64), capturing pairwise feature interactions that classical distance metrics cannot represent.

### 3. Quantum kernel matrix
The kernel value between two windows x₁ and x₂ is:

```
K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²  =  P(measure |000000⟩)
```

Computed by preparing φ(x₁), applying the adjoint of φ(x₂), and measuring the all-zeros probability. This is the mathematically correct quantum kernel (overlap test). A 477×477 kernel matrix requires ~114,000 circuit evaluations (~8-20 minutes on CPU).

### 4. One-class SVM
A one-class SVM with the precomputed quantum kernel is trained on **training windows only** (≤ 2022-12-31), fitting a decision boundary around normal market behaviour. The anomaly threshold is calibrated at the 95th percentile of training scores — the same philosophy as v9.6d's TAU_Z / TAU_KL calibration.

### 5. Crisis detection
A flag is a **True Positive only if it appears strictly before crisis onset**. Flags during or after onset are not early warnings and are not counted. This is the key methodological fix vs earlier implementations.

---

## File structure

```
quantum-kernel-had/
├── quantum_kernel_real.py          ← this module
├── README.md                       ← this file
└── indian_financial_contagion_gmm_v9_6d.py   ← required: run first
```

---

## Variables inherited from v9.6d

| Variable | Type | Description |
|----------|------|-------------|
| `n_windows` | int | Number of TE windows (477) |
| `all_he` | list of lists | Hyperedges per window |
| `TE_mats` | list of arrays | Transfer entropy matrices per window |
| `RVS` | array (477, 17) | Risk Virality Scores per window |
| `TE_dates` | DatetimeIndex | Date of each window |
| `te_train_mask` | bool array | True for training windows |
| `CRISIS_PERIODS` | list of tuples | (start, end) for each crisis |
| `N` | int | Number of variables (17) |
| `hyperedge_activity` | function | Computes avg TE within a hyperedge |
| `had_flag_gmm` | int array | GMM flags from v9.6d |
| `metrics_gmm` | dict | GMM detection metrics from v9.6d |

---

## Key design decisions

**Why quantum kernel instead of GMM?**
The GMM operates on each hyperedge independently as a 1D time series. The quantum kernel operates on the full 6D per-window feature vector simultaneously, capturing cross-feature interactions (e.g. how TE strength correlates with RVS concentration) that the GMM cannot represent.

**Why one-class SVM?**
Crisis windows are rare in financial time series — labelled crisis data is insufficient to train a binary classifier. One-class SVM learns the boundary of normal behaviour from training data only, then flags deviations in test data. This matches the paper's unsupervised anomaly detection philosophy.

**Why MinMaxScaler to [0, π]?**
RY(θ) gates rotate qubits by angle θ. Mapping features to [0, π] ensures each qubit explores its full rotation range. Fitting the scaler on training data only prevents leakage of test/crisis information into the normalisation.

**Why 95th percentile threshold?**
Same calibration philosophy as v9.6d's TAU_Z and TAU_KL. Training at the 95th percentile means approximately 5% of training windows are expected to be flagged — consistent with the NU=0.05 parameter in the one-class SVM.

---

## Limitations

- **Runtime:** ~8-20 minutes on CPU for the 477×477 kernel matrix. No GPU acceleration (PennyLane lightning.qubit is CPU-only).
- **Missed crises:** COVID Crash 2020 and Banking Stress 2023 are not detected. Both were externally-driven shocks (global pandemic, global rate tightening cycle) with limited domestic TE network buildup in the 90-day pre-onset window.
- **Feature 4 (mean RVS):** Near-zero variance due to PageRank normalisation. The kernel effectively uses 5 active features despite being a 6-qubit circuit.
- **Simulator only:** Results are from a classical simulation of quantum circuits (PennyLane lightning.qubit). Real quantum hardware would introduce noise and decoherence not modelled here.

---

## Citation

If you use this module, please cite the original paper this extends:

> Akgüller, Ö., & Balcı, M. A. (2026). Detecting Financial Contagion Through Higher-Order Networks: A Deep Learning Approach to Emerging Market Risk. *Computational Economics*.

And reference this implementation as an extension applied to Indian financial markets (2015–2025).

---

## Contact

For questions about the quantum kernel implementation, open an issue on this repository.
