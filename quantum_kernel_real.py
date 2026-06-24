"""
================================================================
QUANTUM KERNEL ANOMALY DETECTOR — diagnostic add-on to v9.6d
================================================================
Standalone module. Run AFTER v9.6d has executed in the same
session (reuses: all_he, TE_mats, RVS, n_windows, TE_dates,
te_train_mask, CRISIS_PERIODS, N from v9.6d's namespace).

Purpose: test whether a quantum kernel operating on the full
multi-dimensional per-window feature vector can better separate
crisis from non-crisis windows than the classical 1D per-hyperedge
GMM used in HAD (v9.6d). This is a DIAGNOSTIC, not a claimed fix —
report results honestly whichever way they go (see chat discussion).

Requires: pennylane, pennylane-lightning, scikit-learn
  pip install pennylane pennylane-lightning scikit-learn --break-system-packages
================================================================
"""
import numpy as np
import pennylane as qml
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

print("\n[Quantum Kernel HAD] Starting diagnostic comparison...")

# ════════════════════════════════════════════════════════════
# STEP 1-2: Extract + normalise per-window feature vectors
# ════════════════════════════════════════════════════════════
# Reuses classical features already computed in v9.6d Steps 2-3.
# Feature vector per window (6 dims):
#   1. mean TE strength across all hyperedges in window
#   2. max TE strength (peak activity)
#   3. mean hyperedge cardinality (avg clique size)
#   4. number of hyperedges in window
#   5. mean RVS (risk virality / PageRank) across all 17 vars
#   6. max RVS (single most viral variable)

def weighted_hyperdeg_window(he_list, n):
    d = np.zeros(n, dtype=float)
    for h in he_list:
        size = len(h)
        for nd in h:
            d[nd] += size
    return d

print("  Building feature vectors from classical pipeline outputs...")
feat_list = []
for w in range(n_windows):
    he_w  = all_he[w]
    TE_w  = TE_mats[w]
    if he_w:
        acts = [hyperedge_activity(h, TE_w) for h in he_w]
        mean_act  = float(np.mean(acts))
        max_act   = float(np.max(acts))
        mean_card = float(np.mean([len(h) for h in he_w]))
        n_he      = float(len(he_w))
    else:
        mean_act = max_act = mean_card = n_he = 0.0
    mean_rvs = float(RVS[w].mean())
    max_rvs  = float(RVS[w].max())
    feat_list.append([mean_act, max_act, mean_card, n_he, mean_rvs, max_rvs])

X_feat = np.array(feat_list)  # (n_windows, 6)
print(f"  Feature matrix shape: {X_feat.shape}")

# Normalise to [0, pi] for angle encoding (fit on training windows only)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_feat_scaled = np.zeros_like(X_feat)
X_feat_scaled[te_train_mask] = scaler.fit_transform(X_feat[te_train_mask])
X_feat_scaled[~te_train_mask] = scaler.transform(X_feat[~te_train_mask])

N_QUBITS = X_feat.shape[1]   # 6 qubits, one per feature
print(f"  Normalised to [0, pi], {N_QUBITS} qubits (one per feature)")

# ════════════════════════════════════════════════════════════
# STEP 3: Quantum feature map circuit (PennyLane)
# ════════════════════════════════════════════════════════════
# ZZ-style feature map: angle-encode each feature on its own qubit,
# then entangle adjacent qubits with CNOT + RZ(product of features).
# This is what lets the kernel capture pairwise feature correlations
# that a classical 1D GMM (used in HAD) cannot represent.

dev = qml.device("lightning.qubit", wires=N_QUBITS)

def feature_map(x):
    """Angle-encode + entangle. x: array of length N_QUBITS, in [0, pi]."""
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(x[i] * x[i + 1], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """Overlap circuit: prepare phi(x1), then inverse-prepare phi(x2).
    Probability of measuring |0...0> = |<phi(x1)|phi(x2)>|^2 = kernel value.
    """
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(N_QUBITS))

def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]   # P(all zeros) = overlap

print(f"  Quantum feature map circuit defined ({N_QUBITS} qubits, "
      f"{N_QUBITS-1} entangling pairs)")

# ════════════════════════════════════════════════════════════
# STEP 4: Compute the kernel matrix
# ════════════════════════════════════════════════════════════
# Full N x N matrix of pairwise quantum kernel values.
# For 477 windows this is ~113,500 unique circuit evaluations
# (upper triangle + diagonal), each a few ms on lightning.qubit.

print(f"  Computing {n_windows}x{n_windows} quantum kernel matrix "
      f"(this may take a few minutes on CPU)...")

import time
t0 = time.time()
K_full = np.zeros((n_windows, n_windows))
for i in range(n_windows):
    K_full[i, i] = 1.0   # self-overlap is always 1
    for j in range(i + 1, n_windows):
        k_val = quantum_kernel(X_feat_scaled[i], X_feat_scaled[j])
        K_full[i, j] = k_val
        K_full[j, i] = k_val   # symmetric
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"    row {i+1}/{n_windows}  {elapsed:.1f}s")

print(f"  Kernel matrix complete: {time.time()-t0:.1f}s total")

# ════════════════════════════════════════════════════════════
# STEP 5: Train one-class SVM on quantum kernel (training windows)
# ════════════════════════════════════════════════════════════
# Trains the decision boundary on NORMAL (non-crisis, training-period)
# windows only — same philosophy as HAD-GMM which fits the baseline
# on training data. nu = expected fraction of outliers.

print("  Training one-class SVM on quantum kernel matrix...")

train_idx = np.where(te_train_mask)[0]
test_idx  = np.where(~te_train_mask)[0]

K_train = K_full[np.ix_(train_idx, train_idx)]
K_test  = K_full[np.ix_(test_idx, train_idx)]

NU = 0.05   # assume ~5% of training windows are borderline-anomalous
ocsvm = OneClassSVM(kernel='precomputed', nu=NU)
ocsvm.fit(K_train)

# ════════════════════════════════════════════════════════════
# STEP 6: Score all windows (train + test) using decision_function
# ════════════════════════════════════════════════════════════
# decision_function > 0 = inlier (normal), < 0 = outlier (anomalous).
# More negative = more anomalous. We score ALL windows (not just
# test) so the crisis-detection metrics can be computed the same
# way as the classical HAD-GMM (which flags across the whole series).

K_all_vs_train = K_full[:, train_idx]
qk_scores = ocsvm.decision_function(K_all_vs_train)   # higher = more normal
qk_anomaly_score = -qk_scores   # flip sign: higher = more anomalous

print(f"  Scored all {n_windows} windows "
      f"(train + test) via quantum kernel SVM")

# ════════════════════════════════════════════════════════════
# STEP 7: Calibrate threshold, produce flags (same logic as HAD-GMM)
# ════════════════════════════════════════════════════════════
# Use the same percentile-based calibration philosophy as v9.6d's
# classical HAD: threshold calibrated on TRAINING anomaly scores only.

qk_thresh = np.percentile(qk_anomaly_score[te_train_mask], 95)
qk_flag = (qk_anomaly_score > qk_thresh).astype(int)

print(f"  Quantum kernel threshold (95th pct, train): {qk_thresh:.4f}")
print(f"  Quantum kernel flags raised: {qk_flag.sum()} / {n_windows}")

# ════════════════════════════════════════════════════════════
# STEP 8: Crisis detection metrics — same logic as v9.6d, for comparison
# ════════════════════════════════════════════════════════════

def crisis_metrics_for_flags(flag_arr, dates_arr, crisis_periods,
                              max_lead_days=90):
    """Mirrors the classical HAD crisis-metric logic in v9.6d."""
    tp = 0; fn = 0; lead_days = []
    detected = []
    for (cs, ce) in crisis_periods:
        cs_dt = np.datetime64(cs)
        window_mask = (dates_arr >= cs_dt - np.timedelta64(max_lead_days, 'D')) & \
                      (dates_arr <= np.datetime64(ce))
        flagged_in_window = np.where(flag_arr & window_mask)[0]
        if len(flagged_in_window) > 0:
            first_flag_date = dates_arr[flagged_in_window[0]]
            lead = (cs_dt - first_flag_date) / np.timedelta64(1, 'D')
            tp += 1
            lead_days.append(float(lead))
            detected.append((cs, True, str(first_flag_date)[:10], float(lead)))
        else:
            fn += 1
            detected.append((cs, False, None, None))

    total_flags = int(flag_arr.sum())
    fp = max(0, total_flags - tp)   # flags not matched to any crisis window
    precision = tp / max(1, total_flags)
    recall    = tp / max(1, tp + fn)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mean_lead = float(np.mean(lead_days)) if lead_days else 0.0
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'Flags': total_flags,
        'Precision': precision, 'Recall': recall, 'F1': f1,
        'Mean_Lead_Days': mean_lead, 'detected': detected
    }

qk_metrics = crisis_metrics_for_flags(qk_flag, TE_dates, CRISIS_PERIODS)

print("\n  ┌────────────────────────────────────────────────────────┐")
print("  │  QUANTUM KERNEL vs CLASSICAL GMM — HAD COMPARISON      │")
print("  ├──────────────────┬──────────────────┬──────────────────┤")
print("  │ Metric           │ Quantum kernel    │ Classical GMM    │")
print("  │                  │ (this module)     │ (v9.6d, paste)   │")
print("  ├──────────────────┼──────────────────┼──────────────────┤")
print(f"  │ Flags raised     │ {qk_metrics['Flags']:>16}  │ {'470':>16}   │")
print(f"  │ TP (crises hit)  │ {qk_metrics['TP']:>16}  │ {'5':>16}   │")
print(f"  │ FN (missed)      │ {qk_metrics['FN']:>16}  │ {'0':>16}   │")
print(f"  │ Precision        │ {qk_metrics['Precision']:>16.4f}  │ {'0.0106':>16}  │")
print(f"  │ Recall           │ {qk_metrics['Recall']:>16.4f}  │ {'1.0000':>16}  │")
print(f"  │ F1               │ {qk_metrics['F1']:>16.4f}  │ {'0.0211':>16}  │")
print(f"  │ Mean Lead (days) │ {qk_metrics['Mean_Lead_Days']:>16.1f}  │ {'57.0':>16}  │")
print("  └──────────────────┴──────────────────┴──────────────────┘")

print("\n  Per-crisis breakdown (quantum kernel):")
print("  Crisis                  Detected    Flag Date  Lead Days")
print("  ---------------------------------------------------------")
for (name, det, fdate, lead) in qk_metrics['detected']:
    if det:
        print(f"  {name:<22}    ✓ YES   {fdate}   {lead:>5.0f}d")
    else:
        print(f"  {name:<22}    ✗ NO    {'—':>10}   {'—':>5}")

print("\n[Quantum Kernel HAD] Diagnostic complete.")
print("  NOTE: this is a robustness check, not a replacement for v9.6d.")
print("  Compare precision/recall above against the classical HAD-GMM")
print("  numbers from v9.6d to assess whether richer feature structure")
print("  (preserved via quantum entanglement) improves crisis separation")
print("  beyond what the classical 1D per-hyperedge GMM achieves.")