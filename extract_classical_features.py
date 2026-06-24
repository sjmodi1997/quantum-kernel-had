"""
Extract TE_mats, RVS, HDEG from real Indian financial data
Streamlined version focusing only on the pipeline steps needed for quantum kernel HAD
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
import pickle
import sys

print("=" * 70)
print("CLASSICAL PIPELINE - Extracting TE_mats, RVS, HDEG")
print("=" * 70)

# Step 1: Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('india_financial_dataset_prices_clean (1).csv', index_col=0)
df.index = pd.to_datetime(df.index)
print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")

# Step 2: Define parameters (from colleague's notebook)
N = df.shape[1]  # Number of variables
WINDOW_SIZE = 252  # ~1 year of trading days
STEP = 10
BINS = 3
n_windows = len(list(range(0, len(df) - WINDOW_SIZE, STEP)))

print(f"  N variables: {N}")
print(f"  Window size: {WINDOW_SIZE}")
print(f"  Step: {STEP}")
print(f"  Expected windows: {n_windows}")

# Step 3: Define Transfer Entropy computation (from colleague's code)
print("\n[2/4] Computing Transfer Entropy matrices...")

def compute_te_matrix(seg, bins=BINS, alpha=0.01):
    """
    Compute Transfer Entropy between all pairs of variables
    seg: data segment (T × N)
    returns: TE matrix (N × N) where TE[i,j] = TE(i->j)
    """
    T, N = seg.shape
    TE = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Discretize to bins
            x = pd.qcut(seg[:, i], q=bins, labels=False, duplicates='drop')
            y = pd.qcut(seg[:, j], q=bins, labels=False, duplicates='drop')

            # Handle NaN from qcut
            valid = ~(pd.isna(x) | pd.isna(y))
            if valid.sum() < 10:
                TE[i, j] = 0
                continue

            x, y = x[valid].astype(int), y[valid].astype(int)

            # TE(X->Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
            # Simplified: use current values
            try:
                py = np.bincount(y) / len(y)
                pxy = np.zeros((bins, bins))
                for xi in range(bins):
                    for yi in range(bins):
                        pxy[xi, yi] = ((x == xi) & (y == yi)).sum() / len(y)

                # H(Y)
                H_y = entropy(py + 1e-10)

                # H(Y|X) = sum_x P(x) * H(Y|X=x)
                H_y_given_x = 0
                for xi in range(bins):
                    mask = (x == xi)
                    if mask.sum() > 0:
                        py_given_x = pxy[xi, :] / (pxy[xi, :].sum() + 1e-10)
                        H_y_given_x += (mask.sum() / len(y)) * entropy(py_given_x + 1e-10)

                TE[i, j] = max(0, H_y - H_y_given_x)
            except:
                TE[i, j] = 0

    return TE

# Compute TE matrices for all windows
TE_mats = []
window_dates = []
print(f"  Computing {n_windows} TE matrices...")

for w_idx, start in enumerate(range(0, len(df) - WINDOW_SIZE, STEP)):
    if w_idx % 50 == 0:
        print(f"    Window {w_idx}/{n_windows}...", end='\r')

    end = start + WINDOW_SIZE
    seg = df.iloc[start:end, :].values

    # Normalize to [0,1] for TE computation
    seg_norm = (seg - seg.min(axis=0)) / (seg.max(axis=0) - seg.min(axis=0) + 1e-10)

    TE = compute_te_matrix(seg_norm)
    TE_mats.append(TE)
    window_dates.append(df.index[end-1])

print(f"  ✓ Computed {len(TE_mats)} TE matrices                  ")

# Step 4: Compute RVS (Risk Virality Score using PageRank)
print("\n[3/4] Computing Risk Virality Scores...")

def compute_rvs(W, alpha=0.85, tol=1e-8, max_iter=300):
    """PageRank on TE network"""
    N = W.shape[0]
    # Normalize each column
    W_norm = W / (W.sum(axis=0) + 1e-10)
    W_norm[np.isnan(W_norm)] = 1.0 / N

    r = np.ones(N) / N
    for _ in range(max_iter):
        r_new = (1 - alpha) / N + alpha * W_norm @ r
        if np.linalg.norm(r_new - r) < tol:
            break
        r = r_new

    return r / r.sum()

RVS = np.zeros((n_windows, N))
for w_idx, TE in enumerate(TE_mats):
    RVS[w_idx, :] = compute_rvs(TE)
    if w_idx % 50 == 0:
        print(f"    Window {w_idx}/{n_windows}...", end='\r')

print(f"  ✓ Computed RVS for {n_windows} windows              ")

# Step 5: Build hyperedges and compute degrees
print("\n[4/4] Building hyperedges...")

def build_hyperedges(TE_mat, threshold=0.1):
    """
    Build hyperedges from TE matrix
    Hyperedge = group of nodes with high mutual TE
    """
    N = TE_mat.shape[0]
    # Create adjacency matrix from TE (threshold)
    adj = (TE_mat > threshold).astype(int)

    hyperedges = []
    visited = set()

    for i in range(N):
        if i in visited:
            continue
        # Find all nodes connected to i
        group = set([i])
        to_visit = list(np.where(adj[i] > 0)[0])
        while to_visit:
            node = to_visit.pop(0)
            if node not in group:
                group.add(node)
                visited.add(node)
                to_visit.extend(np.where(adj[node] > 0)[0])

        if len(group) >= 2:
            hyperedges.append(list(group))
        visited.add(i)

    # If no hyperedges found, use TE-based grouping
    if not hyperedges:
        te_threshold = np.percentile(TE_mat[TE_mat > 0], 75)
        for i in range(N):
            edge = [i] + list(np.where(TE_mat[i] > te_threshold)[0])
            if len(set(edge)) >= 2:
                hyperedges.append(list(set(edge)))

    return hyperedges if hyperedges else [[i for i in range(N)]]

# Compute hyperedges and HDEG for all windows
all_he = []
HDEG = np.zeros((n_windows, N))

for w_idx, TE_mat in enumerate(TE_mats):
    he = build_hyperedges(TE_mat)
    all_he.append(he)

    # Compute hyperedge degree (how many hyperedges each node belongs to)
    for node in range(N):
        HDEG[w_idx, node] = sum(1 for edge in he if node in edge)

    if w_idx % 50 == 0:
        print(f"    Window {w_idx}/{n_windows}...", end='\r')

print(f"  ✓ Built hyperedges for {n_windows} windows          ")

# Step 6: Save results
print("\n[SAVING] Writing results to disk...")

np.save('TE_mats.npy', np.array(TE_mats, dtype=object))
np.save('RVS.npy', RVS)
np.save('HDEG.npy', HDEG)

with open('window_dates.pkl', 'wb') as f:
    pickle.dump(window_dates, f)
with open('all_hyperedges.pkl', 'wb') as f:
    pickle.dump(all_he, f)

print(f"  ✓ TE_mats.npy: shape {len(TE_mats)} (list of matrices)")
print(f"  ✓ RVS.npy: shape {RVS.shape}")
print(f"  ✓ HDEG.npy: shape {HDEG.shape}")
print(f"  ✓ window_dates.pkl: {len(window_dates)} dates")
print(f"  ✓ all_hyperedges.pkl: {len(all_he)} hyperedge lists")

print("\n" + "=" * 70)
print("✓ CLASSICAL PIPELINE COMPLETE")
print("=" * 70)
print(f"\nData ready for quantum kernel HAD:")
print(f"  - {n_windows} analysis windows")
print(f"  - {N} financial variables")
print(f"  - Transfer Entropy matrices computed")
print(f"  - Risk Virality Scores computed")
print(f"  - Hyperedge degrees computed")
