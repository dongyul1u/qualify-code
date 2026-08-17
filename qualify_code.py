import numpy as np
import pickle
from typing import Union

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import traceback
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import friedmanchisquare, rankdata, shapiro, ttest_ind, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, load_diabetes
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pennylane as qml
import math
import warnings
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / RUN_ID
FIG_DIR = RUN_DIR / "figures"
LOG_DIR = RUN_DIR / "logs"
TABLE_DIR = RUN_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "_shared_cache"
BEST_CFG_PATH = CACHE_DIR / "best_cfgs.pkl"

for d in [RESULTS_DIR, RUN_DIR, FIG_DIR, LOG_DIR, TABLE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_figure(fig, fig_name: str, dpi: int = 200):
    outpath = FIG_DIR / fig_name
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved figure] {outpath}")


def save_table(df, file_name: str):
    outpath = TABLE_DIR / file_name
    df.to_csv(outpath, index=False)
    print(f"[Saved table] {outpath}")


def run_wilcoxon_test_from_saved_results(
    file_name: str = "raw_seed_results.csv",
    method_a: str = "optimized_hybrid",
    method_b: str = "quantum_entropy",
    group_cols=("dataset", "model"),
    score_col: str = "final_accuracy",
    alternative: str = "two-sided",
):
    csv_path = TABLE_DIR / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Result file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"method", score_col, *group_cols}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {sorted(missing_cols)}"
        )

    df = df[df["method"].isin([method_a, method_b])].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for methods '{method_a}' and '{method_b}' in {csv_path.name}"
        )

    grouped = (
        df.groupby(list(group_cols) + ["method"], as_index=False)[score_col]
        .mean()
    )
    paired = (
        grouped.pivot(index=list(group_cols), columns="method", values=score_col)
        .reset_index()
    )

    if method_a not in paired.columns or method_b not in paired.columns:
        raise ValueError(
            f"Unable to build paired samples for '{method_a}' vs '{method_b}'"
        )

    paired = paired.dropna(subset=[method_a, method_b]).copy()
    if paired.empty:
        raise ValueError(
            f"No valid paired rows remain for '{method_a}' vs '{method_b}'"
        )

    x = paired[method_a].to_numpy(dtype=float)
    y = paired[method_b].to_numpy(dtype=float)
    diff = x - y

    paired["difference"] = diff
    paired["winner"] = np.where(
        diff > 1e-12,
        method_a,
        np.where(diff < -1e-12, method_b, "tie"),
    )

    nonzero_diff = diff[np.abs(diff) > 1e-12]
    if len(nonzero_diff) == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic, p_value = stats.wilcoxon(
            x,
            y,
            alternative=alternative,
            zero_method="wilcox",
        )

    wins_a = int(np.sum(diff > 1e-12))
    wins_b = int(np.sum(diff < -1e-12))
    ties = int(np.sum(np.abs(diff) <= 1e-12))
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    significant = bool(p_value < 0.05)

    def _safe_name(text: str) -> str:
        return (
            str(text)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("+", "plus")
            .replace("-", "_")
        )

    prefix = f"wilcoxon_{_safe_name(method_a)}_vs_{_safe_name(method_b)}"

    summary_df = pd.DataFrame([{
        "file_name": file_name,
        "method_a": method_a,
        "method_b": method_b,
        "group_cols": "|".join(group_cols),
        "score_col": score_col,
        "alternative": alternative,
        "n_pairs": len(paired),
        "n_nonzero_pairs": len(nonzero_diff),
        "method_a_mean": float(np.mean(x)),
        "method_b_mean": float(np.mean(y)),
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "wins_method_a": wins_a,
        "wins_method_b": wins_b,
        "ties": ties,
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0_05": significant,
    }])

    save_table(paired, f"{prefix}_pairs.csv")
    save_table(summary_df, f"{prefix}_summary.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = max(0.01, 0.05 * (hi - lo + 1e-12))

    ax1.scatter(y, x, alpha=0.8, s=55, color="steelblue", edgecolor="white")
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1.2)
    ax1.set_xlim(lo - pad, hi + pad)
    ax1.set_ylim(lo - pad, hi + pad)
    ax1.set_xlabel(f"{method_b} score")
    ax1.set_ylabel(f"{method_a} score")
    ax1.set_title("Paired Comparison", fontweight="bold")
    ax1.grid(True, alpha=0.25)

    bins = min(12, max(5, int(np.sqrt(len(diff)))))
    ax2.hist(diff, bins=bins, color="darkorange", alpha=0.8, edgecolor="white")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(f"Difference ({method_a} - {method_b})")
    ax2.set_ylabel("Count")
    ax2.set_title("Difference Distribution", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.text(
        0.98,
        0.98,
        f"n = {len(paired)}\n"
        f"W = {statistic:.3f}\n"
        f"p = {p_value:.4g}\n"
        f"mean diff = {mean_diff:+.4f}\n"
        f"median diff = {median_diff:+.4f}\n"
        f"wins/ties/losses = {wins_a}/{ties}/{wins_b}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75, pad=0.35),
    )

    fig.suptitle(
        f"Wilcoxon Signed-Rank Test: {method_a} vs {method_b}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, f"{prefix}.png")

    print("\nWilcoxon signed-rank test summary")
    print(f"  file: {csv_path}")
    print(f"  paired by: {group_cols}")
    print(f"  methods: {method_a} vs {method_b}")
    print(f"  n_pairs = {len(paired)}")
    print(f"  W = {statistic:.4f}, p = {p_value:.6f}")
    print(f"  mean difference = {mean_diff:+.4f}")
    print(f"  wins/ties/losses = {wins_a}/{ties}/{wins_b}")

    return {
        "paired_df": paired,
        "summary_df": summary_df,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": significant,
    }

# --- configuration ---
class HyperparamConfig:
    def __init__(self,
                 w_qd_q: float = 0.7,
                 w_qd_d: float = 0.3,
                 w_ud_u: float = 0.5,
                 w_ud_d: float = 0.5,
                 w_ed_e: float = 0.7,
                 w_ed_d: float = 0.3,
                 w_hd_q: float = 0.4,
                 w_hd_c: float = 0.3,
                 w_hd_d: float = 0.3,
                 early_weight: float = 0.8,
                 late_weight: float = 0.3,
                 complexity_thresh_high: int = 40,
                 complexity_thresh_mid: int = 15,
                 complexity_boost_high: float = 1.3,
                 complexity_boost_mid: float = 1.1,
                 quantum_dom_threshold: float = 0.7,
                 balanced_threshold: float = 0.4,
                 decay_high: float = 0.2,
                 decay_mid: float = 0.25,
                 decay_low: float = 0.3,
                 wc_time_scale: float = 0.3,
                 wd_time_scale: float = 0.2,
                 ):
        self.w_qd_q = w_qd_q
        self.w_qd_d = w_qd_d
        self.w_ud_u = w_ud_u
        self.w_ud_d = w_ud_d
        self.w_ed_e = w_ed_e
        self.w_ed_d = w_ed_d
        self.w_hd_q = w_hd_q
        self.w_hd_c = w_hd_c
        self.w_hd_d = w_hd_d
        self.early_weight = early_weight
        self.late_weight = late_weight
        self.complexity_thresh_high = complexity_thresh_high
        self.complexity_thresh_mid = complexity_thresh_mid
        self.complexity_boost_high = complexity_boost_high
        self.complexity_boost_mid = complexity_boost_mid
        self.quantum_dom_threshold = quantum_dom_threshold
        self.balanced_threshold = balanced_threshold
        self.decay_high = decay_high
        self.decay_mid = decay_mid
        self.decay_low = decay_low
        self.wc_time_scale = wc_time_scale
        self.wd_time_scale = wd_time_scale

    def __repr__(self):
        return (
            f"HyperparamConfig("
            f"w_qd=({self.w_qd_q:.2f},{self.w_qd_d:.2f}), "
            f"w_ud=({self.w_ud_u:.2f},{self.w_ud_d:.2f}), "
            f"w_ed=({self.w_ed_e:.2f},{self.w_ed_d:.2f}), "
            f"w_hd=({self.w_hd_q:.2f},{self.w_hd_c:.2f},{self.w_hd_d:.2f}), "
            f"sched=({self.early_weight:.2f}->{self.late_weight:.2f}), "
            f"decay=({self.decay_high:.2f},{self.decay_mid:.2f},{self.decay_low:.2f}))"
        )

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# --- quantum sampling ---
class EnhancedQuantumSamplingCircuit:
    def __init__(self, n_features, n_qubits=None, n_layers=8, seed=123):
        self.n_features = n_features

        if n_qubits is None:
            required_qubits = math.ceil(math.log2(max(n_features, 2)))
            self.n_qubits = min(max(3, required_qubits + 1), 10)
        else:
            self.n_qubits = min(n_qubits, 10)

        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)

        print(f"    Enhanced quantum circuit: features={n_features}, qubits={self.n_qubits}, layers={n_layers}")

        self.device = qml.device("default.qubit", wires=self.n_qubits)

        self.weights = 0.1 * self.rng.standard_normal(size=(n_layers, self.n_qubits, 3))

        @qml.qnode(self.device, interface=None)
        def enhanced_quantum_circuit(x, weights):
            self.enhanced_amplitude_embedding(x, wires=range(self.n_qubits))

            for layer in range(n_layers):
                for qubit in range(self.n_qubits):
                    qml.Rot(*weights[layer, qubit], wires=qubit)

                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                    for qubit in range(self.n_qubits - 2):
                        qml.CNOT(wires=[qubit, qubit + 2])

                if layer % 2 == 0 and self.n_qubits > 3:
                    for qubit in range(0, self.n_qubits - 2, 2):
                        qml.CZ(wires=[qubit, qubit + 2])

            return qml.probs(wires=range(self.n_qubits))

        self.quantum_circuit = enhanced_quantum_circuit

    def enhanced_amplitude_embedding(self, x, wires):
        x = np.asarray(x, dtype=float)
        if np.allclose(x, 0):
            x = np.ones_like(x)

        feature_importance = np.abs(x) / (np.sum(np.abs(x)) + 1e-12)
        weighted_x = x * (1 + 0.5 * feature_importance)

        x_norm = weighted_x / np.linalg.norm(weighted_x)

        target_dim = 2 ** len(wires)

        if len(x_norm) < target_dim:
            x_padded = np.zeros(target_dim)
            x_padded[:len(x_norm)] = x_norm
            x_norm = x_padded
        elif len(x_norm) > target_dim:
            pca = PCA(n_components=target_dim)
            weighted_data = np.diag(np.sqrt(feature_importance)) @ x_norm.reshape(1, -1)
            x_norm = pca.fit_transform(weighted_data.T).flatten()
            if np.allclose(x_norm, 0):
                x_norm = np.ones_like(x_norm)
            x_norm = x_norm / np.linalg.norm(x_norm)

        qml.AmplitudeEmbedding(x_norm, wires=wires, normalize=False)

    def compute_quantum_entropy(self, x):
        try:
            probs = self.quantum_circuit(x, self.weights)
            probs = np.clip(probs, 1e-12, 1.0)
            entropy = -np.sum(probs * np.log(probs))
            return float(entropy)
        except Exception as e:
            print(f"    Quantum entropy error: {e}")
            return 0.5

    def compute_quantum_uncertainty(self, x):
        try:
            probs = self.quantum_circuit(x, self.weights)
            max_prob = np.max(probs)
            shannon_entropy = -np.sum(probs * np.log(probs + 1e-12))
            variance = np.var(probs)

            n = len(probs)
            max_variance = (n - 1) / n**2
            normalized_variance = variance / max_variance if max_variance > 0 else 0.0

            uncertainty = (
                0.5 * (1.0 - max_prob) +
                0.3 * (shannon_entropy / np.log(n)) +
                0.2 * normalized_variance
            )
            return float(uncertainty)
        except Exception as e:
            print(f"    Quantum uncertainty error: {e}")
            return 0.5

class EnsembleQuantumSampler:    
    def __init__(self, n_features, n_models=3, seed=123):
        self.models = []
        self.n_features = n_features
        
        for i in range(n_models):
            n_qubits = min(10, max(3, math.ceil(math.log2(max(n_features, 2))) + i % 3))
            n_layers = 6 + i * 2
            
            print(f"    Ensemble model {i+1}: {n_qubits} qubits, {n_layers} layers")
            
            sampler = EnhancedQuantumSamplingCircuit(
                n_features=n_features,
                n_qubits=n_qubits,
                n_layers=n_layers,
                seed=seed + i * 100
            )
            self.models.append(sampler)
    
    def ensemble_quantum_entropy(self, x):
        entropies = []
        for model in self.models:
            try:
                entropy = model.compute_quantum_entropy(x)
                entropies.append(entropy)
            except Exception as e:
                print(f"    Ensemble entropy computation error: {e}")
                entropies.append(0.5)
        
        return np.mean(entropies) if entropies else 0.5
    
    def ensemble_quantum_uncertainty(self, x):
        uncertainties = []
        for model in self.models:
            try:
                uncertainty = model.compute_quantum_uncertainty(x)
                uncertainties.append(uncertainty)
            except Exception as e:
                print(f"    Ensemble model uncertainty computation error: {e}")
                uncertainties.append(0.5)
        
        return np.mean(uncertainties) if uncertainties else 0.5

# --- utility scoring ---
def binary_entropy(p, eps=1e-12):
    p = float(np.clip(p, eps, 1 - eps))
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def structure_diversity_score(x, labeled_X, metric='euclidean'):
    if labeled_X is None or len(labeled_X) == 0:
        return 0.0
    x = np.asarray(x)
    L = np.asarray(labeled_X)
    
    if metric == 'euclidean':
        dists = np.sqrt(((L - x) ** 2).sum(axis=1))
    elif metric == 'cosine':
        norm_x = np.linalg.norm(x)
        norm_L = np.linalg.norm(L, axis=1)
        dots = np.dot(L, x)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_sim = dots / (norm_L * norm_x)
            cos_sim = np.clip(cos_sim, -1, 1)
        dists = 1 - cos_sim
    elif metric == 'manhattan':
        dists = np.abs(L - x).sum(axis=1)
    else:
        dists = np.sqrt(((L - x) ** 2).sum(axis=1))
    
    return float(np.min(dists))



def farthest_first_select(X, candidates, already_selected, k, metric="euclidean"):
    candidates = np.asarray(candidates, dtype=int)
    already_selected = np.asarray(already_selected, dtype=int)

    if k <= 0 or len(candidates) == 0:
        return []

    if metric != "euclidean":
        raise ValueError(f"Unsupported metric: {metric}")

    # Initial minimum distance from every candidate
    # to the EXISTING labeled set
    if len(already_selected) > 0:
        L = X[already_selected]

        min_distances = np.array([
            np.min(np.linalg.norm(L - X[i], axis=1))
            for i in candidates
        ])
    else:
        min_distances = np.full(len(candidates), np.inf)

    selected = []
    available = np.ones(len(candidates), dtype=bool)

    for _ in range(min(k, len(candidates))):

        available_idx = np.where(available)[0]

        if len(available_idx) == 0:
            break

        # Select the point farthest from the CURRENT represented set
        pos = available_idx[
            np.argmax(min_distances[available_idx])
        ]

        chosen = candidates[pos]
        selected.append(chosen)
        available[pos] = False

        # The new selected point is now part of the represented set.
        # Update each remaining candidate's distance to L ∪ selected.
        for j in np.where(available)[0]:

            d_new = np.linalg.norm(
                X[candidates[j]] - X[chosen]
            )

            min_distances[j] = min(
                min_distances[j],
                d_new
            )

    return selected


def normalize_for_scores(scores, method="minmax", constant_value=0.5):
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        return scores

    if method == "minmax":
        s_min = np.min(scores)
        s_max = np.max(scores)
        if np.isclose(s_max, s_min):
            return np.full_like(scores, fill_value=constant_value, dtype=float)
        return (scores - s_min) / (s_max - s_min)

    elif method == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if np.isclose(std, 0.0):
            return np.full_like(scores, fill_value=constant_value, dtype=float)
        return (scores - mean) / (std + 1e-12)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def select_badge_batch_classical(model, X_pool, k, subset=None, rng=None):
    n = len(X_pool)
    idx_all = np.arange(n)
    
    if subset is not None and subset < n:
        if rng is None:
            rng = np.random.default_rng()
        idx_all = rng.choice(idx_all, size=subset, replace=False)

    embeds = []
    for i in idx_all:
        x_np = np.asarray(X_pool[i], dtype=float).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            try:
                p_val = model.predict_proba(x_np)[0, 1]
                gradient = (0.5 - p_val) * x_np.ravel()
                if 0.3 < p_val < 0.7:
                    gradient *= (1 + 2 * p_val * (1 - p_val))
            except Exception:
                gradient = np.zeros_like(x_np.ravel())
        else:
            gradient = np.zeros_like(x_np.ravel())
        embeds.append(gradient)

    if not embeds:
        return []

    embeds = np.stack(embeds, axis=0)
    selected = []

    scores = np.linalg.norm(embeds, axis=1) + 0.5 * np.var(embeds, axis=1)
    selected.append(int(np.argmax(scores)))
    selected_set = set(selected)

    while len(selected) < min(k, len(idx_all)):
        d2 = []
        for i in range(len(idx_all)):
            if i in selected_set:
                d2.append(0.0)
                continue

            mind = np.inf
            for j in selected:
                dist_e = np.linalg.norm(embeds[i] - embeds[j])
                cos_sim = np.dot(embeds[i], embeds[j]) / (
                    np.linalg.norm(embeds[i]) * np.linalg.norm(embeds[j]) + 1e-12
                )
                dist_c = 1 - cos_sim
                mind = min(mind, 0.7 * dist_e + 0.3 * dist_c)
            d2.append(mind)

        nxt = int(np.argmax(d2))
        if nxt in selected_set:
            break
        selected.append(nxt)
        selected_set.add(nxt)

    return list(idx_all[selected])


def adaptive_quantum_weight(round_num, total_rounds, dataset_complexity, strategy='progressive', cfg=None):
    if cfg is None:
        cfg = HyperparamConfig()
    if strategy == 'progressive':
        progress = round_num / max(1, total_rounds - 1)
        base_weight = cfg.early_weight * (1 - progress) + cfg.late_weight * progress

        if dataset_complexity > cfg.complexity_thresh_high:
            base_weight = min(0.9, base_weight * cfg.complexity_boost_high)
        elif dataset_complexity > cfg.complexity_thresh_mid:
            base_weight = base_weight * cfg.complexity_boost_mid
    elif strategy == 'dataset_aware':
        if dataset_complexity > 50:
            base_weight = 0.7
        elif dataset_complexity > 30:
            base_weight = 0.6
        elif dataset_complexity > 15:
            base_weight = 0.5
        else:
            base_weight = 0.4
    else:
        base_weight = 0.5

    return max(0.2, min(0.9, base_weight))

def select_optimized_hybrid_strategy(
    quantum_sampler,
    classical_model,
    X,
    unlabeled_idx,
    labeled_idx,
    k,
    round_num,
    total_rounds,
    dataset_complexity,
    rng,
    cfg=None
):
    
    if cfg is None:
        cfg = HyperparamConfig()
    if len(unlabeled_idx) == 0:
        return []

    quantum_weight = adaptive_quantum_weight(
        round_num, total_rounds, dataset_complexity, "progressive", cfg=cfg
    )
    L_aug = X[labeled_idx] if len(labeled_idx) > 0 else None

    if quantum_weight > cfg.quantum_dom_threshold: 
        print(f"    Round {round_num}: Quantum-dominant (weight={quantum_weight:.2f})")
        M = min(4 * k, len(unlabeled_idx))

        classical_scores = []
        for u in unlabeled_idx:
            if hasattr(classical_model, "predict_proba"):
                try:
                    p1 = classical_model.predict_proba(X[u].reshape(1, -1))[0, 1]
                except Exception:
                    p1 = 0.5
            else:
                p1 = 0.5
            classical_scores.append(binary_entropy(p1))

        classical_scores = np.array(classical_scores)
        top_pos = np.argsort(classical_scores)[-M:]
        cand_idx = unlabeled_idx[top_pos]

        cand_list, q_list, d_list = [], [], []
        for u in cand_idx:
            if hasattr(quantum_sampler, "ensemble_quantum_entropy"):
                q = quantum_sampler.ensemble_quantum_entropy(X[u])
            else:
                q = quantum_sampler.compute_quantum_entropy(X[u])

            d = structure_diversity_score(X[u], L_aug, metric="euclidean")

            cand_list.append(int(u))
            q_list.append(q)
            d_list.append(d)

        q_arr = normalize_for_scores(q_list, method="minmax")
        d_arr = normalize_for_scores(d_list, method="minmax")
        scores = quantum_weight * q_arr + (1 - quantum_weight) * d_arr

        top_k_indices = np.argsort(scores)[-k:]
        selected = [cand_list[i] for i in top_k_indices]

    elif quantum_weight > cfg.balanced_threshold:
        print(f"    Round {round_num}: Balanced (weight={quantum_weight:.2f})")
        M = min(4 * k, len(unlabeled_idx))
        cand_idx = (
            rng.choice(unlabeled_idx, size=M, replace=False)
            if len(unlabeled_idx) > M else unlabeled_idx
        )

        cand_list, q_list, c_list, d_list = [], [], [], []
        for u in cand_idx:
            if hasattr(quantum_sampler, "ensemble_quantum_entropy"):
                q = quantum_sampler.ensemble_quantum_entropy(X[u])
            else:
                q = quantum_sampler.compute_quantum_entropy(X[u])

            if hasattr(classical_model, "predict_proba"):
                try:
                    p1 = classical_model.predict_proba(X[u].reshape(1, -1))[0, 1]
                    c = binary_entropy(p1)
                except Exception:
                    c = 0.5
            else:
                c = 0.5

            d = structure_diversity_score(X[u], L_aug, metric="euclidean")

            cand_list.append(int(u))
            q_list.append(q)
            c_list.append(c)
            d_list.append(d)

        q_arr = normalize_for_scores(q_list, method="minmax")
        c_arr = normalize_for_scores(c_list, method="minmax")
        d_arr = normalize_for_scores(d_list, method="minmax")

        scores = (
            quantum_weight * q_arr
            + 0.5 * (1 - quantum_weight) * c_arr
            + 0.5 * (1 - quantum_weight) * d_arr
        )

        top_k_indices = np.argsort(scores)[-k:]
        selected = [cand_list[i] for i in top_k_indices]

    else:
        print(f"    Round {round_num}: Classical-dominant (weight={quantum_weight:.2f})")
        if hasattr(classical_model, "predict_proba"):
            probs = classical_model.predict_proba(X[unlabeled_idx])[:, 1]
            uncertainties = np.abs(probs - 0.5)
            selected_indices = np.argsort(uncertainties)[:min(k, len(unlabeled_idx))]
            selected = [unlabeled_idx[i] for i in selected_indices]
        else:
            selected = list(
                rng.choice(unlabeled_idx, size=min(k, len(unlabeled_idx)), replace=False)
            )

    return selected

def select_for_high_dimensional_data(quantum_sampler, model, X, unlabeled_idx,
                labeled_idx, k, rng, pca_components=20, cfg=None):
    if cfg is None:
        cfg = HyperparamConfig()
    if len(unlabeled_idx) == 0:
        return []
    
    n_components = min(pca_components, X.shape[1], len(unlabeled_idx))
    if n_components < 2:
        n_components = min(2, X.shape[1])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    L_aug = X[labeled_idx] if len(labeled_idx) > 0 else None
    
    candidate_size = min(150, len(unlabeled_idx))
    if len(unlabeled_idx) > candidate_size:
        candidate_indices = rng.choice(
            unlabeled_idx,
            size=candidate_size,
            replace=False
        )
    else:
        candidate_indices = unlabeled_idx
    
    scores = []
    cand_list, q_list, c_list, d_list = [], [], [], []
    for u in candidate_indices:
        if hasattr(quantum_sampler, 'ensemble_quantum_entropy'):
            q = quantum_sampler.ensemble_quantum_entropy(X_pca[u])
        else:
            q = quantum_sampler.compute_quantum_entropy(X_pca[u])
        
        d = structure_diversity_score(X[u], L_aug, metric='euclidean')
        
        if hasattr(model, 'predict_proba'):
            try:
                p1 = model.predict_proba(X[u].reshape(1, -1))[0, 1]
                c = binary_entropy(p1)
            except Exception:
                c = 0.5
        else:
            c = 0.5

        cand_list.append(int(u))
        q_list.append(q)
        c_list.append(c)
        d_list.append(d)

    q_arr = normalize_for_scores(q_list, method="minmax")
    c_arr = normalize_for_scores(c_list, method="minmax")
    d_arr = normalize_for_scores(d_list, method="minmax")

    scores = cfg.w_hd_q * q_arr + cfg.w_hd_c * c_arr + cfg.w_hd_d * d_arr


    top_k_indices = np.argsort(scores)[-k:]
   
    return [cand_list[i] for i in top_k_indices]


# --- classical baselines ---
class TraditionalModels:
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
        }
        self.trained_models = {}
    
    def fit(self, model_name, X, y):
        try:
            model = self.models[model_name]
            model.fit(X, y)
            self.trained_models[model_name] = model
            return model
        except Exception as e:
            print(f"    model training failed {model_name}: {e}")
            return None
    
    def predict_proba(self, model_name, X):
        if model_name not in self.trained_models:
            return np.zeros(len(X))
            
        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            try:
                return model.predict_proba(X)[:, 1]
            except Exception:
                return np.ones(len(X))*0.5
        else:
            return model.predict(X)


def split_and_scale_for_active_learning(X, y, test_idx, pool_idx):
    X_pool_raw = X[pool_idx]
    y_pool = y[pool_idx]

    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool_raw)
    X_test = scaler.transform(X_test_raw)

    return X_pool, y_pool, X_test, y_test, scaler


# --- active learning loop ---
def build_quantum_candidate_pool(
    X_pool,
    unlabeled_idx,
    model,
    k_actual,
    rng,
    min_candidates=40,
    multiplier=8,
):
    if len(unlabeled_idx) == 0:
        return np.array([], dtype=int)

    candidate_size = min(len(unlabeled_idx), max(min_candidates, multiplier * k_actual))
    if candidate_size >= len(unlabeled_idx):
        return unlabeled_idx.copy()

    if model is None or not hasattr(model, "predict_proba"):
        return np.array(
            rng.choice(unlabeled_idx, size=candidate_size, replace=False),
            dtype=int,
        )

    classical_scores = []
    for u in unlabeled_idx:
        try:
            p1 = model.predict_proba(X_pool[u].reshape(1, -1))[0, 1]
            score = binary_entropy(p1)
        except Exception:
            score = 0.5
        classical_scores.append(score)

    classical_scores = np.asarray(classical_scores, dtype=float)
    top_pos = np.argsort(classical_scores)[-candidate_size:]
    return np.asarray(unlabeled_idx[top_pos], dtype=int)


def auto_weights_by_dataset_complexity(X):
    n, d = X.shape

    if d >= 50:
        w_q, w_c, w_d = 0.6, 0.2, 0.2
    elif d >= 25:
        w_q, w_c, w_d = 0.5, 0.3, 0.2
    elif d >= 10:
        w_q, w_c, w_d = 0.4, 0.4, 0.2
    else:
        w_q, w_c, w_d = 0.3, 0.5, 0.2

    s = w_q + w_c + w_d
    return w_q / s, w_c / s, w_d / s

def auto_weights_by_round(r, R_max, base_wq, base_wc, base_wd, dataset_complexity, cfg=None):
    if cfg is None:
        cfg = HyperparamConfig()
    t = r / max(1, R_max)

    if dataset_complexity > 40:
        decay_factor = cfg.decay_high
    elif dataset_complexity > 15:
        decay_factor = cfg.decay_mid
    else:
        decay_factor = cfg.decay_low

    wq = base_wq * (1.0 - decay_factor * t)
    wc = base_wc * ((1 - cfg.wc_time_scale) + cfg.wc_time_scale * t)
    wd = base_wd * ((1 - cfg.wd_time_scale) + cfg.wd_time_scale * t)

    s = wq + wc + wd
    return wq / s, wc / s, wd / s


def run_active_learning_experiment(X, y, model_name, strategy="random",
                                  init_per_class=2, rounds=15, query_batch=5, seed=42,
                                  use_enhanced_quantum=True, cfg=None):
    rng = np.random.default_rng(seed)
    X_full = X
    y_full = y
    n = len(X_full)

    print(f"  raw dataset complexity: {X_full.shape[1]}, strategy: {strategy}")

    n_test = max(20, int(0.2 * n))
    indices = np.arange(n)
    rng.shuffle(indices)
    test_idx = indices[:n_test]
    pool_idx = indices[n_test:]

    X_pool, y_pool, X_test, y_test, scaler = split_and_scale_for_active_learning(
        X_full, y_full, test_idx, pool_idx
    )

    dataset_complexity = X_pool.shape[1]

    print(f"  pool complexity after split: {dataset_complexity}, strategy: {strategy}")

    labeled_idx = []
    for c in np.unique(y_pool):
        idx_c = np.where(y_pool == c)[0]
        if len(idx_c) >= init_per_class:
            selected = rng.choice(idx_c, size=init_per_class, replace=False)
            labeled_idx.extend(selected.tolist())
        elif len(idx_c) > 0:
            labeled_idx.extend(idx_c.tolist())

    labeled_idx = np.array(sorted(set(labeled_idx)), dtype=int)
    unlabeled_idx = np.setdiff1d(np.arange(len(X_pool)), labeled_idx, assume_unique=False)

    if cfg is None:
        cfg = HyperparamConfig()
    base_wq, base_wc, base_wd = auto_weights_by_dataset_complexity(X_pool)

    traditional_models = TraditionalModels()
    quantum_sampler = None

    if strategy in [
        "quantum_entropy", "quantum_uncertainty", "pure_quantum",
        "hybrid_uncertainty_quantum", "optimized_hybrid", "high_dim_quantum"
    ]:
        if use_enhanced_quantum:
            if dataset_complexity > 30 and strategy != "pure_quantum":
                quantum_sampler = EnsembleQuantumSampler(
                    n_features=X_pool.shape[1],
                    n_models=3,
                    seed=seed
                )
            else:
                n_qubits = min(10, max(3, math.ceil(math.log2(max(X_pool.shape[1], 2))) + 1))
                quantum_sampler = EnhancedQuantumSamplingCircuit(
                    n_features=X_pool.shape[1],
                    n_qubits=n_qubits,
                    n_layers=8,
                    seed=seed
                )
        else:
            n_qubits = min(8, math.ceil(math.log2(max(X_pool.shape[1], 2))))
            quantum_sampler = EnhancedQuantumSamplingCircuit(
                n_features=X_pool.shape[1],
                n_qubits=n_qubits,
                n_layers=4,
                seed=seed
            )

    acc_hist, label_counts = [], []

    for r in range(rounds + 1):
        wq_r, wc_r, wd_r = auto_weights_by_round(
            r, rounds, base_wq, base_wc, base_wd, dataset_complexity, cfg=cfg
        )

        if len(labeled_idx) > 0:
            X_labeled, y_labeled = X_pool[labeled_idx], y_pool[labeled_idx]
            model = traditional_models.fit(model_name, X_labeled, y_labeled)

            if model is not None:
                y_pred_proba = traditional_models.predict_proba(model_name, X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
                acc = (y_pred == y_test).mean()
            else:
                acc = 0.5
        else:
            acc = 0.5
            model = None

        acc_hist.append(float(acc))
        label_counts.append(int(len(labeled_idx)))

        if r == rounds or len(unlabeled_idx) == 0:
            break

        batch = []
        k_actual = min(query_batch, len(unlabeled_idx))
        shared_quantum_cand_idx = None
        if strategy in {"quantum_entropy", "quantum_uncertainty", "pure_quantum"}:
            shared_quantum_cand_idx = build_quantum_candidate_pool(
                X_pool,
                unlabeled_idx,
                model,
                k_actual,
                rng,
            )

        if strategy == "random":
            batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "uncertainty":
            if model is not None:
                probs = traditional_models.predict_proba(model_name, X_pool[unlabeled_idx])
                uncertainties = np.abs(probs - 0.5)
                selected_indices = np.argsort(uncertainties)[:k_actual]
                batch = unlabeled_idx[selected_indices].tolist()
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "coreset":
            batch = farthest_first_select(
                unlabeled_idx.tolist(), X_pool, labeled_idx.tolist(), k_actual, metric='euclidean'
            )

        elif strategy == "badge":
            if model is not None:
                poolX = X_pool[unlabeled_idx]
                sel_local = select_badge_batch_classical(
                    model, poolX, k=k_actual, subset=min(150, len(poolX))
                )
                batch = [unlabeled_idx[i] for i in sel_local]
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "clusteraware":
            if model is not None:
                L_aug = X_pool[labeled_idx]
                probs = traditional_models.predict_proba(model_name, X_pool[unlabeled_idx])
                uncertainty_scores = 0.5 - np.abs(probs - 0.5)
                diversity_scores = np.array([
                    structure_diversity_score(X_pool[u], L_aug) for u in unlabeled_idx
                ])

                norm_uncertainty = normalize_for_scores(uncertainty_scores, method="minmax")
                norm_diversity = normalize_for_scores(diversity_scores, method="minmax")
                combined_scores = cfg.w_ud_u * norm_uncertainty + cfg.w_ud_d * norm_diversity
                selected_indices_relative = np.argsort(combined_scores)[-k_actual:]
                batch = unlabeled_idx[selected_indices_relative].tolist()
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "entropy_diversity":
            if model is not None:
                L_aug = X_pool[labeled_idx]
                M = min(150, len(unlabeled_idx))
                cand_idx = (
                    rng.choice(unlabeled_idx, size=M, replace=False)
                    if len(unlabeled_idx) > M else unlabeled_idx
                )

                entropy_scores, diversity_scores, cand_list = [], [], []
                for u in cand_idx:
                    if hasattr(model, 'predict_proba'):
                        try:
                            p1 = model.predict_proba(X_pool[u].reshape(1, -1))[0, 1]
                        except Exception:
                            p1 = 0.5
                    else:
                        p1 = 0.5

                    entropy_scores.append(binary_entropy(p1))
                    diversity_scores.append(structure_diversity_score(X_pool[u], L_aug))
                    cand_list.append(int(u))

                norm_entropy = normalize_for_scores(entropy_scores, method="minmax")
                norm_diversity = normalize_for_scores(diversity_scores, method="minmax")
                combined_scores = cfg.w_ed_e * norm_entropy + cfg.w_ed_d * norm_diversity
                top_k_indices = np.argsort(combined_scores)[-k_actual:]
                batch = [cand_list[i] for i in top_k_indices]
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "quantum_entropy":
            if quantum_sampler is not None and model is not None:
                L_aug = X_pool[labeled_idx]
                cand_idx = shared_quantum_cand_idx

                q_list, d_list, cand_list = [], [], []
                for u in cand_idx:
                    if hasattr(quantum_sampler, 'ensemble_quantum_entropy'):
                        quantum_score = quantum_sampler.ensemble_quantum_entropy(X_pool[u])
                    else:
                        quantum_score = quantum_sampler.compute_quantum_entropy(X_pool[u])

                    diversity_score = structure_diversity_score(X_pool[u], L_aug)
                    q_list.append(quantum_score)
                    d_list.append(diversity_score)
                    cand_list.append(int(u))

                norm_q = normalize_for_scores(q_list, method="minmax")
                norm_d = normalize_for_scores(d_list, method="minmax")
                total_scores = cfg.w_qd_q * norm_q + cfg.w_qd_d * norm_d
                top_k_indices = np.argsort(total_scores)[-k_actual:]
                batch = [cand_list[i] for i in top_k_indices]
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "quantum_uncertainty":
            if quantum_sampler is not None and model is not None:
                L_aug = X_pool[labeled_idx]
                cand_idx = shared_quantum_cand_idx

                q_list, c_list, d_list, cand_list = [], [], [], []
                for u in cand_idx:
                    if hasattr(quantum_sampler, 'ensemble_quantum_uncertainty'):
                        quantum_score = quantum_sampler.ensemble_quantum_uncertainty(X_pool[u])
                    else:
                        quantum_score = quantum_sampler.compute_quantum_uncertainty(X_pool[u])

                    if hasattr(model, 'predict_proba'):
                        try:
                            p1 = model.predict_proba(X_pool[u].reshape(1, -1))[0, 1]
                            classical_score = binary_entropy(p1)
                        except Exception:
                            classical_score = 0.5
                    else:
                        classical_score = 0.5

                    diversity_score = structure_diversity_score(X_pool[u], L_aug)
                    q_list.append(quantum_score)
                    c_list.append(classical_score)
                    d_list.append(diversity_score)
                    cand_list.append(int(u))

                norm_q = normalize_for_scores(q_list, method="minmax")
                norm_c = normalize_for_scores(c_list, method="minmax")
                norm_d = normalize_for_scores(d_list, method="minmax")
                total_scores = wq_r * norm_q + wc_r * norm_c + wd_r * norm_d
                top_k_indices = np.argsort(total_scores)[-k_actual:]
                batch = [cand_list[i] for i in top_k_indices]
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "pure_quantum":
            if quantum_sampler is not None:
                cand_idx = shared_quantum_cand_idx

                q_list, cand_list = [], []
                for u in cand_idx:
                    if hasattr(quantum_sampler, 'ensemble_quantum_entropy'):
                        quantum_entropy_val = quantum_sampler.ensemble_quantum_entropy(X_pool[u])
                    else:
                        quantum_entropy_val = quantum_sampler.compute_quantum_entropy(X_pool[u])

                    q_list.append(quantum_entropy_val)
                    cand_list.append(int(u))

                norm_q = normalize_for_scores(q_list, method="minmax")
                top_k_indices = np.argsort(norm_q)[-k_actual:]
                batch = [cand_list[i] for i in top_k_indices]
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "hybrid_uncertainty_quantum":
            if quantum_sampler is not None and model is not None:
                if r < 3:
                    probs = traditional_models.predict_proba(model_name, X_pool[unlabeled_idx])
                    uncertainties = np.abs(probs - 0.5)
                    selected_indices = np.argsort(uncertainties)[:k_actual]
                    batch = unlabeled_idx[selected_indices].tolist()
                else:
                    L_aug = X_pool[labeled_idx] if len(labeled_idx) > 0 else None
                    cand_idx = build_quantum_candidate_pool(
                        X_pool,
                        unlabeled_idx,
                        model,
                        k_actual,
                        rng,
                    )

                    q_list, d_list, cand_list = [], [], []
                    for u in cand_idx:
                        if hasattr(quantum_sampler, 'ensemble_quantum_entropy'):
                            quantum_entropy_val = quantum_sampler.ensemble_quantum_entropy(X_pool[u])
                        else:
                            quantum_entropy_val = quantum_sampler.compute_quantum_entropy(X_pool[u])

                        diversity_score = structure_diversity_score(X_pool[u], L_aug)
                        q_list.append(quantum_entropy_val)
                        d_list.append(diversity_score)
                        cand_list.append(int(u))

                    norm_q = normalize_for_scores(q_list, method="minmax")
                    norm_d = normalize_for_scores(d_list, method="minmax")
                    total_scores = cfg.w_qd_q * norm_q + cfg.w_qd_d * norm_d
                    top_k_indices = np.argsort(total_scores)[-k_actual:]
                    batch = [cand_list[i] for i in top_k_indices]
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "optimized_hybrid":
            if quantum_sampler is not None and len(traditional_models.trained_models) > 0:
                batch = select_optimized_hybrid_strategy(
                    quantum_sampler, model, X_pool, unlabeled_idx, labeled_idx, k_actual,
                    r, rounds, dataset_complexity, rng, cfg=cfg
                )
            else:
                batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        elif strategy == "high_dim_quantum":
            if dataset_complexity > 30 and quantum_sampler is not None and model is not None:
                batch = select_for_high_dimensional_data(
                    quantum_sampler, model, X_pool, unlabeled_idx, labeled_idx, k_actual, rng=rng, cfg=cfg
                )
            else:
                if quantum_sampler is not None and model is not None:
                    L_aug = X_pool[labeled_idx] if len(labeled_idx) > 0 else None
                    M = min(150, len(unlabeled_idx))
                    cand_idx = (
                        rng.choice(unlabeled_idx, size=M, replace=False)
                        if len(unlabeled_idx) > M else unlabeled_idx
                    )

                    q_list, d_list, cand_list = [], [], []
                    for u in cand_idx:
                        if hasattr(quantum_sampler, 'ensemble_quantum_entropy'):
                            quantum_entropy_val = quantum_sampler.ensemble_quantum_entropy(X_pool[u])
                        else:
                            quantum_entropy_val = quantum_sampler.compute_quantum_entropy(X_pool[u])

                        diversity_score = structure_diversity_score(X_pool[u], L_aug)
                        q_list.append(quantum_entropy_val)
                        d_list.append(diversity_score)
                        cand_list.append(int(u))

                    norm_q = normalize_for_scores(q_list, method="minmax")
                    norm_d = normalize_for_scores(d_list, method="minmax")
                    total_scores = cfg.w_qd_q * norm_q + cfg.w_qd_d * norm_d
                    top_k_indices = np.argsort(total_scores)[-k_actual:]
                    batch = [cand_list[i] for i in top_k_indices]
                else:
                    batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        else:
            batch = rng.choice(unlabeled_idx, size=k_actual, replace=False).tolist()

        labeled_idx = np.concatenate([labeled_idx, np.array(batch, dtype=int)])
        unlabeled_idx = np.array([i for i in unlabeled_idx if i not in batch], dtype=int)

    return {
        "acc_hist": acc_hist,
        "label_counts": label_counts,
        "final_accuracy": acc_hist[-1] if acc_hist else 0.5,
        "all_scores": [acc_hist[-1]]
    }


def load_real_datasets():
    datasets = {}
    
    print("loading real data...")
    
    uci_datasets = {
        'credit_approval': 'credit-approval',
        'bank_marketing': 'bank-marketing', 
        'adult_income': 'adult',
        'mushroom': 'mushroom',
        'spambase': 'spambase',
        'ionosphere': 'ionosphere',
        'sonar': 'sonar',
        'parkinsons': 'parkinsons',
        'heart_disease': 'heart-disease',
    }
    
    for name, dataset_id in uci_datasets.items():
        try:
            print(f"  loading {name}...")
            ds = fetch_openml(dataset_id, version=1, as_frame=True)
            frame = ds.frame

            y_ser = ds.target
            X_df  = ds.data

            if y_ser is None:
                if dataset_id == "heart-disease":
                    if "num" in frame.columns:
                        y_ser = frame["num"]
                        X_df  = frame.drop(columns=["num"])
                    elif "class" in frame.columns:
                        y_ser = frame["class"]
                        X_df  = frame.drop(columns=["class"])
                    else:
                        y_ser = frame.iloc[:, -1]
                        X_df  = frame.iloc[:, :-1]
                    print("    [info] heart_disease: using custom target column")
                else:
                    raise ValueError("No target column found (ds.target is None)")
            mask = ~X_df.isna().any(axis=1)
            if hasattr(y_ser, "isna"):
                mask = mask & ~y_ser.isna()
            X_df = X_df[mask]
            y_ser = y_ser[mask]

            X_df = X_df.copy()
            for col in X_df.columns:
                if not is_numeric_dtype(X_df[col]):
                    X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

            X = X_df.to_numpy()

            y_ser = y_ser.astype(str)
            y = LabelEncoder().fit_transform(y_ser)
            if len(np.unique(y)) > 2:
                majority_class = np.argmax(np.bincount(y))
                y = (y == majority_class).astype(int)

            mask2 = ~np.any(np.isnan(X), axis=1)
            X, y = X[mask2], y[mask2]

            if len(X) > 100:
                datasets[name] = (X.astype(float), y)
                print(f"    Success: {name}: {X.shape}")
            else:
                print(f"    Warning: {name}: insufficient samples ({X.shape[0]} rows)")

        except Exception as e:
            print(f"    Warning: failed to load {name}: {e}")
    try:
        cancer = load_breast_cancer()
        X_cancer = cancer.data.astype(float)
        datasets['breast_cancer'] = (X_cancer, cancer.target)
        print("    Success: breast_cancer")
    except Exception:
        pass
    
    try:
        digits = load_digits()
        mask = (digits.target == 0) | (digits.target == 1)
        X_digits = digits.data[mask]
        y_digits = digits.target[mask]
        datasets['digits_01'] = (X_digits, y_digits)
        print("    Success: digits_01")
    except Exception:
        pass
    
    try:
        iris = load_iris()
        X_iris = iris.data
        y_iris = (iris.target == 0).astype(int)
        datasets['iris_binary'] = (X_iris, y_iris)
        print("    Success: iris_binary")
    except Exception:
        pass
    
    try:
        wine = load_wine()
        X_wine = wine.data.astype(float)
        y_wine = (wine.target == 0).astype(int)
        datasets['wine_binary'] = (X_wine, y_wine)
        print("    Success: wine_binary")
    except Exception:
        pass
    
    try:
        diabetes = load_diabetes()
        X_diabetes = diabetes.data.astype(float)
        y_diabetes = (diabetes.target > np.median(diabetes.target)).astype(int)
        datasets['diabetes_binary'] = (X_diabetes, y_diabetes)
        print("    Success: diabetes_binary")
    except Exception:
        pass
    
    print(f" Success: {len(datasets)} datasets loaded.")
    return datasets


class EnhancedQuantumSamplingComparison:
    def __init__(self, use_enhanced_quantum=True):
        self.datasets = {}
        self.results = {}
        self.all_acc_histories = {}
        self.all_label_histories = {} 
        self.use_enhanced_quantum = use_enhanced_quantum
        self.stratified_results = {}

        self.best_cfgs = {}
        self.grid_search_results = {}
    
    def load_datasets(self):
        self.datasets = load_real_datasets()
        print(f"\ndatasets:")
        for name, (X, y) in self.datasets.items():
            unique, counts = np.unique(y, return_counts=True)
            balance = min(counts) / max(counts)
            complexity = X.shape[1]
            print(f"  {name:25} {X.shape} | balance: {balance:.2f} | complexity: {complexity}D")
    
    def run_experiments(self, seeds=range(0, 5), rounds=15, query_batch=5):
        sampling_methods = [
            "random", "uncertainty", "coreset", "badge", "clusteraware",
            "entropy_diversity", "quantum_entropy", "quantum_uncertainty",
            "pure_quantum", "hybrid_uncertainty_quantum", 
            "optimized_hybrid", "high_dim_quantum"
        ]
        
        model_types = ["logistic", "random_forest", "svm", "xgboost", "mlp"]
        
        all_results = {}
        self.all_acc_histories = {}
        self.all_label_histories = {}
        
        for dataset_name, (X, y) in self.datasets.items():
            print(f"\n=== {dataset_name}: ({X.shape[1]} features) ===")
            dataset_results = {}
            dataset_histories = {}
            dataset_label_histories = {}
            
            for model_type in model_types:
                print(f"  model: {model_type}")
                model_results = {}
                model_histories = {}
                model_label_histories = {}
                
                for method in sampling_methods:
                    print(f"    method: {method:25}", end="")
                    
                    cfg = None
                    if self.method_needs_cfg(method):
                        if method == "hybrid_uncertainty_quantum":
                            key = (dataset_name, model_type, "quantum_entropy")
                        else:
                            key = (dataset_name, model_type, method)
                        cfg = self.best_cfgs.get(key, None)
                        print(
                            f"   using cfg for ({dataset_name}, {model_type}, {method})"
                        )
                        if cfg is not None:
                            print(f" [cfg loaded]", end="")
                        else:
                            print(f" [default cfg]", end="")

                    acc_scores = []
                    all_acc_histories = []
                    label_histories = []
                    
                    for seed in seeds:
                        try:
                            result = run_active_learning_experiment(
                                X, y, model_type, method,
                                init_per_class=2, rounds=rounds, 
                                query_batch=query_batch, seed=seed,
                                use_enhanced_quantum=self.use_enhanced_quantum,
                                cfg=cfg
                            )
                            acc_scores.append(result['final_accuracy'])
                            all_acc_histories.append(result['acc_hist'])
                            label_histories.append(result['label_counts'])
                        except Exception as e:
                            print(f"E({str(e)[:20]})", end="")
                            traceback.print_exc()
                            acc_scores.append(0.5)
                            all_acc_histories.append([0.5] * (rounds + 1))
                            label_histories.append([0] * (rounds + 1))
                    
                    mean_acc = np.mean(acc_scores)
                    std_acc = np.std(acc_scores)
                    model_results[method] = {
                        'mean_accuracy': mean_acc,
                        'std_accuracy': std_acc,
                        'accuracy_str': f"{mean_acc:.3f} ± {std_acc:.3f}",
                        'all_scores': acc_scores,
                        'dataset_complexity': X.shape[1]
                    }
                    model_histories[method] = all_acc_histories
                    model_label_histories[method] = label_histories
                    
                    print(f" | acc: {mean_acc:.3f} ± {std_acc:.3f}")
                
                dataset_results[model_type] = model_results
                dataset_histories[model_type] = model_histories
                dataset_label_histories[model_type] = model_label_histories
            
            all_results[dataset_name] = dataset_results
            self.all_acc_histories[dataset_name] = dataset_histories
            self.all_label_histories[dataset_name] = dataset_label_histories
        
        self.results = all_results
        self.stratified_analysis()
        
        return all_results
    

    def stratified_analysis(self):
        print("\n" + "=" * 80)
        print("Stratified Analysis: By Dataset Complexity")
        print("=" * 80)

        complexity_levels = {
            'low': lambda d: d < 15,
            'medium': lambda d: 15 <= d < 40,
            'high': lambda d: d >= 40
        }

        quantum_methods = [
            'quantum_entropy', 'quantum_uncertainty', 'pure_quantum',
            'hybrid_uncertainty_quantum', 'optimized_hybrid', 'high_dim_quantum'
        ]
        classical_methods = ['random', 'uncertainty', 'badge', 'coreset']

        self.stratified_results = {
            level: {'quantum': [], 'classical': []}
            for level in complexity_levels
        }

        raw_records = []
        summary_records = []

        for dataset_name, (X, y) in self.datasets.items():
            complexity = X.shape[1]

            level = None
            for lvl, condition in complexity_levels.items():
                if condition(complexity):
                    level = lvl
                    break

            if level and dataset_name in self.results:
                dataset_results = self.results[dataset_name]

                for model_name, model_results in dataset_results.items():
                    for method, res in model_results.items():
                        if method in quantum_methods:
                            self.stratified_results[level]['quantum'].append(res['mean_accuracy'])
                            raw_records.append({
                                "complexity_level": level,
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method,
                                "group": "quantum",
                                "mean_accuracy": res['mean_accuracy']
                            })
                        elif method in classical_methods:
                            self.stratified_results[level]['classical'].append(res['mean_accuracy'])
                            raw_records.append({
                                "complexity_level": level,
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method,
                                "group": "classical",
                                "mean_accuracy": res['mean_accuracy']
                            })

        for level in complexity_levels.keys():
            quantum_scores = self.stratified_results[level]['quantum']
            classical_scores = self.stratified_results[level]['classical']

            p_value = None
            u_stat = None
            significant = None
            interpretation = "Insufficient data for comparison"

            if quantum_scores and classical_scores:
                quantum_mean = np.mean(quantum_scores)
                classical_mean = np.mean(classical_scores)
                difference = quantum_mean - classical_mean

                print(
                    f"\n{level.upper()} complexity datasets "
                    f"(n={len(quantum_scores)} quantum methods, {len(classical_scores)} classical methods):"
                )
                print(f"  Quantum methods mean: {quantum_mean:.4f}")
                print(f"  Classical methods mean: {classical_mean:.4f}")
                print(f"  Difference: {difference:.4f} ({'+' if difference > 0 else ''}{difference:.2%})")

                if len(quantum_scores) > 5 and len(classical_scores) > 5:
                    u_stat, p_value = mannwhitneyu(
                        quantum_scores, classical_scores, alternative='two-sided'
                    )
                    print(f"  Mann-Whitney U test: U={u_stat:.0f}, p={p_value:.6f}")

                    significant = bool(p_value < 0.05)

                    if p_value < 0.05:
                        if difference > 0:
                            interpretation = (
                                f"Quantum methods outperform classical methods on {level} complexity datasets."
                            )
                            print(f"  Significant result: {interpretation}")
                        else:
                            interpretation = (
                                f"Classical methods outperform quantum methods on {level} complexity datasets."
                            )
                            print(f"  Significant result: {interpretation}")
                    else:
                        interpretation = (
                            f"No significant difference between quantum and classical methods on {level} complexity datasets."
                        )
                        print(f"  No significant difference: On {level} complexity datasets, quantum and classical methods perform similarly.")
                else:
                    interpretation = "Not enough samples for Mann-Whitney U test."

                summary_records.append({
                    "complexity_level": level,
                    "n_quantum": len(quantum_scores),
                    "n_classical": len(classical_scores),
                    "quantum_mean": quantum_mean,
                    "classical_mean": classical_mean,
                    "difference": difference,
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "significant": significant,
                    "interpretation": interpretation
                })

        raw_df = pd.DataFrame(raw_records)
        summary_df = pd.DataFrame(summary_records)

        save_table(raw_df, "stratified_analysis_raw.csv")
        save_table(summary_df, "stratified_analysis_summary.csv")

        return summary_df, raw_df

    def generate_ranking_tables(self):
        print("\n" + "=" * 80)
        print("Accuracy rankings by dataset (including complexity information)")
        print("=" * 80)

        all_rankings = {}
        all_records = []

        for dataset_name, dataset_results in self.results.items():
            combinations = []

            for model, methods in dataset_results.items():
                for method, results in methods.items():
                    combinations.append({
                        'dataset': dataset_name,
                        'model': model,
                        'method': method,
                        'mean_accuracy': results['mean_accuracy'],
                        'std_accuracy': results['std_accuracy'],
                        'accuracy_str': results['accuracy_str'],
                        'dataset_complexity': results['dataset_complexity']
                    })

            combinations.sort(key=lambda x: x['mean_accuracy'], reverse=True)
            ranking_df = pd.DataFrame(combinations)
            from scipy.stats import rankdata
            ranking_df['rank'] = rankdata(
                -ranking_df['mean_accuracy'], method='average'
            )

            all_rankings[dataset_name] = ranking_df

            all_records.extend(combinations)

            print(f"\n{dataset_name} Accuracy Rankings (Top 5):")
            print(ranking_df.head().to_string(index=False))

            save_table(ranking_df, f"ranking_{dataset_name}.csv")

        all_df = pd.DataFrame(all_records)

        global_summary = (
            all_df.groupby(['model', 'method'])['mean_accuracy']
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .sort_values('mean', ascending=False)
        )

        from scipy.stats import rankdata
        global_summary['rank'] = rankdata(
            -global_summary['mean'], method='average'
        )

        print("\n" + "=" * 80)
        print("Global Ranking Across All Datasets")
        print("=" * 80)
        print(global_summary.head(10).to_string(index=False))

        save_table(global_summary, "ranking_global_summary.csv")

        return all_rankings
    

    def generate_summary(self):
        print("\n" + "=" * 80)
        print("Overall Summary of Method Performance Across All Datasets")
        print("=" * 80)

        combo_scores = {}
        raw_records = []

        for dataset_name, dataset_results in self.results.items():
            for model, methods in dataset_results.items():
                for method, results in methods.items():
                    combo = f"{model}_{method}"

                    if combo not in combo_scores:
                        combo_scores[combo] = []

                    combo_scores[combo].append(results['mean_accuracy'])

                    raw_records.append({
                        'dataset': dataset_name,
                        'model': model,
                        'method': method,
                        'combination': combo,
                        'mean_accuracy': results['mean_accuracy'],
                        'std_accuracy': results['std_accuracy'],
                        'dataset_complexity': results['dataset_complexity']
                    })

        summary_data = []
        for combo, scores in combo_scores.items():
            summary_data.append({
                'combination': combo,
                'avg_accuracy': np.mean(scores),
                'std_accuracy': np.std(scores),
                'accuracy_str': f"{np.mean(scores):.3f} ± {np.std(scores):.3f}",
                'min_accuracy': np.min(scores),
                'max_accuracy': np.max(scores),
                'n_datasets': len(scores)
            })

        summary_data.sort(key=lambda x: x['avg_accuracy'], reverse=True)
        summary_df = pd.DataFrame(summary_data)
        summary_df['overall_rank'] = range(1, len(summary_df) + 1)

        raw_df = pd.DataFrame(raw_records)

        print("\nTop 20 Best Combinations:")
        print(summary_df.head(20).to_string(index=False))

        save_table(summary_df, "overall_summary.csv")
        save_table(raw_df, "overall_summary_raw.csv")

        return summary_df


    def analyze_quantum_performance(self):
        print("\n" + "=" * 80)
        print("Quantum Sampling Performance Analysis (Enhanced Version)")
        print("=" * 80)

        quantum_methods = [
            'quantum_entropy', 'quantum_uncertainty', 'pure_quantum',
            'hybrid_uncertainty_quantum', 'optimized_hybrid', 'high_dim_quantum'
        ]
        classical_methods = ['random', 'uncertainty', 'badge', 'coreset']

        quantum_scores = {method: [] for method in quantum_methods}
        classical_scores = {method: [] for method in classical_methods}

        raw_records = []
        method_summary_records = []
        comparison_records = []

        for dataset_name, dataset_results in self.results.items():
            for model, methods in dataset_results.items():
                for method, results in methods.items():
                    if method in quantum_methods:
                        quantum_scores[method].append(results['mean_accuracy'])
                        raw_records.append({
                            'dataset': dataset_name,
                            'model': model,
                            'method': method,
                            'group': 'quantum',
                            'mean_accuracy': results['mean_accuracy']
                        })
                    elif method in classical_methods:
                        classical_scores[method].append(results['mean_accuracy'])
                        raw_records.append({
                            'dataset': dataset_name,
                            'model': model,
                            'method': method,
                            'group': 'classical',
                            'mean_accuracy': results['mean_accuracy']
                        })

        print("\nQuantum Sampling Methods Average Performance:")
        for method in quantum_methods:
            if quantum_scores[method]:
                avg = np.mean(quantum_scores[method])
                std = np.std(quantum_scores[method])
                count = len(quantum_scores[method])

                print(f"  {method:30}: {avg:.3f} ± {std:.3f} (n={count})")

                method_summary_records.append({
                    'method': method,
                    'group': 'quantum',
                    'avg_accuracy': avg,
                    'std_accuracy': std,
                    'n': count
                })

        print("\nClassical Sampling Methods Average Performance:")
        for method in classical_methods:
            if classical_scores[method]:
                avg = np.mean(classical_scores[method])
                std = np.std(classical_scores[method])
                count = len(classical_scores[method])

                print(f"  {method:30}: {avg:.3f} ± {std:.3f} (n={count})")

                method_summary_records.append({
                    'method': method,
                    'group': 'classical',
                    'avg_accuracy': avg,
                    'std_accuracy': std,
                    'n': count
                })

        old_quantum_methods = [
            'quantum_entropy', 'quantum_uncertainty',
            'pure_quantum', 'hybrid_uncertainty_quantum'
        ]
        new_quantum_methods = ['optimized_hybrid', 'high_dim_quantum']

        old_scores = []
        new_scores = []
        for method in old_quantum_methods:
            old_scores.extend(quantum_scores[method])
        for method in new_quantum_methods:
            new_scores.extend(quantum_scores[method])

        if old_scores and new_scores:
            old_avg = np.mean(old_scores)
            new_avg = np.mean(new_scores)
            improvement = new_avg - old_avg
            p_value = None
            u_stat = None

            print(f"\nNew vs. Old Quantum Methods Comparison:")
            print(f"  Old Quantum Methods Average: {old_avg:.4f} (n={len(old_scores)})")
            print(f"  New Quantum Methods Average: {new_avg:.4f} (n={len(new_scores)})")
            print(f"  Improvement: {improvement:.4f} ({'+' if improvement > 0 else ''}{improvement:.2%})")

            if len(old_scores) > 10 and len(new_scores) > 10:
                u_stat, p_value = mannwhitneyu(new_scores, old_scores, alternative='two-sided')
                print(f"  Mann-Whitney U Test: U={u_stat:.0f}, p={p_value:.6f}")

            comparison_records.append({
                'comparison': 'new_quantum_vs_old_quantum',
                'group1_name': 'new_quantum',
                'group2_name': 'old_quantum',
                'group1_mean': new_avg,
                'group2_mean': old_avg,
                'difference': improvement,
                'n_group1': len(new_scores),
                'n_group2': len(old_scores),
                'u_stat': u_stat,
                'p_value': p_value
            })

        all_quantum = []
        all_classical = []
        for method in quantum_methods:
            all_quantum.extend(quantum_scores[method])
        for method in classical_methods:
            all_classical.extend(classical_scores[method])

        if all_quantum and all_classical:
            quantum_avg = np.mean(all_quantum)
            classical_avg = np.mean(all_classical)
            improvement = quantum_avg - classical_avg
            p_value = None
            u_stat = None

            print(f"\nOverall comparison:")
            print(f"  Mean of all quantum methods: {quantum_avg:.4f} (n={len(all_quantum)})")
            print(f"  Mean of all classical methods: {classical_avg:.4f} (n={len(all_classical)})")
            print(f"  Difference: {improvement:.4f} ({'+' if improvement > 0 else ''}{improvement:.2%})")

            if len(all_quantum) > 10 and len(all_classical) > 10:
                u_stat, p_value = mannwhitneyu(all_quantum, all_classical, alternative='two-sided')
                print(f"  Mann-Whitney U Test: U={u_stat:.0f}, p={p_value:.6f}")

            comparison_records.append({
                'comparison': 'all_quantum_vs_all_classical',
                'group1_name': 'all_quantum',
                'group2_name': 'all_classical',
                'group1_mean': quantum_avg,
                'group2_mean': classical_avg,
                'difference': improvement,
                'n_group1': len(all_quantum),
                'n_group2': len(all_classical),
                'u_stat': u_stat,
                'p_value': p_value
            })

        raw_df = pd.DataFrame(raw_records)
        method_summary_df = pd.DataFrame(method_summary_records)
        comparison_df = pd.DataFrame(comparison_records)

        save_table(raw_df, "quantum_performance_raw.csv")
        save_table(method_summary_df, "quantum_method_summary.csv")
        save_table(comparison_df, "quantum_group_comparison.csv")

    def analyze_label_efficiency(self, tau_list=[0.9, 0.95]):
        if not self.all_acc_histories or not self.all_label_histories:
            print("Experiments have not been run yet. Label efficiency analysis is unavailable.")
            return

        methods_of_interest = ['random', 'uncertainty', 'quantum_entropy', 'pure_quantum', 'optimized_hybrid']

        print("\n" + "=" * 80)
        print("Label Efficiency Analysis: Sample Count N_T Required to Reach Given Threshold Accuracy")
        print("=" * 80)

        raw_records = []
        summary_records = []

        for tau in tau_list:
            print(f"\n--- Threshold Ratio tau = {tau:.2f} ---")
            method_Nt_global = {m: [] for m in methods_of_interest}
            method_total_cases = {m: 0 for m in methods_of_interest}
            method_reached_cases = {m: 0 for m in methods_of_interest}

            for dataset_name in self.datasets.keys():
                if dataset_name not in self.all_acc_histories or dataset_name not in self.all_label_histories:
                    continue

                dataset_hist = self.all_acc_histories[dataset_name]
                dataset_labels = self.all_label_histories[dataset_name]

                for model_name in dataset_hist.keys():
                    model_hist = dataset_hist[model_name]
                    model_label_hist = dataset_labels[model_name]

                    best_acc = 0.0
                    for method_all, curves_list in model_hist.items():
                        for curve in curves_list:
                            best_acc = max(best_acc, max(curve))

                    if best_acc <= 0:
                        continue

                    threshold = tau * best_acc

                    for method in methods_of_interest:
                        if method not in model_hist:
                            continue

                        curves = model_hist[method]
                        label_curves = model_label_hist[method]

                        for s in range(len(curves)):
                            method_total_cases[method] += 1

                            acc_curve = curves[s]
                            lab_curve = label_curves[s]
                            Nt = None

                            for r, acc in enumerate(acc_curve):
                                if acc >= threshold:
                                    Nt = lab_curve[r]
                                    break

                            reached = Nt is not None
                            if reached:
                                method_Nt_global[method].append(Nt)
                                method_reached_cases[method] += 1

                            raw_records.append({
                                "tau": tau,
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method,
                                "seed_idx": s,
                                "best_acc": best_acc,
                                "threshold": threshold,
                                "reached_threshold": reached,
                                "N_T": Nt if Nt is not None else np.nan
                            })

            for method in methods_of_interest:
                Nt_list = method_Nt_global[method]
                total_cases = method_total_cases[method]
                reached_cases = method_reached_cases[method]
                reach_rate = reached_cases / total_cases if total_cases > 0 else np.nan

                if Nt_list:
                    Nt_arr = np.array(Nt_list)
                    mean_Nt = Nt_arr.mean()
                    std_Nt = Nt_arr.std()
                    print(f"  Method {method:20}: N_T = {mean_Nt:6.1f} ± {std_Nt:6.1f} (n={len(Nt_arr)}), reach_rate={reach_rate:.2%}")
                else:
                    mean_Nt = np.nan
                    std_Nt = np.nan
                    print(f"  Method {method:20}: threshold not reached, N_T data is empty, reach_rate={reach_rate:.2%}")

                summary_records.append({
                    "tau": tau,
                    "method": method,
                    "mean_NT": mean_Nt,
                    "std_NT": std_Nt,
                    "n_reached": reached_cases,
                    "n_total": total_cases,
                    "reach_rate": reach_rate
                })

            if abs(tau - 0.95) < 1e-6:
                methods_plot = []
                means_plot = []
                stds_plot = []

                for method in methods_of_interest:
                    Nt_list = method_Nt_global[method]
                    if Nt_list:
                        methods_plot.append(method)
                        Nt_arr = np.array(Nt_list)
                        means_plot.append(Nt_arr.mean())
                        stds_plot.append(Nt_arr.std())

                if methods_plot:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(methods_plot))
                    bars = ax.bar(x, means_plot, yerr=stds_plot, capsize=5, alpha=0.8)

                    ax.set_xticks(x)
                    ax.set_xticklabels(methods_plot, rotation=30, ha='right')
                    ax.set_ylabel(f'N_T (labels to reach {tau:.0%} * best accuracy)')
                    ax.set_title(
                        f'Label Efficiency Comparison (tau = {tau:.2f}) - The Lower, The Better',
                        fontweight='bold'
                    )
                    ax.grid(axis='y', alpha=0.3)

                    for bar, m in zip(bars, means_plot):
                        h = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5, f'{m:.1f}',
                                ha='center', va='bottom')

                    save_figure(fig, "label_efficiency_tau_095.png")

        raw_df = pd.DataFrame(raw_records)
        summary_df = pd.DataFrame(summary_records)

        save_table(raw_df, "label_efficiency_raw.csv")
        save_table(summary_df, "label_efficiency_summary.csv")
        
    def statistical_significance_test(self):
        print("\n" + "=" * 80)
        print("Statistical Significance Test: Quantum Methods vs Classical Methods")
        print("=" * 80)

        quantum_methods = ['quantum_entropy', 'quantum_uncertainty', 'pure_quantum',
                          'hybrid_uncertainty_quantum', 'optimized_hybrid', 'high_dim_quantum']
        classical_methods = ['random', 'uncertainty', 'badge', 'coreset', 'clusteraware']

        quantum_scores_all, classical_scores_all = [], []
        quantum_by_dataset, classical_by_dataset = {}, {}
        raw_records, dataset_records = [], []

        for dataset_name in self.datasets.keys():
            quantum_scores, classical_scores = [], []
            for model_name, methods_results in self.results[dataset_name].items():
                for method_name, results in methods_results.items():
                    if method_name in quantum_methods:
                        quantum_scores.extend(results['all_scores'])
                        raw_records.extend([{"dataset": dataset_name, "model": model_name, "method": method_name,
                                            "group": "quantum", "score": score} for score in results['all_scores']])
                    elif method_name in classical_methods:
                        classical_scores.extend(results['all_scores'])
                        raw_records.extend([{"dataset": dataset_name, "model": model_name, "method": method_name,
                                            "group": "classical", "score": score} for score in results['all_scores']])

            if quantum_scores and classical_scores:
                q_mean, c_mean = np.mean(quantum_scores), np.mean(classical_scores)
                quantum_by_dataset[dataset_name], classical_by_dataset[dataset_name] = q_mean, c_mean
                quantum_scores_all.extend(quantum_scores)
                classical_scores_all.extend(classical_scores)
                dataset_records.append({"dataset": dataset_name, "quantum_mean": q_mean, "classical_mean": c_mean,
                                       "difference": q_mean - c_mean, "n_quantum": len(quantum_scores),
                                       "n_classical": len(classical_scores)})

        if not quantum_scores_all or not classical_scores_all:
            print("Data insufficient for statistical testing")
            return

        print("\n1. Descriptive Statistics:")
        q_mean_all, c_mean_all = np.mean(quantum_scores_all), np.mean(classical_scores_all)
        q_std_all, c_std_all = np.std(quantum_scores_all), np.std(classical_scores_all)
        mean_diff = q_mean_all - c_mean_all
        print(f"   Quantum Methods: n={len(quantum_scores_all)}, mean={q_mean_all:.4f}, std={q_std_all:.4f}")
        print(f"   Classical Methods: n={len(classical_scores_all)}, mean={c_mean_all:.4f}, std={c_std_all:.4f}")
        print(f"   Mean Difference: {mean_diff:.4f}")

        print("\n2. Normality Test (Shapiro-Wilk):")
        shapiro_q, shapiro_c = shapiro(quantum_scores_all), shapiro(classical_scores_all)
        print(f"   Quantum Methods: W={shapiro_q.statistic:.4f}, p={shapiro_q.pvalue:.6f}")
        print(f"   Classical Methods: W={shapiro_c.statistic:.4f}, p={shapiro_c.pvalue:.6f}")
        parametric_test = shapiro_q.pvalue > 0.05 and shapiro_c.pvalue > 0.05
        print("   Success: Both methods' data follow a normal distribution" if parametric_test
              else "   Warning: Data does not follow a normal distribution, using non-parametric test")

        print("\n3. Mean Difference Test:")
        test_statistic = None

        if parametric_test:
            t_stat, p_value = ttest_ind(quantum_scores_all, classical_scores_all, equal_var=False)
            print(f"   Welch's t-test: t={t_stat:.4f}, p={p_value:.6f}")
            test_name, test_statistic = "Welch's t-test", t_stat
        else:
            u_stat, p_value = mannwhitneyu(quantum_scores_all, classical_scores_all, alternative='two-sided')
            print(f"   Mann-Whitney U-test: U={u_stat:.0f}, p={p_value:.6f}")
            test_name, test_statistic = "Mann-Whitney U-test", u_stat

        print("\n4. Effect Size Analysis:")
        n1, n2 = len(quantum_scores_all), len(classical_scores_all)
        pooled_std = np.sqrt(((n1 - 1) * np.var(quantum_scores_all, ddof=1) + (n2 - 1) * np.var(classical_scores_all, ddof=1)) / (n1 + n2 - 2))
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        effect_label = ("Negligible" if abs(cohens_d) < 0.2 else "Small" if abs(cohens_d) < 0.5
                       else "Medium" if abs(cohens_d) < 0.8 else "Large")
        print(f"   Cohen's d = {cohens_d:.4f}\n   Effect size: {effect_label}")

        print("\n5. Statistical significance conclusion:")
        alpha = 0.05
        significant = p_value < alpha
        if significant:
            conclusion = ("Quantum methods significantly outperform classical methods" if mean_diff > 0
                         else "Classical methods significantly outperform quantum methods")
            print(f"   Success: {conclusion} (p={p_value:.6f} < {alpha})")
        else:
            conclusion = "No significant difference between quantum and classical methods"
            print(f"   Warning: {conclusion} (p={p_value:.6f} > {alpha})")

        raw_df, dataset_df = pd.DataFrame(raw_records), pd.DataFrame(dataset_records)
        summary_df = pd.DataFrame([{"n_quantum": n1, "n_classical": n2, "quantum_mean": q_mean_all,
                                   "classical_mean": c_mean_all, "mean_difference": mean_diff, "quantum_std": q_std_all,
                                   "classical_std": c_std_all, "shapiro_q_w": shapiro_q.statistic,
                                   "shapiro_q_p": shapiro_q.pvalue, "shapiro_c_w": shapiro_c.statistic,
                                   "shapiro_c_p": shapiro_c.pvalue, "parametric_test": parametric_test,
                                   "test_name": test_name, "test_statistic": test_statistic, "p_value": p_value,
                                   "cohens_d": cohens_d, "effect_size_label": effect_label, "alpha": alpha,
                                   "significant": significant, "conclusion": conclusion}])
        save_table(raw_df, "statistical_test_raw_scores.csv")
        save_table(dataset_df, "statistical_test_dataset_summary.csv")
        save_table(summary_df, "statistical_test_summary.csv")
        self.plot_statistical_comparison(quantum_scores_all, classical_scores_all, quantum_by_dataset,
                                        classical_by_dataset, test_name, p_value, cohens_d, mean_diff)
        return {'quantum_scores': quantum_scores_all, 'classical_scores': classical_scores_all, 'p_value': p_value,
                'cohens_d': cohens_d, 'mean_difference': mean_diff, 'significant': significant}


    def plot_statistical_comparison(self, quantum_scores, classical_scores,
                                quantum_by_dataset, classical_by_dataset,
                                test_name, p_value, cohens_d, mean_diff):

        datasets = list(quantum_by_dataset.keys())
        quantum_means = [quantum_by_dataset[d] for d in datasets]
        classical_means = [classical_by_dataset[d] for d in datasets]
        differences = [q - c for q, c in zip(quantum_means, classical_means)]

        self._plot_stat_distribution(
            quantum_scores, classical_scores, p_value
        )
        self._plot_stat_per_dataset(
            datasets, quantum_means, classical_means
        )
        self._plot_stat_advantage(
            differences, mean_diff
        )
        self._plot_stat_overall_summary(
            quantum_scores, classical_scores,
            test_name, p_value, cohens_d, mean_diff
        )

    def _plot_stat_distribution(self, quantum_scores, classical_scores, p_value):
        fig, ax = plt.subplots(figsize=(7, 5))

        data = [quantum_scores, classical_scores]
        labels = ['Quantum Methods', 'Classical Methods']
        box = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        med_q = np.median(quantum_scores)
        med_c = np.median(classical_scores)

        ax.set_ylabel('Accuracy')
        ax.set_title('Distribution Comparison: Quantum vs Classical', fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax.text(
            0.5, 0.95, f'p-value = {p_value:.6f}',
            transform=ax.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        ax.text(1, med_q + 0.005, f'median={med_q:.3f}', ha='center', fontsize=9)
        ax.text(2, med_c + 0.005, f'median={med_c:.3f}', ha='center', fontsize=9)

        save_figure(fig, "stat_dist_quantum_vs_classical.png")

    def _plot_stat_per_dataset(self, datasets, quantum_means, classical_means):
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(datasets))
        width = 0.35

        ax.bar(x - width/2, classical_means, width, label='Classical', alpha=0.7, color='lightblue')
        ax.bar(x + width/2, quantum_means, width, label='Quantum', alpha=0.7, color='lightcoral')

        for i, (c, q) in enumerate(zip(classical_means, quantum_means)):
            ax.plot([x[i] - width/2, x[i] + width/2], [c, q], 'k-', alpha=0.5, linewidth=1)

            mid_y = (c + q) / 2
            diff = q - c
            ax.text(x[i], mid_y + 0.003, f'{diff:+.3f}', ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([d[:10] + '...' if len(d) > 10 else d for d in datasets],
                        rotation=45, ha='right')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Per-Dataset Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_figure(fig, "stat_per_dataset_comparison.png")

    def _plot_stat_advantage(self, differences, mean_diff):
        fig, ax = plt.subplots(figsize=(8, 5))

        colors_diff = ['green' if diff > 0 else 'red' for diff in differences]
        bars = ax.bar(range(len(differences)), differences, color=colors_diff, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=mean_diff, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Mean diff: {mean_diff:.4f}')

        ax.text(len(differences) - 1, mean_diff, f' mean={mean_diff:.4f}',
                color='blue', fontsize=9, va='bottom', ha='right')

        if differences:
            max_idx = int(np.argmax(differences))
            min_idx = int(np.argmin(differences))
            for idx in {max_idx, min_idx}:
                ax.text(idx, differences[idx] + 0.003*np.sign(differences[idx] if differences[idx] != 0 else 1),
                        f'{differences[idx]:+.3f}', ha='center', fontsize=8)

        ax.set_xlabel('Dataset Index')
        ax.set_ylabel('Accuracy Difference (Quantum - Classical)')
        ax.set_title('Quantum Advantage by Dataset', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_figure(fig, "stat_quantum_advantage_by_dataset.png")


    def _plot_stat_overall_summary(self, quantum_scores, classical_scores,
                                test_name, p_value, cohens_d, mean_diff):
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['Quantum\nMethods', 'Classical\nMethods']
        means = [np.mean(quantum_scores), np.mean(classical_scores)]
        stds = [np.std(quantum_scores), np.std(classical_scores)]

        bars = ax.bar(methods, means, yerr=stds, capsize=10, alpha=0.7,
                    color=['lightcoral', 'lightblue'])

        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Performance Summary', fontweight='bold')
        ax.grid(True, alpha=0.3)

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')

        summary_text = (
            f"Statistical Test: {test_name}\n"
            f"p-value: {p_value:.6f}\n"
            f"Cohen's d: {cohens_d:.4f}\n"
            f"Mean difference: {mean_diff:.4f}"
        )

        if p_value < 0.05:
            summary_text += "\nQuantum better" if mean_diff > 0 else "\nClassical better"
        else:
            summary_text += "\nNo significant difference"

        ax.text(
            1.05, 0.5,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            ha='left',
            va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75, pad=0.4)
        )
        fig.subplots_adjust(right=0.68)
        save_figure(fig, "stat_overall_summary.png")

    def plot_comprehensive_analysis(self):
        print("\nGenerating visualization analysis...")
        self.plot_performance_comparison()
        self.plot_learning_curves()
        self.plot_ranking_heatmap()
        self.plot_quantum_advantage()
        self.plot_model_adaptability()
        self.plot_stratified_performance()
        self.plot_stratified_performance_test()


    def plot_stratified_performance_test(self):
        if not self.stratified_results:
            print("No stratified results found, skipping stratified performance chart")
            return

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import mannwhitneyu

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        complexity_levels = ['low', 'medium', 'high']
        level_names = ['Low Complexity', 'Medium Complexity', 'High Complexity']

        for ax, level, name in zip(axes, complexity_levels, level_names):
            quantum_scores = self.stratified_results[level]['quantum']
            classical_scores = self.stratified_results[level]['classical']

            ax.set_title(name, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.grid(True, alpha=0.3)

            if quantum_scores and classical_scores:
                box = ax.boxplot(
                    [quantum_scores, classical_scores],
                    labels=['Quantum', 'Classical'],
                    patch_artist=True
                )
                for patch, color in zip(box['boxes'], ['lightcoral', 'lightblue']):
                    patch.set_facecolor(color)

                q_mean = np.mean(quantum_scores)
                c_mean = np.mean(classical_scores)
                diff = q_mean - c_mean

                try:
                    stat, p_value = mannwhitneyu(quantum_scores, classical_scores, alternative='two-sided')
                    p_text = f"{p_value:.4f}" if p_value >= 1e-4 else "< 1e-4"
                except Exception:
                    p_text = "N/A"

                ax.text(
                    0.98, 0.98,
                    f"Q mean = {q_mean:.3f}\n"
                    f"C mean = {c_mean:.3f}\n"
                    f"Diff = {diff:+.3f}\n"
                    f"p-value = {p_text}",
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75, pad=0.35)
                )
            else:
                ax.text(
                    0.5, 0.5, 'Insufficient data',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=12
                )

        fig.suptitle(
            'Quantum vs Classical Performance Across Dataset Complexity Levels',
            fontsize=16, fontweight='bold', y=1.02
        )
        fig.tight_layout()
        save_figure(fig, "stratified_performance_test.png")


    def plot_learning_curves(self):
        methods = [
            ('random', 'Random', 'gray'),
            ('uncertainty', 'Uncertainty', 'blue'),
            ('pure_quantum', 'Pure-Q', 'red'),
            ('optimized_hybrid', 'Optimized-Hybrid', 'purple'),
        ]

        for dataset_name, dataset_histories in self.all_acc_histories.items():
            fig, ax = plt.subplots(figsize=(8, 5))

            for method, label, color in methods:
                curves = [
                    curve
                    for model_histories in dataset_histories.values()
                    if method in model_histories
                    for curve in model_histories[method]
                ]
                if not curves:
                    continue

                curves = np.asarray(curves)
                mean_curve = curves.mean(axis=0)
                rounds = np.arange(len(mean_curve))

                ax.plot(rounds, mean_curve, label=label, linewidth=2, color=color)

            ax.set_xlabel('Round')
            ax.set_ylabel('Accuracy')
            ax.set_title(dataset_name, fontweight='bold')
            ax.set_xlim(left=0)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=10, loc='best', frameon=True)

            fig.tight_layout()
            safe_name = dataset_name.lower().replace(" ", "_").replace("/", "_")
            save_figure(fig, f"learning_curve_{safe_name}.png")

        self.plot_average_learning_curve()


    def plot_average_learning_curve(self):
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = [
            ('random', 'Random', 'gray'),
            ('uncertainty', 'Uncertainty', 'blue'),
            ('pure_quantum', 'Pure-Q', 'red'),
            ('optimized_hybrid', 'Optimized-Hybrid', 'purple'),
        ]

        for method, label, color in methods:
            all_dataset_curves = []

            for dataset_name, dataset_histories in self.all_acc_histories.items():
                curves = [
                    curve
                    for model_histories in dataset_histories.values()
                    if method in model_histories
                    for curve in model_histories[method]
                ]
                if not curves:
                    continue

                curves = np.asarray(curves)
                dataset_mean = curves.mean(axis=0)
                all_dataset_curves.append(dataset_mean)

            if not all_dataset_curves:
                continue

            all_dataset_curves = np.asarray(all_dataset_curves)
            overall_mean = all_dataset_curves.mean(axis=0)
            rounds = np.arange(len(overall_mean))

            ax.plot(rounds, overall_mean, label=label, linewidth=2.8, color=color)

        ax.set_xlabel('Learning Round', fontsize=12)
        ax.set_ylabel('Mean Accuracy', fontsize=12)
        ax.set_title('Average Learning Curve Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=11, loc='lower right', frameon=True)

        fig.tight_layout()
        save_figure(fig, "average_learning_curve.png")


    def plot_stratified_performance(self):
        if not self.stratified_results:
            print("No stratified results found, skipping stratified performance chart")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        complexity_levels = ['low', 'medium', 'high']
        level_names = ['Low Complexity', 'Medium Complexity', 'High Complexity']

        for ax, level, name in zip(axes, complexity_levels, level_names):
            quantum_scores = self.stratified_results[level]['quantum']
            classical_scores = self.stratified_results[level]['classical']

            ax.set_title(name, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.grid(True, alpha=0.3)

            if quantum_scores and classical_scores:
                box = ax.boxplot([quantum_scores, classical_scores], labels=['Quantum', 'Classical'], patch_artist=True)
                for patch, color in zip(box['boxes'], ['lightcoral', 'lightblue']):
                    patch.set_facecolor(color)

                q_mean, c_mean = np.mean(quantum_scores), np.mean(classical_scores)
                ax.text(0.98, 0.98, f"Q mean = {q_mean:.3f}\nC mean = {c_mean:.3f}\nMean diff = {q_mean-c_mean:+.3f}",
                        transform=ax.transAxes, ha='right', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75, pad=0.35))
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)

        fig.suptitle('Quantum vs Classical Performance Across Dataset Complexity Levels',
                    fontsize=16, fontweight='bold', y=1.02)
        save_figure(fig, "stratified_performance.png")


    def plot_performance_comparison(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        method_cfg = [
            ('random', 'Random'), ('uncertainty', 'Unc'), ('badge', 'BADGE'),
            ('coreset', 'Coreset'), ('quantum_entropy', 'Q-Ent'),
            ('quantum_uncertainty', 'Q+Unc'), ('pure_quantum', 'Pure-Q'),
            ('optimized_hybrid', 'Opt-Hybrid'), ('high_dim_quantum', 'HighDim-Q'),
        ]
        quantum_methods = {'quantum_entropy', 'quantum_uncertainty', 'pure_quantum', 'optimized_hybrid', 'high_dim_quantum'}
        classical_methods = {'random', 'uncertainty', 'badge', 'coreset'}

        method_performance = {m: [] for m, _ in method_cfg}
        for dataset_results in self.results.values():
            for model_results in dataset_results.values():
                for method, results in model_results.items():
                    if method in method_performance:
                        method_performance[method].append(results['mean_accuracy'])

        valid_methods = [(m, label) for m, label in method_cfg if method_performance[m]]
        method_names = [label for _, label in valid_methods]
        performance_data = [method_performance[m] for m, _ in valid_methods]
        means = [np.mean(method_performance[m]) for m, _ in valid_methods]
        stds = [np.std(method_performance[m]) for m, _ in valid_methods]

        def add_labels(ax, bars, values):
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        box = ax1.boxplot(performance_data, labels=method_names, patch_artist=True)
        colors = ['lightgray']*4 + ['lightcoral']*3 + ['plum']*2
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_title('Performance Distribution Across Sampling Methods', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        bars = ax2.bar(method_names, means, yerr=stds, capsize=5, alpha=0.75)
        ax2.set_title('Average Performance of Sampling Methods', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, axis='y', alpha=0.3)
        add_labels(ax2, bars, means)

        q_vals = [np.mean(method_performance[m]) for m, _ in valid_methods if m in quantum_methods]
        c_vals = [np.mean(method_performance[m]) for m, _ in valid_methods if m in classical_methods]
        if q_vals and c_vals:
            cat_means = [np.mean(q_vals), np.mean(c_vals)]
            cat_stds = [np.std(q_vals), np.std(c_vals)]
            bars = ax3.bar(['Quantum', 'Classical'], cat_means, yerr=cat_stds, capsize=5, alpha=0.75, color=['lightcoral', 'lightblue'])
            ax3.set_title('Quantum vs Classical Performance', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Average Accuracy')
            ax3.grid(True, axis='y', alpha=0.3)
            add_labels(ax3, bars, cat_means)

        model_cfg = [('logistic', 'Logistic'), ('random_forest', 'RF'), ('svm', 'SVM'), ('xgboost', 'XGBoost'), ('mlp', 'MLP')]
        model_performance = {m: [] for m, _ in model_cfg}
        for dataset_results in self.results.values():
            for model_name, methods_results in dataset_results.items():
                for results in methods_results.values():
                    model_performance[model_name].append(results['mean_accuracy'])

        model_names = [label for m, label in model_cfg]
        model_means = [np.mean(model_performance[m]) for m, _ in model_cfg]
        model_stds = [np.std(model_performance[m]) for m, _ in model_cfg]
        bars = ax4.bar(model_names, model_means, yerr=model_stds, capsize=5, alpha=0.75)
        ax4.set_title('Average Performance Across Models', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, axis='y', alpha=0.3)
        add_labels(ax4, bars, model_means)

        save_figure(fig, "performance_comparison.png")


    def plot_ranking_heatmap(self):
        methods = ['random', 'uncertainty', 'badge', 'coreset', 'clusteraware',
                'entropy_diversity', 'quantum_entropy', 'quantum_uncertainty',
                'pure_quantum', 'hybrid_uncertainty_quantum', 'optimized_hybrid', 'high_dim_quantum']
        method_names = ['Random', 'Unc', 'BADGE', 'Coreset', 'Cluster',
                        'Ent+Div', 'Q-Ent', 'Q+Unc', 'Pure-Q', 'Hybrid-Q', 'Opt-Hybrid', 'HighDim-Q']
        dataset_names = list(self.datasets.keys())
        if not dataset_names:
            print("[warn] No datasets found for ranking heatmap")
            return

        ranking_matrix = np.full((len(methods), len(dataset_names)), np.nan)
        
        for j, dataset_name in enumerate(dataset_names):
            method_scores = {m: [] for m in methods}
            for model_results in self.results.get(dataset_name, {}).values():
                for m, res in model_results.items():
                    if m in method_scores:
                        method_scores[m].append(res['mean_accuracy'])
            
            avg_scores = np.array([np.mean(method_scores[m]) if method_scores[m] else np.nan for m in methods])
            valid = ~np.isnan(avg_scores)
            if np.any(valid):
                ranking_matrix[valid, j] = rankdata(-avg_scores[valid], method='average')

        method_order = np.nanmean(ranking_matrix, axis=1).argsort()
        dataset_order = np.nanmean(ranking_matrix, axis=0).argsort()
        
        ranking_matrix = ranking_matrix[method_order][:, dataset_order]
        method_names = [method_names[i] for i in method_order]
        dataset_names = [dataset_names[i] for i in dataset_order]

        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(np.arange(len(dataset_names)))
        ax.set_yticks(np.arange(len(method_names)))
        ax.set_xticklabels([d[:12]+'...' if len(d)>12 else d for d in dataset_names], rotation=45, ha='right')
        ax.set_yticklabels(method_names)

        threshold = np.nanmean(ranking_matrix)
        for i in range(len(method_names)):
            for j in range(len(dataset_names)):
                val = ranking_matrix[i, j]
                txt = "-" if np.isnan(val) else f"{int(val)}"
                color = "black" if np.isnan(val) or val <= threshold else "white"
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)

        ax.set_title('Ranking Heatmap of Sampling Methods Across Datasets (1 = Best)', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Rank (1 = Best)')
        save_figure(fig, "ranking_heatmap.png")
        save_table(pd.DataFrame(ranking_matrix, index=method_names, columns=dataset_names).reset_index().rename(columns={'index': 'method'}), "ranking_heatmap_matrix.csv")


    def plot_quantum_advantage(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        quantum_methods = ['quantum_entropy', 'quantum_uncertainty', 'pure_quantum', 'hybrid_uncertainty_quantum', 'optimized_hybrid', 'high_dim_quantum']
        classical_methods = ['random', 'uncertainty', 'badge', 'coreset']
        groups = {'Low Complexity (d<15)': [], 'Medium Complexity (15≤d<40)': [], 'High Complexity (d≥40)': []}
        
        for dataset_name, (X, y) in self.datasets.items():
            d = X.shape[1]
            if d < 15: groups['Low Complexity (d<15)'].append(dataset_name)
            elif d < 40: groups['Medium Complexity (15≤d<40)'].append(dataset_name)
            else: groups['High Complexity (d≥40)'].append(dataset_name)

        group_names, quantum_perf, classical_perf, group_records = list(groups.keys()), [], [], []
        for group_name, group_datasets in groups.items():
            q_scores, c_scores = [], []
            for ds_name in group_datasets:
                for model_results in self.results.get(ds_name, {}).values():
                    for m, res in model_results.items():
                        if m in quantum_methods: q_scores.append(res['mean_accuracy'])
                        elif m in classical_methods: c_scores.append(res['mean_accuracy'])
            q_mean, c_mean = (np.mean(q_scores) if q_scores else np.nan), (np.mean(c_scores) if c_scores else np.nan)
            quantum_perf.append(q_mean)
            classical_perf.append(c_mean)
            group_records.append({"complexity_group": group_name, "n_datasets": len(group_datasets), "quantum_mean": q_mean, "classical_mean": c_mean, "difference": q_mean - c_mean if not (np.isnan(q_mean) or np.isnan(c_mean)) else np.nan})

        x, width = np.arange(len(group_names)), 0.35
        bars_c, bars_q = ax1.bar(x - width/2, classical_perf, width, label='Classical', alpha=0.75, color='lightblue'), ax1.bar(x + width/2, quantum_perf, width, label='Quantum', alpha=0.75, color='lightcoral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(group_names)
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Quantum vs Classical Performance Across Complexity Groups', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        for bars in [bars_c, bars_q]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h): ax1.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

        quantum_advantages, dataset_complexity, dataset_records = [], [], []
        for dataset_name, (X, y) in self.datasets.items():
            d, q_scores, c_scores = X.shape[1], [], []
            for model_results in self.results.get(dataset_name, {}).values():
                for m, res in model_results.items():
                    if m in quantum_methods: q_scores.append(res['mean_accuracy'])
                    elif m in classical_methods: c_scores.append(res['mean_accuracy'])
            if q_scores and c_scores:
                quantum_avg, classical_avg, advantage = np.mean(q_scores), np.mean(c_scores), np.mean(q_scores) - np.mean(c_scores)
                dataset_complexity.append(d)
                quantum_advantages.append(advantage)
                dataset_records.append({"dataset": dataset_name, "n_features": d, "quantum_mean": quantum_avg, "classical_mean": classical_avg, "quantum_advantage": advantage})

        if dataset_complexity:
            ax2.scatter(dataset_complexity, quantum_advantages, alpha=0.75, s=70)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax2.set_xlabel('Feature Dimension d')
            ax2.set_ylabel('Quantum Advantage (Q - C Accuracy)')
            ax2.set_title('Quantum Advantage vs Dataset Complexity', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            if len(dataset_complexity) > 1:
                z, p = np.polyfit(dataset_complexity, quantum_advantages, 1), np.poly1d(np.polyfit(dataset_complexity, quantum_advantages, 1))
                xs = np.linspace(min(dataset_complexity), max(dataset_complexity), 100)
                ax2.plot(xs, p(xs), 'r--', alpha=0.8, label='Trend Line')
                ax2.legend()
                ax2.text(0.98, 0.05, f"slope = {z[0]:.5f}\nn = {len(dataset_complexity)}", transform=ax2.transAxes, ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75, pad=0.35))
        else:
            ax2.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax2.transAxes, fontsize=12)

        save_figure(fig, "quantum_advantage.png")
        save_table(pd.DataFrame(group_records), "quantum_advantage_group_summary.csv")
        save_table(pd.DataFrame(dataset_records), "quantum_advantage_dataset_summary.csv")


    def plot_model_adaptability(self):
        model_cfg = [('logistic', 'Logistic'), ('random_forest', 'RF'), ('svm', 'SVM'), ('xgboost', 'XGBoost'), ('mlp', 'MLP')]
        method_cfg = [('random', 'Random'), ('uncertainty', 'Unc'), ('quantum_entropy', 'Q-Ent'), ('optimized_hybrid', 'Opt-Hybrid'), ('high_dim_quantum', 'HighDim-Q')]
        matrix = np.array([
            [np.mean([ds[m][t]['mean_accuracy'] for ds in self.results.values() if m in ds and t in ds[m]]) if any(m in ds and t in ds[m] for ds in self.results.values()) else np.nan
            for t, _ in method_cfg]
            for m, _ in model_cfg
        ], dtype=float)

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        model_names, method_names = [n for _, n in model_cfg], [n for _, n in method_cfg]
        ax.set_xticks(np.arange(len(method_names))); ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(method_names); ax.set_yticklabels(model_names)

        threshold = np.nanmean(matrix)
        for i in range(len(model_names)):
            for j in range(len(method_names)):
                val = matrix[i, j]
                ax.text(j, i, "-" if np.isnan(val) else f"{val:.3f}",
                        ha="center", va="center",
                        color="black" if np.isnan(val) or val < threshold else "white", fontsize=9)

        ax.set_title('Model-Sampling Strategy Adaptability Heatmap', fontsize=14, fontweight='bold', pad=20)
        fig.colorbar(im, ax=ax, label='Average Accuracy')
        save_figure(fig, "model_adaptability_heatmap.png")
        save_table(pd.DataFrame(matrix, index=model_names, columns=method_names).reset_index().rename(columns={'index': 'model'}),
                "model_adaptability_matrix.csv")

    def save_best_cfgs(self, path: Union[str, Path]=BEST_CFG_PATH):
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "best_cfgs": self.best_cfgs,
                "grid_search_results": self.grid_search_results
            }, f)
        print(f"Saved best cfgs to {path}")


    def load_best_cfgs(self, path: Union[str, Path]=BEST_CFG_PATH):
        path = Path(path)
        if not path.exists():
            print("No saved cfg file found.")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.best_cfgs = data.get("best_cfgs", {})
        self.grid_search_results = data.get("grid_search_results", {})

        print(f"Loaded best cfgs from {path} (total={len(self.best_cfgs)})")
        return True


    def build_all_best_cfgs(
        self,
        target_models=None,
        target_methods=None,
        probe_rounds=8,
        probe_seeds=range(0, 3),
        query_batch=5,
        use_cache=True,
        save_path: Union[str, Path]=BEST_CFG_PATH,
    ):

        if use_cache and self.load_best_cfgs(save_path):
            print("\n" + "=" * 80)
            print("Using cached best cfgs, skipping grid search.")
            print("=" * 80)
            return self.best_cfgs, self.grid_search_results
        
        if not self.datasets:
            self.load_datasets()

        if target_models is None:
            target_models = ["logistic", "random_forest", "svm", "xgboost", "mlp"]

        if target_methods is None:
            target_methods = [
                "clusteraware",
                "entropy_diversity",
                "quantum_entropy",
                "quantum_uncertainty",
                "optimized_hybrid",
                "high_dim_quantum",
            ]

        total_jobs = len(self.datasets) * len(target_models) * len(target_methods)
        job_id = 0

        print("\n" + "=" * 80)
        print("Batch hyperparameter search: starting to build all best cfgs")
        print("=" * 80)
        print(f"Total jobs: {total_jobs}")
        print(f"probe_rounds={probe_rounds}, probe_seeds={list(probe_seeds)}, query_batch={query_batch}")

        for dataset_name, (X, y) in self.datasets.items():
            print(f"\n=== Dataset: {dataset_name} | shape={X.shape} ===")

            for model_name in target_models:
                print(f"  model: {model_name}")

                for strategy in target_methods:
                    job_id += 1
                    key = (dataset_name, model_name, strategy)

                    print(f"    [{job_id}/{total_jobs}] strategy: {strategy}")

                    try:
                        best_cfg, search_results = run_hyperparameter_grid_search(
                            X=X,
                            y=y,
                            model_name=model_name,
                            strategy=strategy,
                            probe_rounds=probe_rounds,
                            probe_seeds=probe_seeds,
                            query_batch=query_batch,
                            use_enhanced_quantum=self.use_enhanced_quantum,
                        )

                        self.best_cfgs[key] = best_cfg
                        self.grid_search_results[key] = search_results

                        if search_results:
                            print(f"      best mean_accuracy = {search_results[0]['mean_accuracy']:.4f}")
                        else:
                            print(f"      best cfg saved")

                    except Exception as e:
                        print(f"      Grid search failed: {e}")
                        self.best_cfgs[key] = HyperparamConfig()
                        self.grid_search_results[key] = []

        print("\n" + "=" * 80)
        print("Batch hyperparameter search completed")
        print(f"Total number of saved best cfgs: {len(self.best_cfgs)}")
        print("=" * 80)

        self.save_best_cfgs(save_path)

        return self.best_cfgs, self.grid_search_results


    def method_needs_cfg(self, method):
        return method in {
            "clusteraware",
            "entropy_diversity",
            "quantum_entropy",
            "quantum_uncertainty",
            "hybrid_uncertainty_quantum",
            "optimized_hybrid",
            "high_dim_quantum",
        }



    def export_raw_experiment_results(self):

        if not self.results:
            print("[Warning] self.results is empty. Run experiments before exporting raw results.")
            return None, None

        if not self.all_acc_histories:
            print("[Warning] self.all_acc_histories is empty. Learning curve export unavailable.")

        seed_records = []
        curve_records = []

        for dataset_name, dataset_results in self.results.items():
            for model_name, method_results in dataset_results.items():
                for method_name, result_info in method_results.items():

                    all_scores = result_info.get("all_scores", [])
                    dataset_complexity = result_info.get("dataset_complexity", np.nan)

                    for seed_idx, final_acc in enumerate(all_scores):
                        seed_records.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "method": method_name,
                            "seed_idx": seed_idx,
                            "final_accuracy": final_acc,
                            "dataset_complexity": dataset_complexity,
                            "mean_accuracy": result_info.get("mean_accuracy", np.nan),
                            "std_accuracy": result_info.get("std_accuracy", np.nan),
                        })

        for dataset_name, dataset_histories in self.all_acc_histories.items():
            for model_name, method_histories in dataset_histories.items():
                for method_name, seed_curves in method_histories.items():

                    label_curves = (
                        self.all_label_histories
                        .get(dataset_name, {})
                        .get(model_name, {})
                        .get(method_name, [])
                    )

                    for seed_idx, acc_curve in enumerate(seed_curves):
                        if seed_idx < len(label_curves):
                            label_curve = label_curves[seed_idx]
                        else:
                            label_curve = [np.nan] * len(acc_curve)

                        for round_idx, acc in enumerate(acc_curve):
                            label_count = (
                                label_curve[round_idx]
                                if round_idx < len(label_curve)
                                else np.nan
                            )

                            curve_records.append({
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method_name,
                                "seed_idx": seed_idx,
                                "round": round_idx,
                                "accuracy": acc,
                                "label_count": label_count,
                            })

        seed_df = pd.DataFrame(seed_records)
        curve_df = pd.DataFrame(curve_records)

        save_table(seed_df, "raw_seed_results.csv")
        save_table(curve_df, "raw_learning_curve_results.csv")

        print("[Saved raw results]")
        print(f"  raw_seed_results.csv: {len(seed_df)} rows")
        print(f"  raw_learning_curve_results.csv: {len(curve_df)} rows")

        return seed_df, curve_df
    def run_complete_analysis(
            self, 
            seeds=range(0, 5), 
            rounds=15, 
            query_batch=5,
            do_grid_search=True,
            probe_rounds=2,
            probe_seeds=range(0, 1),
            use_cache=True,
            save_path: Union[str, Path]=BEST_CFG_PATH,
            ):
        print("=" * 80)
        print("Enhanced Quantum Sampling Comparison Experiment")
        print("=" * 80)
        
        print("Starting full experiment analysis...")
        self.load_datasets()

        if do_grid_search:
            print("\nStarting batch hyperparameter search...")
            self.build_all_best_cfgs(
                probe_rounds=probe_rounds,
                probe_seeds=probe_seeds,
                query_batch=query_batch,
                use_cache=use_cache,
                save_path=save_path,
            )

        print(f"\nRunning experiments... (seeds={list(seeds)}, rounds={rounds}, query_batch={query_batch})")
        self.run_experiments(seeds=seeds, rounds=rounds, query_batch=query_batch)
        
        print("\nGenerating ranking tables...")
        self.generate_ranking_tables()
        print("\nGenerating summary...")
        self.generate_summary()
        print("\nAnalyzing quantum sampling performance...")
        self.analyze_quantum_performance()
        print("\nRunning statistical significance tests...")
        self.statistical_significance_test()
        print("\nAnalyzing label efficiency (N_T metric)...")
        self.analyze_label_efficiency(tau_list=[0.9, 0.95])
        print("\nGenerating visual analysis...")
        self.plot_comprehensive_analysis()

        print("\nExporting raw experiment results...")
        self.export_raw_experiment_results()

        print("\n Enhanced experiment completed!")


def run_hyperparameter_grid_search(
    X, y,
    model_name="logistic",
    strategy="quantum_entropy",
    probe_rounds=8,
    probe_seeds=range(0, 3),
    query_batch=5,
    use_enhanced_quantum=True,
):
    from itertools import product as iproduct
    import numpy as np

    probe_seeds = list(probe_seeds)
    fixed_complexity_cfg = {
        "complexity_thresh_mid": 15,
        "complexity_thresh_high": 40,
    }

    def make_cfg(**kwargs):
        return HyperparamConfig(**fixed_complexity_cfg, **kwargs)

    def evaluate_cfg(cfg):
        scores = []
        for seed in probe_seeds:
            try:
                result = run_active_learning_experiment(
                    X, y, model_name, strategy,
                    rounds=probe_rounds,
                    query_batch=query_batch,
                    seed=seed,
                    use_enhanced_quantum=use_enhanced_quantum,
                    cfg=cfg,
                )
                scores.append(result["final_accuracy"])
            except Exception as e:
                print(f"    [WARN] cfg failed on seed={seed}: {e}")
                traceback.print_exc()
                scores.append(0.5)

        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.5
        return mean_score, scores

    if strategy == "quantum_entropy":
        print(f"\n{'='*70}")
        print(f"[Direct Grid Search] strategy={strategy}, model={model_name}")
        print(f"  probe_rounds: {probe_rounds}  |  seeds: {probe_seeds}")
        print(f"{'='*70}")

        qd_pairs = [
            (0.5, 0.5),
            (0.6, 0.4),
            (0.7, 0.3),
            (0.8, 0.2),
        ]

        search_cfgs = [
            make_cfg(
                w_qd_q=wq,
                w_qd_d=wd,
            )
            for wq, wd in qd_pairs
        ]

        print(f"  qd_pairs: {qd_pairs}")
        print(f"  Number of configs: {len(search_cfgs)}")

        best_score = -1.0
        best_cfg = None
        search_results = []

        for i, cfg in enumerate(search_cfgs, 1):
            mean_score, scores = evaluate_cfg(cfg)

            record = {
                "stage": "qd_pairs",
                "cfg": cfg,
                "mean_accuracy": mean_score,
                "scores": scores,
            }
            search_results.append(record)

            marker = ""
            if mean_score > best_score:
                best_score = mean_score
                best_cfg = cfg
                marker = " <-- best"

            print(f"  [{i:2d}/{len(search_cfgs)}] {cfg} -> {mean_score:.4f}{marker}")

        if best_cfg is None:
            print("\n[WARN] All qd-pair configs failed. Falling back to default qd config.")
            best_cfg = make_cfg(w_qd_q=0.7, w_qd_d=0.3)
            best_score = 0.5
            search_results.append({
                "stage": "fallback_default",
                "cfg": best_cfg,
                "mean_accuracy": best_score,
                "scores": [0.5] * len(probe_seeds),
            })

        search_results.sort(key=lambda r: r["mean_accuracy"], reverse=True)

        print(f"\n[Direct Grid Search] Final best cfg (score={best_score:.4f}):")
        print(f"  {best_cfg}")

        print(f"\nTop-5 configs:")
        for rank, r in enumerate(search_results[:5], 1):
            print(f"  #{rank}  {r['mean_accuracy']:.4f}  [{r['stage']}]  {r['cfg']}")

        return best_cfg, search_results

    elif strategy == "quantum_uncertainty":
        print(f"\n{'='*70}")
        print(f"[4-State Scheduler Search] strategy={strategy}, model={model_name}")
        print(f"  probe_rounds: {probe_rounds}  |  seeds: {probe_seeds}")
        print(f"{'='*70}")

        scheduler_states = [
            (
                "default",
                make_cfg(
                    decay_high=0.20,
                    decay_mid=0.25,
                    decay_low=0.30,
                    wc_time_scale=0.30,
                    wd_time_scale=0.20,
                ),
            ),
            (
                "quantum_preserving",
                make_cfg(
                    decay_high=0.15,
                    decay_mid=0.20,
                    decay_low=0.25,
                    wc_time_scale=0.20,
                    wd_time_scale=0.15,
                ),
            ),
            (
                "balanced",
                make_cfg(
                    decay_high=0.20,
                    decay_mid=0.25,
                    decay_low=0.30,
                    wc_time_scale=0.35,
                    wd_time_scale=0.25,
                ),
            ),
            (
                "classical_reinforced",
                make_cfg(
                    decay_high=0.25,
                    decay_mid=0.30,
                    decay_low=0.35,
                    wc_time_scale=0.40,
                    wd_time_scale=0.30,
                ),
            ),
        ]

        print(f"  Scheduler states: {[name for name, _ in scheduler_states]}")
        print(f"  Number of configs: {len(scheduler_states)}")

        search_results = []
        best_score = -1.0
        best_cfg = None

        for i, (state_name, cfg) in enumerate(scheduler_states, 1):
            mean_score, scores = evaluate_cfg(cfg)

            record = {
                "stage": state_name,
                "cfg": cfg,
                "mean_accuracy": mean_score,
                "scores": scores,
            }
            search_results.append(record)

            marker = ""
            if mean_score > best_score:
                best_score = mean_score
                best_cfg = cfg
                marker = " <-- best"

            print(f"  [{i:2d}/{len(scheduler_states)}] [{state_name}] {cfg} -> {mean_score:.4f}{marker}")

        if best_cfg is None:
            print("\n[WARN] All scheduler-state configs failed. Falling back to default scheduler config.")
            best_cfg = make_cfg(
                decay_high=0.20,
                decay_mid=0.25,
                decay_low=0.30,
                wc_time_scale=0.30,
                wd_time_scale=0.20,
            )
            best_score = 0.5
            search_results.append({
                "stage": "fallback_default",
                "cfg": best_cfg,
                "mean_accuracy": best_score,
                "scores": [0.5] * len(probe_seeds),
            })

        search_results.sort(key=lambda r: r["mean_accuracy"], reverse=True)

        print(f"\n[4-State Scheduler Search] Final best cfg (score={best_score:.4f}):")
        print(f"  {best_cfg}")

        print(f"\nTop-5 configs:")
        for rank, r in enumerate(search_results[:5], 1):
            print(f"  #{rank}  {r['mean_accuracy']:.4f}  [{r['stage']}]  {r['cfg']}")

        return best_cfg, search_results

    elif strategy == "pure_quantum":
        print(f"\n{'='*70}")
        print(f"[Fixed Config Evaluation] strategy={strategy}, model={model_name}")
        print(f"  probe_rounds: {probe_rounds}  |  seeds: {probe_seeds}")
        print(f"{'='*70}")

        fixed_cfg = make_cfg()
        mean_score, scores = evaluate_cfg(fixed_cfg)
        search_results = [{
            "stage": "fixed_default",
            "cfg": fixed_cfg,
            "mean_accuracy": mean_score,
            "scores": scores,
        }]

        print(f"  Fixed cfg -> {mean_score:.4f}")
        print(f"  {fixed_cfg}")
        return fixed_cfg, search_results

    elif strategy == "hybrid_uncertainty_quantum":
        print(f"\n{'='*70}")
        print(f"[Fixed Config Evaluation] strategy={strategy}, model={model_name}")
        print(f"  probe_rounds: {probe_rounds}  |  seeds: {probe_seeds}")
        print(f"{'='*70}")

        fixed_cfg = make_cfg(w_qd_q=0.7, w_qd_d=0.3)
        mean_score, scores = evaluate_cfg(fixed_cfg)
        search_results = [{
            "stage": "fixed_qent_shared_weights",
            "cfg": fixed_cfg,
            "mean_accuracy": mean_score,
            "scores": scores,
        }]

        print("  Shared with Q-Ent qd weights: (0.7, 0.3)")
        print(f"  Fixed cfg -> {mean_score:.4f}")
        print(f"  {fixed_cfg}")
        return fixed_cfg, search_results

    elif strategy == "clusteraware":
        search_cfgs = [
            make_cfg(w_ud_u=wu, w_ud_d=wd)
            for wu, wd in [
                (0.3, 0.7),
                (0.4, 0.6),
                (0.5, 0.5),
                (0.7, 0.3),
            ]
        ]

    elif strategy == "entropy_diversity":
        search_cfgs = [
            make_cfg(w_ed_e=we, w_ed_d=wd)
            for we, wd in [
                (0.5, 0.5),
                (0.6, 0.4),
                (0.7, 0.3),
                (0.8, 0.2),
            ]
        ]

    elif strategy == "high_dim_quantum":
        search_cfgs = [
            make_cfg(w_hd_q=wq, w_hd_c=wc, w_hd_d=wd)
            for wq, wc, wd in [
                (0.4, 0.3, 0.3),
                (0.5, 0.3, 0.2),
                (0.3, 0.4, 0.3),
                (0.6, 0.2, 0.2),
            ]
        ]

    elif strategy == "optimized_hybrid":
        search_cfgs = [
            make_cfg(
                early_weight=ew,
                late_weight=lw,
                quantum_dom_threshold=qdt,
                balanced_threshold=bt
            )
            for ew, lw in [
                (0.7, 0.3),
                (0.8, 0.2),
                (0.9, 0.1),
            ]
            for qdt, bt in [
               (0.65, 0.35),
               (0.75, 0.25),
               (0.85, 0.15),
            ]
        ]

    else:
        search_cfgs = [
            make_cfg(early_weight=ew, late_weight=lw)
            for ew, lw in iproduct([0.7, 0.8, 0.9], [0.2, 0.3, 0.4])
        ]

    n_cfgs = len(search_cfgs)
    print(f"\n{'='*70}")
    print(f"[Grid Search] strategy={strategy}, model={model_name}")
    print(f"  Total configurations: {n_cfgs}  |  probe_rounds: {probe_rounds}  |  seeds: {probe_seeds}")
    print(f"{'='*70}")

    search_results = []
    best_score = -1.0
    best_cfg = make_cfg()

    for i, cfg in enumerate(search_cfgs, 1):
        scores = []
        for seed in probe_seeds:
            try:
                result = run_active_learning_experiment(
                    X, y, model_name, strategy,
                    rounds=probe_rounds,
                    query_batch=query_batch,
                    seed=seed,
                    use_enhanced_quantum=use_enhanced_quantum,
                    cfg=cfg,
                )
                scores.append(result["final_accuracy"])
            except Exception as e:
                print(f"    [WARN] cfg failed on seed={seed}: {e}")
                traceback.print_exc()
                scores.append(0.5)

        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.5
        search_results.append({
            "cfg": cfg,
            "mean_accuracy": mean_score,
            "scores": scores
        })

        marker = " <-- best" if mean_score > best_score else ""
        print(f"  [{i:3d}/{n_cfgs}] {cfg}  ->  {mean_score:.4f}{marker}")

        if mean_score > best_score:
            best_score = mean_score
            best_cfg = cfg

    search_results.sort(key=lambda r: r["mean_accuracy"], reverse=True)

    print(f"\n[Grid Search] Best configuration (score={best_score:.4f}):")
    print(f"  {best_cfg}")
    print(f"\nTop-5 configurations:")
    for rank, r in enumerate(search_results[:5], 1):
        print(f"  #{rank}  {r['mean_accuracy']:.4f}  {r['cfg']}")

    return best_cfg, search_results


if __name__ == "__main__":
    print("Enhanced Quantum Sampling Comparison Experiment - Complete Real Dataset Analysis")
    print("=" * 80)

    experiment = EnhancedQuantumSamplingComparison(use_enhanced_quantum=True)

    experiment.run_complete_analysis(
        seeds=range(10, 15),
        rounds=15,
        query_batch=5,
        do_grid_search=True,
        probe_rounds=3,
        probe_seeds=range(0, 2),
        use_cache=False,
        save_path=BEST_CFG_PATH,
    )
