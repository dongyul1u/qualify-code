from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


class ResultAnalysis:
    def __init__(self, raw_seed_path, raw_curve_path, output_dir, table_dir):
        self.raw_seed_path = Path(raw_seed_path)
        self.raw_curve_path = Path(raw_curve_path)
        self.output_dir = Path(output_dir)
        self.table_dir = Path(table_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

        self.method_label_full = {
            "random": "Random",
            "uncertainty": "Uncertainty",
            "coreset": "CoreSet",
            "badge": "Approx. BADGE",
            "clusteraware": "Uncertainty--Diversity",
            "entropy_diversity": "Entropy--Diversity",
            "quantum_entropy": "Q-Ent",
            "quantum_uncertainty": "Q+Unc",
            "pure_quantum": "Q-Only",
            "hybrid_uncertainty_quantum": "Hybrid-Q",
            "optimized_hybrid": "Opt-Hybrid",
            "high_dim_quantum": "HighDim-Q",
        }
        self.method_label_short = {
            "random": "Random",
            "uncertainty": "Unc",
            "coreset": "CoreSet",
            "badge": "BADGE",
            "clusteraware": "Unc+Div",
            "entropy_diversity": "Ent+Div",
            "quantum_entropy": "Q-Ent",
            "quantum_uncertainty": "Q+Unc",
            "pure_quantum": "Q-Only",
            "hybrid_uncertainty_quantum": "Hybrid-Q",
            "optimized_hybrid": "Opt-Hybrid",
            "high_dim_quantum": "HighDim-Q",
        }
        self.method_cfg = list(self.method_label_full.items())
        self.method_order = list(self.method_label_full.keys())
        self.quantum_methods = {
            "quantum_entropy",
            "quantum_uncertainty",
            "pure_quantum",
            "hybrid_uncertainty_quantum",
            "optimized_hybrid",
            "high_dim_quantum",
        }
        self.classical_methods = {
            "random",
            "uncertainty",
            "coreset",
            "badge",
            "clusteraware",
            "entropy_diversity",
        }
        self.model_label_map = {
            "logistic": "Logistic",
            "random_forest": "RF",
            "svm": "SVM",
            "xgboost": "XGBoost",
            "mlp": "MLP",
        }
        self.df = self.load_raw_seed_results()
        self.curve_df = self.load_raw_learning_curve_results()

    def load_raw_seed_results(self):
        df = pd.read_csv(self.raw_seed_path)
        required = {
            "dataset",
            "model",
            "method",
            "seed_idx",
            "final_accuracy",
            "dataset_complexity",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in raw seed file: {sorted(missing)}")
        df = df.copy()
        df["final_accuracy"] = pd.to_numeric(df["final_accuracy"], errors="coerce")
        df = df.dropna(subset=["final_accuracy"])
        return df

    def load_raw_learning_curve_results(self):
        df = pd.read_csv(self.raw_curve_path)
        required = {
            "dataset",
            "model",
            "method",
            "seed_idx",
            "round",
            "accuracy",
            "label_count",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in raw curve file: {sorted(missing)}")
        df = df.copy()
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df["label_count"] = pd.to_numeric(df["label_count"], errors="coerce")
        df = df.dropna(subset=["accuracy"])
        return df

    def save_table(self, df, file_name):
        outpath = self.table_dir / file_name
        df.to_csv(outpath, index=False)
        print(f"[Saved table] {outpath}")

    def save_figure(self, fig, fig_name, dpi=300):
        outpath = self.output_dir / fig_name
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved figure] {outpath}")

    def add_bar_labels(self, ax, bars, values, fontsize=8):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    def complexity_level(self, d):
        if d < 15:
            return "low"
        if d < 40:
            return "medium"
        return "high"

    def complexity_label(self, level):
        return {
            "low": "Low (d < 15)",
            "medium": "Medium (15 <= d < 40)",
            "high": "High (d >= 40)",
        }[level]

    def method_summary(self):
        summary = (
            self.df.groupby("method", as_index=False)["final_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["group"] = summary["method"].map(
            lambda m: "Quantum" if m in self.quantum_methods else "Classical"
        )
        summary["method_label"] = summary["method"].map(self.method_label_full)
        summary["method_order"] = summary["method"].map(
            {m: i for i, m in enumerate(self.method_order)}
        )
        summary = summary.sort_values("method_order").reset_index(drop=True)
        self.save_table(summary, "method_summary_from_raw_seed.csv")
        return summary

    def model_summary(self):
        summary = (
            self.df.groupby("model", as_index=False)["final_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["model_label"] = summary["model"].map(self.model_label_map)
        summary["model_order"] = summary["model"].map(
            {m: i for i, m in enumerate(self.model_label_map)}
        )
        summary = summary.sort_values("model_order").reset_index(drop=True)
        self.save_table(summary, "model_summary_from_raw_seed.csv")
        return summary

    def quantum_classical_summary(self):
        df = self.df.copy()
        df["group"] = df["method"].map(
            lambda m: "Quantum" if m in self.quantum_methods else "Classical"
        )
        summary = (
            df.groupby("group", as_index=False)["final_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        self.save_table(summary, "quantum_classical_summary_from_raw_seed.csv")
        return summary

    def build_task_level_final_performance_table(self):
        df = self.df[self.df["method"].isin(self.method_order)].copy()
        task_level = (
            df.groupby(["dataset", "model", "method"], as_index=False)
            .agg(task_mean_accuracy=("final_accuracy", "mean"))
        )
        task_level["group"] = task_level["method"].map(
            lambda m: "Quantum-enh." if m in self.quantum_methods else "Classical"
        )
        task_level["task_rank"] = task_level.groupby(
            ["dataset", "model"]
        )["task_mean_accuracy"].rank(ascending=False, method="average")
        task_max = task_level.groupby(["dataset", "model"])[
            "task_mean_accuracy"
        ].transform("max")
        task_level["top_task_hit"] = (
            np.abs(task_level["task_mean_accuracy"] - task_max) <= 1e-12
        ).astype(int)

        summary = (
            task_level.groupby(["method", "group"], as_index=False)
            .agg(
                mean_acc=("task_mean_accuracy", "mean"),
                task_sd=("task_mean_accuracy", "std"),
                mean_rank=("task_rank", "mean"),
                top_tasks=("top_task_hit", "sum"),
                n_tasks=("task_mean_accuracy", "count"),
            )
        )
        summary["task_sd"] = summary["task_sd"].fillna(0.0)
        summary["strategy"] = summary["method"].map(self.method_label_full)
        summary["method_order"] = summary["method"].map(
            {m: i for i, m in enumerate(self.method_order)}
        )
        summary = summary.sort_values(
            ["mean_rank", "mean_acc", "method_order"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        summary = summary[
            ["strategy", "group", "mean_acc", "task_sd", "mean_rank", "top_tasks", "n_tasks"]
        ]
        self.save_table(summary, "task_level_final_performance_summary.csv")
        self.save_table(task_level, "task_level_seed_averaged_scores.csv")
        return summary, task_level

    def build_method_dataset_mean_table(self):
        df = self.df[self.df["method"].isin(self.method_order)].copy()
        dataset_meta = (
            df.groupby("dataset", as_index=False)
            .agg(dataset_complexity=("dataset_complexity", "first"))
        )
        dataset_meta["complexity_level"] = dataset_meta["dataset_complexity"].map(
            self.complexity_level
        )
        dataset_meta["complexity_order"] = dataset_meta["complexity_level"].map(
            {"low": 0, "medium": 1, "high": 2}
        )
        dataset_meta = dataset_meta.sort_values(
            ["complexity_order", "dataset_complexity", "dataset"]
        ).reset_index(drop=True)
        dataset_order = dataset_meta["dataset"].tolist()

        matrix = (
            df.groupby(["dataset", "method"], as_index=False)
            .agg(mean_accuracy=("final_accuracy", "mean"))
            .pivot(index="method", columns="dataset", values="mean_accuracy")
            .reindex(columns=dataset_order)
        )
        low_cols = dataset_meta.loc[dataset_meta["complexity_level"] == "low", "dataset"]
        med_cols = dataset_meta.loc[dataset_meta["complexity_level"] == "medium", "dataset"]
        high_cols = dataset_meta.loc[dataset_meta["complexity_level"] == "high", "dataset"]
        matrix["low_mean"] = matrix[list(low_cols)].mean(axis=1) if len(low_cols) else np.nan
        matrix["medium_mean"] = matrix[list(med_cols)].mean(axis=1) if len(med_cols) else np.nan
        matrix["high_mean"] = matrix[list(high_cols)].mean(axis=1) if len(high_cols) else np.nan
        matrix["overall_mean"] = matrix[dataset_order].mean(axis=1)
        matrix = matrix.sort_values("overall_mean", ascending=False).reset_index()
        matrix["strategy"] = matrix["method"].map(self.method_label_full)
        cols = ["strategy", "low_mean", "medium_mean", "high_mean", "overall_mean"] + dataset_order
        matrix = matrix[["method"] + cols]
        self.save_table(matrix, "method_dataset_mean_table.csv")
        return matrix

    def export_strategy_dataset_metric_table(
        self,
        metric_col="final_accuracy",
        agg_name=None,
        save_prefix=None,
        include_values=True,
    ):
        df = self.df.copy()
        if metric_col not in df.columns:
            raise ValueError(
                f"Metric column '{metric_col}' not found. "
                f"Available columns: {sorted(df.columns)}"
            )

        df = df[df["method"].isin({m for m, _ in self.method_cfg})].copy()
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        df = df.dropna(subset=[metric_col]).copy()
        if df.empty:
            raise ValueError(f"No valid rows remain after filtering metric '{metric_col}'.")

        if agg_name is None:
            agg_name = metric_col.replace("final_", "").replace("_", " ").title()
        if save_prefix is None:
            save_prefix = metric_col.lower()

        dataset_meta = (
            df.groupby("dataset", as_index=False)
            .agg(dataset_complexity=("dataset_complexity", "first"))
        )
        dataset_meta["complexity_level"] = dataset_meta["dataset_complexity"].map(
            self.complexity_level
        )
        dataset_meta["complexity_order"] = dataset_meta["complexity_level"].map(
            {"low": 0, "medium": 1, "high": 2}
        )
        dataset_meta = dataset_meta.sort_values(
            ["complexity_order", "dataset_complexity", "dataset"]
        ).reset_index(drop=True)
        dataset_order = dataset_meta["dataset"].tolist()

        grouped = (
            df.groupby(["dataset", "method"], as_index=False)
            .agg(
                mean_metric=(metric_col, "mean"),
                std_metric=(metric_col, "std"),
                n=(metric_col, "count"),
            )
        )

        if include_values:
            values_df = (
                df.groupby(["dataset", "method"])[metric_col]
                .apply(lambda s: [round(float(x), 6) for x in s.tolist()])
                .reset_index(name="metric_values")
            )
            grouped = grouped.merge(values_df, on=["dataset", "method"], how="left")

        grouped = grouped.merge(
            dataset_meta[["dataset", "dataset_complexity", "complexity_level", "complexity_order"]],
            on="dataset",
            how="left",
        )
        grouped["strategy"] = grouped["method"].map(self.method_label_full)
        grouped["std_metric"] = grouped["std_metric"].fillna(0.0)

        strategy_order = (
            grouped.groupby("method", as_index=False)["mean_metric"]
            .mean()
            .sort_values("mean_metric", ascending=False)["method"]
            .tolist()
        )
        strategy_rank = {m: i for i, m in enumerate(strategy_order)}
        grouped["strategy_order"] = grouped["method"].map(strategy_rank)
        grouped = grouped.sort_values(
            ["strategy_order", "complexity_order", "dataset_complexity", "dataset"]
        ).reset_index(drop=True)

        long_cols = [
            "strategy",
            "dataset",
            "dataset_complexity",
            "complexity_level",
            "mean_metric",
            "std_metric",
            "n",
        ]
        if include_values:
            long_cols.append("metric_values")

        long_table = grouped[long_cols].copy()

        mean_matrix = (
            grouped.pivot(index="strategy", columns="dataset", values="mean_metric")
            .reindex(
                index=[self.method_label_full[m] for m in strategy_order],
                columns=dataset_order,
            )
        )
        mean_matrix["overall_mean"] = mean_matrix[dataset_order].mean(axis=1)
        mean_matrix = mean_matrix.sort_values("overall_mean", ascending=False).reset_index()

        self.save_table(long_table, f"{save_prefix}_strategy_dataset_long.csv")
        self.save_table(mean_matrix, f"{save_prefix}_strategy_dataset_matrix.csv")

        printable = long_table.copy()
        printable["mean_metric"] = printable["mean_metric"].map(lambda x: f"{x:.4f}")
        printable["std_metric"] = printable["std_metric"].map(lambda x: f"{x:.4f}")

        print(f"\n[{agg_name} by Strategy and Dataset]")
        print(printable.to_string(index=False))

        printable_matrix = mean_matrix.copy()
        for col in printable_matrix.columns:
            if col != "strategy":
                printable_matrix[col] = printable_matrix[col].map(
                    lambda x: "-" if pd.isna(x) else f"{x:.4f}"
                )

        print(f"\n[{agg_name} Mean Matrix]")
        print(printable_matrix.to_string(index=False))

        return long_table, mean_matrix

    def build_imbalance_aware_stress_test_table(
        self,
        datasets=("adult_income", "bank_marketing"),
        methods=(
            "uncertainty",
            "entropy_diversity",
            "quantum_entropy",
            "quantum_uncertainty",
        ),
    ):
        metric_cols = [
            "final_accuracy",
            "final_f1",
            "final_balanced_accuracy",
            "final_recall",
        ]
        missing = [col for col in metric_cols if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing metric columns for imbalance table: {sorted(missing)}"
            )

        df = self.df.copy()
        df = df[df["dataset"].isin(datasets) & df["method"].isin(methods)].copy()
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=metric_cols).copy()
        if df.empty:
            raise ValueError("No valid rows remain for imbalance-aware stress test.")

        summary = (
            df.groupby(["dataset", "method"], as_index=False)[metric_cols]
            .mean()
        )
        summary["dataset_order"] = summary["dataset"].map(
            {dataset: i for i, dataset in enumerate(datasets)}
        )
        summary["method_order"] = summary["method"].map(
            {method: i for i, method in enumerate(methods)}
        )
        summary["method_label"] = summary["method"].map(self.method_label_full)
        summary = summary.sort_values(
            ["dataset_order", "method_order"]
        ).reset_index(drop=True)

        table = summary.rename(
            columns={
                "dataset": "Dataset",
                "method_label": "Method",
                "final_accuracy": "Acc.",
                "final_f1": "F1",
                "final_balanced_accuracy": "Bal. Acc.",
                "final_recall": "Recall",
            }
        )[
            ["Dataset", "Method", "Acc.", "F1", "Bal. Acc.", "Recall"]
        ]

        self.save_table(table, "imbalance_aware_stress_test_table.csv")

        printable = table.copy()
        for col in ["Acc.", "F1", "Bal. Acc.", "Recall"]:
            printable[col] = printable[col].map(lambda x: f"{x:.4f}")

        print("\n[Imbalance-Aware Stress Test Table]")
        print(printable.to_string(index=False))

        return table

    def plot_average_performance_by_method(self):
        summary = self.method_summary()
        colors = [
            "lightcoral" if g == "Quantum" else "lightgray"
            for g in summary["group"]
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(summary))
        bars = ax.bar(x, summary["mean"], color=colors, alpha=0.85)
        self.add_bar_labels(ax, bars, summary["mean"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            summary["method"].map(lambda m: self.method_label_short.get(m, m)),
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("Final Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        self.save_figure(fig, "average_performance_by_method_from_raw_seed.png")

    def plot_metric_by_method(
        self,
        metric_col="final_recall",
        metric_label="Recall",
        fig_name="average_recall_by_method_from_raw_seed.png",
    ):
        if metric_col not in self.df.columns:
            raise ValueError(
                f"Metric column '{metric_col}' not found. "
                f"Available columns: {sorted(self.df.columns)}"
            )

        df = self.df.copy()
        df = df[df["method"].isin({m for m, _ in self.method_cfg})].copy()
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        df = df.dropna(subset=[metric_col]).copy()
        if df.empty:
            raise ValueError(f"No valid rows remain for metric '{metric_col}'.")

        summary = (
            df.groupby("method")[metric_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        method_order = {m: i for i, (m, _) in enumerate(self.method_cfg)}
        summary["group"] = summary["method"].map(
            lambda m: "Quantum" if m in self.quantum_methods else "Classical"
        )
        summary["method_order"] = summary["method"].map(method_order)
        summary = summary.sort_values("method_order").reset_index(drop=True)
        self.save_table(summary, f"{metric_col}_method_summary.csv")

        colors = [
            "lightcoral" if g == "Quantum" else "lightgray"
            for g in summary["group"]
        ]

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(summary))
        bars = ax.bar(x, summary["mean"], color=colors, alpha=0.85)
        self.add_bar_labels(ax, bars, summary["mean"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            summary["method"].map(lambda m: self.method_label_short.get(m, m)),
            rotation=35,
            ha="right",
        )
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        self.save_figure(fig, fig_name)
        return summary

    def plot_f1_by_method(self):
        return self.plot_metric_by_method(
            metric_col="final_f1",
            metric_label="F1",
            fig_name="average_f1_by_method_from_raw_seed.png",
        )

    def plot_balanced_accuracy_by_method(self):
        return self.plot_metric_by_method(
            metric_col="final_balanced_accuracy",
            metric_label="Balanced Accuracy",
            fig_name="average_balanced_accuracy_by_method_from_raw_seed.png",
        )

    def plot_recall_by_method(self):
        return self.plot_metric_by_method(
            metric_col="final_recall",
            metric_label="Recall",
            fig_name="average_recall_by_method_from_raw_seed.png",
        )

    def plot_distribution_by_method(self):
        data = []
        labels = []
        colors = []
        for method in self.method_order:
            vals = self.df.loc[self.df["method"] == method, "final_accuracy"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(self.method_label_short.get(method, method))
            colors.append("lightcoral" if method in self.quantum_methods else "lightgray")
        fig, ax = plt.subplots(figsize=(11, 5))
        box = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Final Accuracy")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        self.save_figure(fig, "performance_distribution_by_method_from_raw_seed.png")

    def plot_quantum_vs_classical(self):
        summary = self.quantum_classical_summary()
        summary["order"] = summary["group"].map({"Quantum": 0, "Classical": 1})
        summary = summary.sort_values("order")
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(
            summary["group"],
            summary["mean"],
            yerr=summary["std"],
            capsize=5,
            alpha=0.85,
            color=["lightcoral", "lightblue"],
        )
        self.add_bar_labels(ax, bars, summary["mean"], fontsize=9)
        ax.set_ylabel("Final Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        self.save_figure(fig, "quantum_vs_classical_from_raw_seed.png")

    def plot_average_performance_by_model(self):
        summary = self.model_summary()
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(summary))
        bars = ax.bar(
            x,
            summary["mean"],
            yerr=summary["std"],
            capsize=4,
            alpha=0.85,
        )
        self.add_bar_labels(ax, bars, summary["mean"], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["model_label"], rotation=20, ha="right")
        ax.set_ylabel("Final Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        self.save_figure(fig, "average_performance_by_model_from_raw_seed.png")

    def statistical_significance_analysis(self):
        q = self.df[self.df["method"].isin(self.quantum_methods)]["final_accuracy"].to_numpy(dtype=float)
        c = self.df[self.df["method"].isin(self.classical_methods)]["final_accuracy"].to_numpy(dtype=float)
        q_mean, c_mean = np.mean(q), np.mean(c)
        q_std = np.std(q, ddof=1)
        c_std = np.std(c, ddof=1)
        u_stat, p_value = mannwhitneyu(q, c, alternative="two-sided")
        pooled_std = np.sqrt(
            (((len(q) - 1) * q_std**2) + ((len(c) - 1) * c_std**2))
            / (len(q) + len(c) - 2)
        )
        summary = pd.DataFrame(
            [{
                "n_quantum": len(q),
                "n_classical": len(c),
                "quantum_mean": q_mean,
                "classical_mean": c_mean,
                "mean_difference": q_mean - c_mean,
                "quantum_std": q_std,
                "classical_std": c_std,
                "u_statistic": u_stat,
                "p_value": p_value,
                "cohens_d": (q_mean - c_mean) / pooled_std if pooled_std > 0 else np.nan,
            }]
        )
        self.save_table(summary, "statistical_significance_summary.csv")
        return summary

    def run_wilcoxon_test_from_saved_results(self):
        df = self.df[self.df["method"].isin(self.quantum_methods | self.classical_methods)].copy()
        df["group"] = df["method"].map(
            lambda m: "quantum" if m in self.quantum_methods else "classical"
        )
        df["complexity_level"] = df["dataset_complexity"].map(self.complexity_level)
        block_df = (
            df.groupby(
                ["dataset", "model", "seed_idx", "dataset_complexity", "complexity_level", "group"],
                as_index=False,
            )["final_accuracy"]
            .mean()
        )
        paired = (
            block_df.pivot(
                index=["dataset", "model", "seed_idx", "dataset_complexity", "complexity_level"],
                columns="group",
                values="final_accuracy",
            )
            .reset_index()
        )
        paired = paired.dropna(subset=["quantum", "classical"]).copy()
        paired["difference"] = paired["quantum"] - paired["classical"]
        paired["winner"] = np.where(
            paired["difference"] > 1e-12,
            "quantum",
            np.where(paired["difference"] < -1e-12, "classical", "tie"),
        )
        self.save_table(paired, "wilcoxon_quantum_vs_classical_pairs.csv")

        records = []
        for level in ["low", "medium", "high"]:
            level_df = paired[paired["complexity_level"] == level].copy()
            if level_df.empty:
                records.append({
                    "complexity_level": level,
                    "complexity_label": self.complexity_label(level),
                    "n_blocks": 0,
                    "n_nonzero_blocks": 0,
                    "quantum_mean": np.nan,
                    "classical_mean": np.nan,
                    "mean_difference": np.nan,
                    "median_difference": np.nan,
                    "wins_quantum": 0,
                    "wins_classical": 0,
                    "ties": 0,
                    "wilcoxon_statistic": np.nan,
                    "p_value": np.nan,
                    "significant_at_0_05": False,
                })
                continue
            x = level_df["quantum"].to_numpy(dtype=float)
            y = level_df["classical"].to_numpy(dtype=float)
            diff = level_df["difference"].to_numpy(dtype=float)
            nonzero = diff[np.abs(diff) > 1e-12]
            if len(nonzero) == 0:
                stat, p_value = 0.0, 1.0
            else:
                stat, p_value = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            records.append({
                "complexity_level": level,
                "complexity_label": self.complexity_label(level),
                "n_blocks": len(level_df),
                "n_nonzero_blocks": len(nonzero),
                "quantum_mean": float(np.mean(x)),
                "classical_mean": float(np.mean(y)),
                "mean_difference": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                "wins_quantum": int(np.sum(diff > 1e-12)),
                "wins_classical": int(np.sum(diff < -1e-12)),
                "ties": int(np.sum(np.abs(diff) <= 1e-12)),
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "significant_at_0_05": bool(p_value < 0.05),
            })
        summary_df = pd.DataFrame(records)
        self.save_table(summary_df, "wilcoxon_quantum_vs_classical_summary.csv")
        self.plot_complexity_group_mean_difference(summary_df)
        return {"paired_df": paired, "summary_df": summary_df}

    def plot_complexity_group_mean_difference(self, summary_df):
        plot_df = summary_df.copy()
        plot_df["complexity_order"] = plot_df["complexity_level"].map(
            {"low": 0, "medium": 1, "high": 2}
        )
        plot_df = plot_df.sort_values("complexity_order").dropna(
            subset=["quantum_mean", "classical_mean"]
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(plot_df))
        width = 0.36
        bars_q = ax.bar(
            x - width / 2,
            plot_df["quantum_mean"],
            width,
            color="lightcoral",
            alpha=0.88,
            label="Quantum mean",
        )
        bars_c = ax.bar(
            x + width / 2,
            plot_df["classical_mean"],
            width,
            color="lightblue",
            alpha=0.88,
            label="Classical mean",
        )
        for bars in [bars_q, bars_c]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["complexity_label"])
        ax.set_ylabel("Mean Final Accuracy")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        self.save_figure(fig, "wilcoxon_quantum_vs_classical_by_complexity.png")

    def calculate_label_efficiency_all_methods(self, tau_list=(0.90, 0.95)):
        raw_records = []
        for tau in tau_list:
            for (dataset, model), group_df in self.curve_df.groupby(["dataset", "model"]):
                best_acc = group_df["accuracy"].max()
                if pd.isna(best_acc) or best_acc <= 0:
                    continue
                threshold = tau * best_acc
                for method in self.method_order:
                    method_df = group_df[group_df["method"] == method]
                    if method_df.empty:
                        continue
                    for seed_idx, seed_df in method_df.groupby("seed_idx"):
                        seed_df = seed_df.sort_values("round")
                        reached_df = seed_df[seed_df["accuracy"] >= threshold]
                        reached = not reached_df.empty
                        n_tau = reached_df.iloc[0]["label_count"] if reached else np.nan
                        reached_round = reached_df.iloc[0]["round"] if reached else np.nan
                        raw_records.append({
                            "tau": tau,
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "method_label": self.method_label_full.get(method, method),
                            "seed_idx": seed_idx,
                            "best_acc_dataset_model": best_acc,
                            "threshold": threshold,
                            "reached_threshold": reached,
                            "N_tau": n_tau,
                            "reached_round": reached_round,
                        })
        raw_df = pd.DataFrame(raw_records)
        summary_records = []
        for (tau, method), method_df in raw_df.groupby(["tau", "method"]):
            reached_df = method_df[method_df["reached_threshold"] == True]
            n_total = len(method_df)
            n_reached = len(reached_df)
            summary_records.append({
                "tau": tau,
                "method": method,
                "method_label": self.method_label_full.get(method, method),
                "group": "Quantum" if method in self.quantum_methods else "Classical",
                "mean_N_tau": reached_df["N_tau"].mean() if n_reached else np.nan,
                "median_N_tau": reached_df["N_tau"].median() if n_reached else np.nan,
                "std_N_tau": reached_df["N_tau"].std() if n_reached else np.nan,
                "n_reached": n_reached,
                "n_total": n_total,
                "reach_rate": (n_reached / n_total) if n_total else np.nan,
            })
        summary_df = pd.DataFrame(summary_records)
        summary_df["method_order"] = summary_df["method"].map(
            {m: i for i, m in enumerate(self.method_order)}
        )
        summary_df = summary_df.sort_values(["tau", "method_order"]).drop(
            columns=["method_order"]
        )
        self.save_table(raw_df, "label_efficiency_all_methods_raw.csv")
        self.save_table(summary_df, "label_efficiency_all_methods_summary.csv")
        self.plot_label_efficiency_summary(summary_df)
        return raw_df, summary_df

    def plot_label_efficiency_summary(self, summary_df):
        df_plot = summary_df[summary_df["mean_N_tau"].notna()].copy()
        if df_plot.empty:
            return
        tau_095 = df_plot[df_plot["tau"] == 0.95].sort_values("mean_N_tau")
        methods = tau_095["method"].tolist()
        if not methods:
            methods = df_plot["method"].drop_duplicates().tolist()
        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        for tau, offset in zip([0.90, 0.95], [-width / 2, width / 2]):
            tau_df = (
                df_plot[df_plot["tau"] == tau]
                .set_index("method")
                .reindex(methods)
            )
            values = tau_df["mean_N_tau"].to_numpy(dtype=float)
            bars = ax.bar(x + offset, values, width, label=rf"$\tau={tau:.2f}$", alpha=0.85)
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.5,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.method_label_short.get(m, m) for m in methods],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel(r"$N_{\tau}$")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        self.save_figure(fig, "label_efficiency_all_methods_tau_90_95.png")

    def plot_quantum_vs_classical_by_classifier(self):
        df = self.df[self.df["method"].isin(self.quantum_methods | self.classical_methods)].copy()
        df["group"] = np.where(
            df["method"].isin(self.quantum_methods),
            "Quantum-enhanced",
            "Classical",
        )
        dataset_level = (
            df.groupby(["dataset", "model", "group"], as_index=False)
            .agg(group_mean_accuracy=("final_accuracy", "mean"))
        )
        summary = (
            dataset_level.groupby(["model", "group"], as_index=False)
            .agg(
                mean_accuracy=("group_mean_accuracy", "mean"),
                std_accuracy=("group_mean_accuracy", "std"),
                n_datasets=("dataset", "nunique"),
            )
        )
        mean_pivot = summary.pivot(index="model", columns="group", values="mean_accuracy")
        std_pivot = summary.pivot(index="model", columns="group", values="std_accuracy")
        available_models = [m for m in self.model_label_map if m in mean_pivot.index]
        classical_means = mean_pivot.loc[available_models, "Classical"].to_numpy()
        quantum_means = mean_pivot.loc[available_models, "Quantum-enhanced"].to_numpy()
        classical_stds = std_pivot.loc[available_models, "Classical"].fillna(0).to_numpy()
        quantum_stds = std_pivot.loc[available_models, "Quantum-enhanced"].fillna(0).to_numpy()
        x = np.arange(len(available_models))
        width = 0.34
        fig, ax = plt.subplots(figsize=(9, 5.5))
        classical_bars = ax.bar(
            x - width / 2,
            classical_means,
            width,
            yerr=classical_stds,
            capsize=4,
            label="Classical",
            color="lightblue",
            alpha=0.85,
        )
        quantum_bars = ax.bar(
            x + width / 2,
            quantum_means,
            width,
            yerr=quantum_stds,
            capsize=4,
            label="Quantum-enhanced",
            color="lightcoral",
            alpha=0.85,
        )
        for bars, values in [(classical_bars, classical_means), (quantum_bars, quantum_means)]:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([self.model_label_map[m] for m in available_models])
        ax.set_ylabel("Average Final Accuracy")
        ax.set_ylim(0, 1.08)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        self.save_figure(fig, "quantum_vs_classical_by_classifier.png")
        comparison = pd.DataFrame({
            "model": available_models,
            "classifier": [self.model_label_map[m] for m in available_models],
            "classical_mean": classical_means,
            "quantum_mean": quantum_means,
            "difference_quantum_minus_classical": quantum_means - classical_means,
            "classical_std_across_datasets": classical_stds,
            "quantum_std_across_datasets": quantum_stds,
        })
        self.save_table(comparison, "quantum_vs_classical_by_classifier.csv")
        return comparison

    def plot_dataset_level_quantum_classical_difference(self):
        df = self.df[self.df["method"].isin(self.quantum_methods | self.classical_methods)].copy()
        df["group"] = np.where(
            df["method"].isin(self.quantum_methods),
            "Quantum-enhanced",
            "Classical",
        )
        task_group_mean = (
            df.groupby(["dataset", "model", "group"], as_index=False)
            .agg(group_mean_accuracy=("final_accuracy", "mean"))
        )
        task_pivot = (
            task_group_mean.pivot(
                index=["dataset", "model"],
                columns="group",
                values="group_mean_accuracy",
            )
            .reset_index()
        )
        task_pivot = task_pivot.dropna(subset=["Quantum-enhanced", "Classical"]).copy()
        task_pivot["difference_quantum_minus_classical"] = (
            task_pivot["Quantum-enhanced"] - task_pivot["Classical"]
        )
        dataset_summary = (
            task_pivot.groupby("dataset", as_index=False)
            .agg(
                mean_difference=("difference_quantum_minus_classical", "mean"),
                sd_difference=("difference_quantum_minus_classical", "std"),
                n_classifiers=("difference_quantum_minus_classical", "count"),
                quantum_mean=("Quantum-enhanced", "mean"),
                classical_mean=("Classical", "mean"),
            )
        )
        dataset_summary["sd_difference"] = dataset_summary["sd_difference"].fillna(0.0)
        dataset_summary["se_difference"] = (
            dataset_summary["sd_difference"]
            / np.sqrt(dataset_summary["n_classifiers"].clip(lower=1))
        )
        dataset_summary = dataset_summary.sort_values("mean_difference", ascending=False).reset_index(drop=True)
        self.save_table(task_pivot, "classifier_level_quantum_classical_difference.csv")
        self.save_table(dataset_summary, "dataset_level_quantum_classical_difference.csv")
        fig, ax = plt.subplots(figsize=(9.5, 7.2))
        y = np.arange(len(dataset_summary))
        colors = ["#d96b6b" if val >= 0 else "#7a9bc2" for val in dataset_summary["mean_difference"]]
        ax.barh(
            y,
            dataset_summary["mean_difference"],
            xerr=dataset_summary["se_difference"],
            color=colors,
            edgecolor="#666666",
            ecolor="black",
            capsize=4,
            alpha=0.95,
        )
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(dataset_summary["dataset"])
        ax.invert_yaxis()
        ax.set_xlabel("Mean final-accuracy difference (quantum-enhanced - classical)")
        ax.grid(True, axis="x", alpha=0.25)
        self.save_figure(fig, "dataset_level_quantum_classical_difference.png")
        return dataset_summary, task_pivot

    def plot_learning_curves(self):
        methods = [
            "random",
            "uncertainty",
            "entropy_diversity",
            "quantum_entropy",
            "quantum_uncertainty",
        ]
        datasets = [
            ("heart_disease", "Low-dimensional: Heart Disease"),
            ("ionosphere", "Medium-dimensional: Ionosphere"),
            ("sonar", "High-dimensional: Sonar"),
        ]
        color_map = {
            "random": "gray",
            "uncertainty": "black",
            "entropy_diversity": "tab:blue",
            "quantum_entropy": "tab:orange",
            "quantum_uncertainty": "red",
        }
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, (dataset, title) in zip(axes, datasets):
            dataset_df = self.curve_df[self.curve_df["dataset"] == dataset]
            if dataset_df.empty:
                ax.axis("off")
                continue
            plotted = []
            for method in methods:
                method_df = dataset_df[dataset_df["method"] == method]
                if method_df.empty:
                    continue
                summary = (
                    method_df.groupby("round", as_index=False)
                    .agg(mean_acc=("accuracy", "mean"))
                    .sort_values("round")
                )
                plotted.extend(summary["mean_acc"].tolist())
                ax.plot(
                    summary["round"],
                    summary["mean_acc"],
                    linewidth=2.2,
                    label=self.method_label_short.get(method, method),
                    color=color_map.get(method),
                )
            if plotted:
                y_min = max(0.0, min(plotted) - 0.03)
                y_max = min(1.0, max(plotted) + 0.03)
                if y_max - y_min < 0.12:
                    mid = (y_min + y_max) / 2
                    y_min = max(0.0, mid - 0.06)
                    y_max = min(1.0, mid + 0.06)
                ax.set_ylim(y_min, y_max)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Active Learning Round")
            ax.set_ylabel("Accuracy")
            ax.set_xlim(0, 15)
            ax.set_xticks([0, 3, 6, 9, 12, 15])
            ax.grid(True, axis="y", alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.08))
        fig.subplots_adjust(bottom=0.22, wspace=0.25)
        self.save_figure(fig, "learning_curves_complexity_representative.png")

    def plot_all_learning_curves(self):
        methods = [
            "random",
            "uncertainty",
            "entropy_diversity",
            "quantum_entropy",
            "quantum_uncertainty",
            "optimized_hybrid",
        ]
        datasets = sorted(self.curve_df["dataset"].dropna().unique().tolist())
        color_map = {
            "random": "gray",
            "uncertainty": "black",
            "entropy_diversity": "tab:blue",
            "quantum_entropy": "tab:orange",
            "quantum_uncertainty": "red",
            "optimized_hybrid": "tab:green",
        }
        ncols = 4
        nrows = int(np.ceil(len(datasets) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharey=False)
        axes = np.array(axes).flatten()
        for ax, dataset in zip(axes, datasets):
            dataset_df = self.curve_df[self.curve_df["dataset"] == dataset]
            if dataset_df.empty:
                ax.axis("off")
                continue
            for method in methods:
                method_df = dataset_df[dataset_df["method"] == method]
                if method_df.empty:
                    continue
                summary = (
                    method_df.groupby("round", as_index=False)
                    .agg(mean_acc=("accuracy", "mean"))
                    .sort_values("round")
                )
                ax.plot(
                    summary["round"],
                    summary["mean_acc"],
                    linewidth=2,
                    label=self.method_label_short.get(method, method),
                    color=color_map.get(method),
                )
            ax.set_title(dataset.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.grid(True, axis="y", alpha=0.25)
        for ax in axes[len(datasets):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.subplots_adjust(bottom=0.08, top=0.92, hspace=0.35, wspace=0.25)
        self.save_figure(fig, "learning_curves_all_datasets.png")

    def run_overall_performance_plots(self):
        self.build_task_level_final_performance_table()
        self.build_method_dataset_mean_table()
        self.export_strategy_dataset_metric_table(
            metric_col="final_accuracy",
            agg_name="Accuracy",
            save_prefix="accuracy",
        )
        self.export_strategy_dataset_metric_table(
            metric_col="final_f1",
            agg_name="F1",
            save_prefix="f1",
        )
        self.export_strategy_dataset_metric_table(
            metric_col="final_balanced_accuracy",
            agg_name="Balanced Accuracy",
            save_prefix="balanced_accuracy",
        )
        self.export_strategy_dataset_metric_table(
            metric_col="final_recall",
            agg_name="Recall",
            save_prefix="recall",
        )
        self.build_imbalance_aware_stress_test_table()
        self.plot_average_performance_by_method()
        self.plot_f1_by_method()
        self.plot_balanced_accuracy_by_method()
        self.plot_recall_by_method()
        self.plot_distribution_by_method()
        self.plot_quantum_vs_classical()
        self.plot_average_performance_by_model()
        self.plot_learning_curves()
        self.plot_all_learning_curves()
        self.statistical_significance_analysis()
        self.calculate_label_efficiency_all_methods()
        self.plot_quantum_vs_classical_by_classifier()
        self.run_wilcoxon_test_from_saved_results()
        self.plot_dataset_level_quantum_classical_difference()
        print("[Done] Result analysis finished.")


def build_paths(run_dir: Path):
    run_dir = run_dir.resolve()
    return {
        "raw_seed_path": run_dir / "tables" / "raw_seed_results.csv",
        "raw_curve_path": run_dir / "tables" / "raw_learning_curve_results.csv",
        "output_dir": run_dir / "replot_figures",
        "table_dir": run_dir / "replot_tables",
    }


def find_default_run_dir():
    project_root = Path(__file__).resolve().parents[1]
    candidates = []

    for child in project_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit() and not (
            len(child.name) == 15
            and child.name[:8].isdigit()
            and child.name[8] == "_"
            and child.name[9:].isdigit()
        ):
            continue

        raw_seed_path = child / "tables" / "raw_seed_results.csv"
        raw_curve_path = child / "tables" / "raw_learning_curve_results.csv"
        if raw_seed_path.exists() and raw_curve_path.exists():
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            "No run directory with tables/raw_seed_results.csv and "
            "tables/raw_learning_curve_results.csv was found."
        )

    return sorted(candidates)[-1]


def main():
    run_dir = find_default_run_dir()
    print(f"[Run directory] {run_dir.name}")
    paths = build_paths(run_dir)
    analysis = ResultAnalysis(**paths)
    analysis.run_overall_performance_plots()


if __name__ == "__main__":
    main()
