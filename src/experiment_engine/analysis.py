from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


# Condition display order and colours for plots.
_CONDITION_ORDER = ["D", "SR", "PR", "RT", "Fill"]
_COLOURS = {"D": "#333333", "SR": "#1f77b4", "PR": "#ff7f0e", "RT": "#d62728", "Fill": "#9467bd"}
_MARKERS = {"D": "o", "SR": "s", "PR": "^", "RT": "D", "Fill": "x"}


class AnalysisEngine:
    def __init__(self, log_path: str, output_dir: str, run_id: str):
        self.log_path = Path(log_path)
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = self._load()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def summary_table(self) -> pd.DataFrame:
        """Mean accuracy and token stats per (condition, k), across seeds."""
        return (
            self.df.groupby(["condition", "k"])
            .agg(
                accuracy_mean=("correct", "mean"),
                accuracy_std=("correct", "std"),
                total_tokens_mean=("total_tokens", "mean"),
                total_tokens_std=("total_tokens", "std"),
                n=("correct", "count"),
            )
            .reset_index()
        )

    def plot_accuracy_vs_k(self, save: bool = True) -> plt.Figure:
        return self._plot_metric_vs_k(
            metric="correct",
            ylabel="Accuracy",
            title=f"Accuracy vs. Repetition Count — {self.run_id}",
            fname=f"{self.run_id}_accuracy_vs_k.png",
            save=save,
        )

    def plot_tokens_vs_k(self, save: bool = True) -> plt.Figure:
        return self._plot_metric_vs_k(
            metric="total_tokens",
            ylabel="Total Generated Tokens",
            title=f"Generated Tokens vs. Repetition Count — {self.run_id}",
            fname=f"{self.run_id}_tokens_vs_k.png",
            save=save,
        )

    def to_latex_table(self) -> str:
        """
        Generate the \\begin{tabular}...\\end{tabular} block for the paper.
        Rows = conditions, columns = k values.  Cells = accuracy ± std.
        """
        tbl = self.summary_table()
        ks = sorted(tbl["k"].unique())
        conditions = [c for c in _CONDITION_ORDER if c in tbl["condition"].values]

        col_spec = "l" + "c" * len(ks)
        header_ks = " & ".join(f"$k={k}$" for k in ks)

        lines = [
            "\\begin{tabular}{" + col_spec + "}",
            "\\toprule",
            f"Condition & {header_ks} \\\\",
            "\\midrule",
        ]

        for cond in conditions:
            sub = tbl[tbl["condition"] == cond].set_index("k")
            cells = []
            for k in ks:
                if k in sub.index:
                    mu = sub.loc[k, "accuracy_mean"]
                    sd = sub.loc[k, "accuracy_std"]
                    cells.append(f"${mu:.3f}_{{\\pm {sd:.3f}}}$")
                else:
                    cells.append("[--]")
            lines.append(f"$\\mathrm{{{cond}}}$ & " + " & ".join(cells) + " \\\\")

        lines += ["\\bottomrule", "\\end{tabular}"]
        latex = "\n".join(lines)

        out = self.output_dir / f"{self.run_id}_table.tex"
        out.write_text(latex)
        print(f"LaTeX table written to {out}")
        return latex

    def save_csv(self) -> None:
        path = self.output_dir / f"{self.run_id}_summary.csv"
        self.summary_table().to_csv(path, index=False)
        print(f"Summary CSV written to {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        rows = []
        with self.log_path.open(encoding="utf-8") as fh:
            import json
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        df["k"] = df["k"].astype(int)
        df["correct"] = df["correct"].astype(float)
        df["total_tokens"] = df["total_tokens"].astype(float)
        return df

    def _plot_metric_vs_k(
        self,
        metric: str,
        ylabel: str,
        title: str,
        fname: str,
        save: bool,
    ) -> plt.Figure:
        tbl = (
            self.df.groupby(["condition", "k"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        conditions = [c for c in _CONDITION_ORDER if c in tbl["condition"].values]

        for cond in conditions:
            sub = tbl[tbl["condition"] == cond].sort_values("k")
            ax.plot(
                sub["k"], sub["mean"],
                marker=_MARKERS.get(cond, "o"),
                color=_COLOURS.get(cond, "black"),
                label=cond,
                linewidth=1.8,
            )
            ax.fill_between(
                sub["k"],
                sub["mean"] - sub["std"],
                sub["mean"] + sub["std"],
                color=_COLOURS.get(cond, "black"),
                alpha=0.15,
            )

        ax.set_xlabel("Repetition count $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = self.output_dir / fname
            fig.savefig(path, dpi=150)
            print(f"Plot saved to {path}")

        return fig
