"""PCA capacity sweep: a linear autoencoder as a sanity check for the paradox.

Fits PCA on normal training beats only, reconstructs test beats with k
components, and scores each beat by reconstruction error.
"""
import csv
from pathlib import Path

from sklearn.decomposition import PCA

from data import load_splits
from evaluate import plot_paradox_curve, reconstruction_error, summarize

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

KS = [1, 2, 5, 10, 20, 30, 50, 80, 100, 120]


def reconstruct(pca, X):
    return pca.inverse_transform(pca.transform(X))


def run_sweep(ks=KS):
    splits = load_splits()
    X_train, X_test, y_test = splits["X_train"], splits["X_test"], splits["y_test"]

    rows = []
    for k in ks:
        pca = PCA(n_components=k).fit(X_train)
        scores = reconstruction_error(X_test, reconstruct(pca, X_test))
        rows.append({"k": k, **summarize(scores, y_test)})
    return rows


def main():
    rows = run_sweep()

    print(f"{'k':>4} {'normal err':>11} {'anomaly err':>12} {'AUROC':>7}")
    for r in rows:
        print(f"{r['k']:>4} {r['normal_err']:>11.4g} {r['anomaly_err']:>12.4g} {r['auroc']:>7.3f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "pca_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_paradox_curve([r["k"] for r in rows], rows,
                       xlabel="PCA components (k)",
                       title="PCA capacity sweep on ECG5000",
                       path=FIGURES_DIR / "paradox_sanity_check.png")


if __name__ == "__main__":
    main()
