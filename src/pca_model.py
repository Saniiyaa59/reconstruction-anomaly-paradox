"""PCA capacity sweep: a linear autoencoder as a sanity check for the paradox.

Fits PCA on normal training data only, reconstructs test samples with k
components, and scores each sample by reconstruction error.

    python src/pca_model.py                          # ECG5000
    python src/pca_model.py --dataset mnist --normal 8
"""
import argparse
import csv
from pathlib import Path

from sklearn.decomposition import PCA

import data
import mnist_data
from evaluate import plot_paradox_curve, reconstruction_error, summarize

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

KS = {
    "ecg": [1, 2, 5, 10, 20, 30, 50, 80, 100, 120],
    "mnist": [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700],
}


def reconstruct(pca, X):
    return pca.inverse_transform(pca.transform(X))


def run_sweep(X_train, X_test, y_test, normal_class, ks):
    rows = []
    for k in ks:
        pca = PCA(n_components=k).fit(X_train)
        scores = reconstruction_error(X_test, reconstruct(pca, X_test))
        rows.append({"k": k, **summarize(scores, y_test, normal_class)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ecg", "mnist"], default="ecg")
    parser.add_argument("--normal", type=int, default=0, help="normal digit (mnist only)")
    args = parser.parse_args()

    if args.dataset == "ecg":
        splits = data.load_splits()
        normal_class = data.NORMAL_CLASS
        name, fig_name = "pca_ecg", "paradox_sanity_check.png"
        title = "PCA capacity sweep on ECG5000"
    else:
        splits = mnist_data.load_one_class(args.normal)
        normal_class = args.normal
        name = f"pca_mnist_normal{args.normal}"
        fig_name = f"{name}.png"
        title = f"PCA one-class MNIST (normal = {args.normal})"

    rows = run_sweep(splits["X_train"], splits["X_test"], splits["y_test"],
                     normal_class, KS[args.dataset])

    print(f"{'k':>4} {'normal err':>11} {'anomaly err':>12} {'AUROC':>7}")
    for r in rows:
        print(f"{r['k']:>4} {r['normal_err']:>11.4g} {r['anomaly_err']:>12.4g} {r['auroc']:>7.3f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_paradox_curve([r["k"] for r in rows], rows, xlabel="PCA components (k)",
                       title=title, path=FIGURES_DIR / fig_name)


if __name__ == "__main__":
    main()
