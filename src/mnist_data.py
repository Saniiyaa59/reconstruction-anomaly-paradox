"""Download MNIST and build one-class splits.

Uses the standard 60k/10k train/test split. For a chosen normal digit:

    train : training images of the normal digit only
    test  : all 10k test images; y_test keeps the original digit labels
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_mnist():
    """Return (X_train, y_train, X_test, y_test); images flattened to 784, scaled to [0, 1]."""
    path = PROCESSED_DIR / "mnist.npz"
    if not path.exists():
        print("downloading MNIST (first run only)")
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True,
                            parser="liac-arff", data_home=str(RAW_DIR / "openml"))
        X = (X / 255.0).astype(np.float32)
        y = y.astype(int)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, X_train=X[:60000], y_train=y[:60000], X_test=X[60000:], y_test=y[60000:])
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def load_one_class(normal_digit):
    X_train, y_train, X_test, y_test = load_mnist()
    return {
        "X_train": X_train[y_train == normal_digit],
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train {X_train.shape} | test {X_test.shape}")
    print("train counts:", dict(zip(*map(np.ndarray.tolist, np.unique(y_train, return_counts=True)))))
