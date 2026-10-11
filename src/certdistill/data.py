"""MNIST IDX loading (numpy, no torchvision), client partitioning, probe set."""
import gzip
import numpy as np
import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "mnist")


def _read_idx(path, kind):
    with gzip.open(path + ".gz", "rb") as f:
        data = f.read()
    if kind == "images":
        assert int.from_bytes(data[0:4], "big") == 2051
        n = int.from_bytes(data[4:8], "big")
        arr = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, 28, 28)
        return (arr.astype(np.float32) / 255.0)[:, None, :, :]  # NCHW
    else:
        assert int.from_bytes(data[0:4], "big") == 2049
        n = int.from_bytes(data[4:8], "big")
        return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def load_mnist():
    Xtr = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte"), "images")
    ytr = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"), "labels")
    Xte = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"), "images")
    yte = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"), "labels")
    return Xtr, ytr, Xte, yte


def make_federation(Xtr, ytr, n_clients, probe_size, iid=True, seed=0):
    """Returns (client_data: list of (X, y), probe_X). Probe labels are discarded."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[perm], ytr[perm]
    probe_X = Xtr[:probe_size]
    rest_X, rest_y = Xtr[probe_size:], ytr[probe_size:]
    n = len(rest_X)
    if iid:
        idx = rng.permutation(n)
        chunks = np.array_split(idx, n_clients)
        client_data = [(rest_X[c], rest_y[c]) for c in chunks]
    else:
        # Dirichlet(0.5) label skew
        alpha = 0.5
        label_idx = [np.where(rest_y == c)[0] for c in range(10)]
        for lst in label_idx:
            rng.shuffle(lst)
        ptr = [0] * 10
        client_data = []
        for _ in range(n_clients):
            props = rng.dirichlet([alpha] * 10)
            take = []
            for c in range(10):
                cnt = int(round(props[c] * n / n_clients))
                take.extend(label_idx[c][ptr[c]:ptr[c] + cnt].tolist())
                ptr[c] += cnt
            take = np.array(take, dtype=np.int64)
            rng.shuffle(take)
            client_data.append((rest_X[take], rest_y[take]))
    return client_data, probe_X
