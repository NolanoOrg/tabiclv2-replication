"""Context-scaling: accuracy vs in-context (train) rows on large datasets.

Fixed test set (30% split, seed 0); train context subsampled (stratified) to
increasing sizes. stage2 vs stage3 vs released. -> logs/ctx_scaling.tsv
"""
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier

import os

# Root for run outputs (checkpoints, logs). Override with TABICL_RUN_DIR.
RUN_DIR = os.environ.get("TABICL_RUN_DIR", os.path.dirname(os.path.abspath(__file__)))


RNG = 0
MODELS = [
    ("stage2", os.path.join(RUN_DIR, "ckpt_stage2", "step-40000.ckpt")),
    ("stage3", os.path.join(RUN_DIR, "ckpt_stage3", "step-10000.ckpt")),
    ("released", None),
]
DATASETS = [("electricity", 1), ("adult", 2), ("bank-marketing", 1)]
SIZES = [1000, 5000, 15000, 30000, None]  # None = full train split

os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
out = open(os.path.join(RUN_DIR, "logs", "ctx_scaling.tsv"), "w", buffering=1)
out.write("model\tdataset\tctx_rows\tacc\tsecs\n")

data = {}
for name, ver in DATASETS:
    d = fetch_openml(name, version=ver, as_frame=True, parser="auto")
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    data[name] = (Xtr, Xte, ytr, yte)
    print(f"[data] {name}: train={len(Xtr)} test={len(Xte)}", flush=True)

for mname, mpath in MODELS:
    for dname, (Xtr, Xte, ytr, yte) in data.items():
        for k in SIZES:
            if k is not None and k >= len(Xtr):
                continue
            if k is None:
                Xs, ys, ctx = Xtr, ytr, len(Xtr)
            else:
                Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=k, random_state=RNG, stratify=ytr)
                ctx = k
            try:
                t0 = time.time()
                clf = TabICLClassifier(n_estimators=8, device="cuda", model_path=mpath,
                                       allow_auto_download=mpath is None, random_state=42)
                clf.fit(Xs, ys)
                acc = accuracy_score(yte, clf.predict(Xte))
                dt = time.time() - t0
                out.write(f"{mname}\t{dname}\t{ctx}\t{acc:.4f}\t{dt:.1f}\n")
                print(f"[{mname}] {dname:16s} ctx={ctx:6d} acc={acc:.4f} {dt:.1f}s", flush=True)
            except Exception as e:
                print(f"[{mname}] {dname} ctx={ctx} FAILED: {type(e).__name__}: {e}", flush=True)

out.close()
print("ctx_scaling done", flush=True)
