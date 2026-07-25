"""Stage-3 eval: stage2-final vs stage3-final vs released-v2.

Same protocol as eval_stage2.py: small suite + large suite, single split seed 0,
n_estimators=8, cuda. ET baseline once per dataset. -> logs/stage3_eval.tsv
"""
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

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
SMALL = [("credit-g", 1), ("diabetes", 1), ("vehicle", 1), ("phoneme", 1), ("kc1", 1), ("cmc", 1)]
LARGE = [("eeg-eye-state", 1), ("MagicTelescope", 1), ("electricity", 1), ("adult", 2), ("bank-marketing", 1)]

os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
out = open(os.path.join(RUN_DIR, "logs", "stage3_eval.tsv"), "w", buffering=1)
out.write("suite\tmodel\tdataset\tn\tacc\tet_acc\tsecs\n")

data = {}
for suite, tasks in (("small", SMALL), ("large", LARGE)):
    for name, ver in tasks:
        try:
            d = fetch_openml(name, version=ver, as_frame=True, parser="auto")
            X, y = d.data, d.target
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
            cat = Xtr.select_dtypes(include=["object", "category"]).columns
            Xtr_et, Xte_et = Xtr.copy(), Xte.copy()
            if len(cat):
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                Xtr_et[cat] = enc.fit_transform(Xtr[cat]); Xte_et[cat] = enc.transform(Xte[cat])
            et = ExtraTreesClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
            et.fit(Xtr_et, ytr)
            et_acc = accuracy_score(yte, et.predict(Xte_et))
            data[name] = (suite, Xtr, Xte, ytr, yte, et_acc, len(X))
            print(f"[data] {suite}/{name}: n={len(X)} f={X.shape[1]} et={et_acc:.4f}", flush=True)
        except Exception as e:
            print(f"[data] {name} FAILED: {type(e).__name__}: {e}", flush=True)

for mname, mpath in MODELS:
    for dname, (suite, Xtr, Xte, ytr, yte, et_acc, n) in data.items():
        try:
            t0 = time.time()
            clf = TabICLClassifier(n_estimators=8, device="cuda", model_path=mpath,
                                   allow_auto_download=mpath is None, random_state=42)
            clf.fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            dt = time.time() - t0
            out.write(f"{suite}\t{mname}\t{dname}\t{n}\t{acc:.4f}\t{et_acc:.4f}\t{dt:.1f}\n")
            print(f"[{mname}] {suite}/{dname:16s} acc={acc:.4f} (ET {et_acc:.4f}) {dt:.1f}s", flush=True)
        except Exception as e:
            print(f"[{mname}] {dname} FAILED: {type(e).__name__}: {e}", flush=True)

out.close()
print("eval_stage3 done", flush=True)
