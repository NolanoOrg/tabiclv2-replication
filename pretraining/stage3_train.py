"""TabICLv2 Stage 3: long-context pretraining (v2 paper recipe).

Recipe (paper Sec. "Three pretraining stages" + appendix "Speed and memory"):
  - init from Stage-2 final checkpoint (model only; fresh optimizer)
  - 10K steps, batch 64, datasets 400-60,000 samples (log-uniform), 80% train
  - Muon max LR 2e-5, cosine; grad clip 1.0 (the clip-10 setting was stages 1-2)
  - NO freezing (v1 froze col/row; v2 trains everything)
  - gradient (activation) checkpointing for datasets exceeding 20K samples
  - [RUN-7/S2 CONTINUITY] wd plain 0.1, aux AdamW LR = Muon max LR, warmup 2%
    (200 steps; paper silent), bf16 autocast, fp32 ssmax, flash/efficient SDPA
    (cuDNN backend disabled: crashes + shape-churn overhead, see stage 2)
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

import stage1_train as st
from gen_client import RobustFetcher
from tabicl._model.tabicl import TabICL

import os

# Root for run outputs (checkpoints, logs). Override with TABICL_RUN_DIR.
RUN_DIR = os.environ.get("TABICL_RUN_DIR", os.path.dirname(os.path.abspath(__file__)))


CKPT_ON = False   # toggled per micro-batch (datasets > ckpt_threshold rows)


def install_toggleable_ckpt(model):
    """Wrap every transformer block so its forward routes through activation
    checkpointing when CKPT_ON is set. Dropout is 0 -> recompute is exact."""
    def wrap(blk):
        orig = blk.forward
        def fwd(*a, **k):
            if CKPT_ON and torch.is_grad_enabled():
                return _ckpt(orig, *a, use_reentrant=False, **k)
            return orig(*a, **k)
        blk.forward = fwd
    n = 0
    for blocks in (model.col_embedder.tf_col.blocks,
                   model.row_interactor.tf_row.blocks,
                   model.icl_predictor.tf_icl.blocks):
        for blk in blocks:
            wrap(blk); n += 1
    print(f"[model] toggleable activation checkpointing on {n} blocks", flush=True)


def main():
    global CKPT_ON
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--init_from", default=os.path.join(RUN_DIR, "ckpt_stage2", "step-40000.ckpt"))
    ap.add_argument("--lr_muon", type=float, default=2e-5)
    ap.add_argument("--lr_adamw", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--wd_mode", default="plain", choices=["cautious", "plain"])
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad_clip", type=float, default=1.0)   # v2: stages 1-2 only use 10
    ap.add_argument("--ckpt_threshold", type=int, default=20000)
    ap.add_argument("--row_budget", type=int, default=26000)
    ap.add_argument("--remote_gen", default="10.0.0.7:29900")
    ap.add_argument("--checkpoint_dir", default=os.path.join(RUN_DIR, "ckpt_stage3"))
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--resume", default=None)
    cfg = ap.parse_args()
    cfg.regression = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    dev = "cuda"

    st.install_fp32_ssmax()
    model = TabICL(max_classes=10).to(dev)
    model.train()
    st.install_skip_path_dtype_fix()
    install_toggleable_ckpt(model)

    n_params = sum(p.numel() for p in model.parameters())
    mu, ad = st.split_params(model)
    opt = st.MuonAdamW(mu, ad, lr_muon=cfg.lr_muon, lr_adamw=cfg.lr_adamw,
                       weight_decay=cfg.weight_decay)
    opt.wd_mode = cfg.wd_mode

    start_step = 0
    resume_path = None
    if cfg.resume == "auto":
        import glob
        cands = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "step-*.ckpt")),
                       key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
        resume_path = cands[-1] if cands else None
    elif cfg.resume:
        resume_path = cfg.resume
    if resume_path:
        ck = torch.load(resume_path, map_location=dev, weights_only=True)
        model.load_state_dict(ck["state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer_state"])
        for g in opt.param_groups:
            g["weight_decay"] = cfg.weight_decay
            g["lr_muon"] = cfg.lr_muon
            g["lr_adamw"] = cfg.lr_adamw
        start_step = ck["curr_step"]
        print(f"[resume] {resume_path} -> continuing at step {start_step}", flush=True)
    else:
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=True)
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"[init] stage-2 weights from {cfg.init_from} (fresh optimizer)", flush=True)
    print(f"[model] {n_params/1e6:.2f}M params | stage 3 | wd={cfg.wd_mode} {cfg.weight_decay} "
          f"| lr {cfg.lr_muon}/{cfg.lr_adamw} | clip {cfg.grad_clip} | warmup {cfg.warmup} "
          f"| ckpt>{cfg.ckpt_threshold} rows", flush=True)

    host, _, port = cfg.remote_gen.partition(":")
    fetch = RobustFetcher(host, int(port), start_index=start_step)
    import queue as _queue
    import threading as _threading
    _q = _queue.Queue(maxsize=2)
    def _prefetch():
        while True:
            _q.put(fetch.fetch())
    _threading.Thread(target=_prefetch, daemon=True).start()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    ema = None
    t_log = time.time()
    for step in range(start_step, cfg.steps):
        if step < cfg.warmup:
            opt.lr_scale = step / max(1, cfg.warmup)
        else:
            opt.lr_scale = 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup) / (cfg.steps - cfg.warmup)))

        t0 = time.time()
        Xs, ys, d, sl, ts_all = _q.get()
        t_gen = time.time() - t0
        n_ds = len(Xs)
        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=dev)
        acc_num = 0; acc_den = 0
        with amp:
            i = 0
            while i < n_ds:
                n_i = int(sl[i].item()); j = i + 1
                while j < n_ds and int(sl[j].item()) == n_i:
                    j += 1
                g = j - i
                per = max(1, cfg.row_budget // n_i)
                k = -(-g // per); per = -(-g // k)
                CKPT_ON_local = n_i > cfg.ckpt_threshold
                globals()["CKPT_ON"] = CKPT_ON_local
                for a in range(i, j, per):
                    b = min(a + per, j)
                    dd = int(d[a:b].max().item()); ts = int(ts_all[a].item())
                    mb_X = torch.stack([Xs[t][:, :dd] for t in range(a, b)]).to(dev, non_blocking=True)
                    mb_y = torch.stack([ys[t] for t in range(a, b)]).to(dev, non_blocking=True)
                    pred = model(mb_X, mb_y[:, :ts], None)
                    tgt = mb_y[:, ts:].long().flatten()
                    loss = F.cross_entropy(pred.flatten(end_dim=-2), tgt)
                    w = (b - a) / n_ds
                    with torch.autocast(device_type="cuda", enabled=False):
                        (loss * w).backward()
                    loss_acc += loss.detach().float() * w
                    acc_num += (pred.argmax(-1).flatten() == tgt).sum()
                    acc_den += tgt.numel()
                i = j
        globals()["CKPT_ON"] = False
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lv = loss_acc.item(); gn = total_norm.item()
        if math.isfinite(gn) and math.isfinite(lv):
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            print(f"[guard] step {step} gnorm={gn} loss={lv}", flush=True)
        ema = lv if ema is None else 0.9 * ema + 0.1 * lv

        if step % cfg.log_every == 0:
            acc = (acc_num / max(1, acc_den)).item()
            mult = st.icl0_multiplier(model)
            rate = cfg.log_every / max(1e-9, time.time() - t_log) if step > start_step else 0.0
            t_log = time.time()
            print(f"step {step:5d} | loss {lv:.4f} (ema {ema:.4f}) | acc {acc:.4f} | "
                  f"lr {cfg.lr_muon*opt.lr_scale:.2e} | gen {t_gen:.2f}s | "
                  f"{rate:.2f} step/s | mult {mult:.2f}", flush=True)
        if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
            st.save_checkpoint(model, opt, step + 1, cfg)

    print(f"Done {cfg.steps} steps. final loss {lv:.4f} | "
          f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
