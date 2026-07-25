"""TabICLv2 Stage 2: context-extension pretraining (v2 paper recipe).

Recipe (paper Sec. "Three pretraining stages" + optimizer paragraph):
  - init from Stage-1 final checkpoint (model weights only; fresh optimizer)
  - 40K steps, batch 64, datasets 400-10,240 samples (log-uniform), 80% train
  - Muon max LR 1e-4, cosine; grad clip 10
  - [RUN-7 CONTINUITY] wd_mode plain 0.1 (validated), aux AdamW LR = 1e-4
    (equal to Muon max LR, the reference impl's single-LR convention; also
    exactly run-7's validated aux LR), warmup 2% (800 steps, stage-1 convention;
    paper does not state stage-2 warmup)

Data: remote gen_server --stage 2 (list-of-datasets protocol, one dataset per
micro-step; 64 grad-accumulation micro-steps per optimizer step).
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

import stage1_train as st
from gen_client import RobustFetcher
from tabicl._model.tabicl import TabICL

import os

# Root for run outputs (checkpoints, logs). Override with TABICL_RUN_DIR.
RUN_DIR = os.environ.get("TABICL_RUN_DIR", os.path.dirname(os.path.abspath(__file__)))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--init_from", default=os.path.join(RUN_DIR, "ckpt_stage1", "step-500000.ckpt"))
    ap.add_argument("--lr_muon", type=float, default=1e-4)
    ap.add_argument("--lr_adamw", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--wd_mode", default="plain", choices=["cautious", "plain"])
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--grad_clip", type=float, default=10.0)
    ap.add_argument("--remote_gen", default="10.0.0.7:29800")
    ap.add_argument("--checkpoint_dir", default=os.path.join(RUN_DIR, "ckpt_stage2"))
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume", default=None, help="'auto' or ckpt path (stage-2 resume)")
    ap.add_argument("--compile_blocks", type=int, default=0,
                    help="OFF: variable shapes make compile a wash (5.9s w/ recompile "
                         "stalls vs 4.8s eager, measured)")
    ap.add_argument("--fp32_col_attn", type=int, default=0,
                    help="OFF for stage 2 = the reference autocast computation. The fp32 "
                         "patch was a stage-1 stability measure (bf16 SDPA overflow at "
                         "sustained 8e-4); at LR 1e-4 on a converged lineage it costs "
                         "~2x step time (9.3s vs 4.8s). Revert to 1 if guard events appear.")
    ap.add_argument("--row_budget", type=int, default=26000,
                    help="max rows per forward (30K = 70.2GiB peak measured, eager)")
    cfg = ap.parse_args()
    cfg.regression = False          # stage 2 classification (save_checkpoint reads this)
    # cuDNN SDPA backward crashes on some stage-2 shapes (mha_graph.execute
    # failure, torch 2.12+cu130): crash-looped at step ~330. Flash/efficient only.
    torch.backends.cuda.enable_cudnn_sdp(False)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    dev = "cuda"

    st.install_fp32_ssmax()
    model = TabICL(max_classes=10).to(dev)
    model.train()
    if cfg.fp32_col_attn:
        st.install_fp32_col_attention(model)
    st.install_skip_path_dtype_fix()

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
        for g in opt.param_groups:      # CLI wins over restored values
            g["weight_decay"] = cfg.weight_decay
            g["lr_muon"] = cfg.lr_muon
            g["lr_adamw"] = cfg.lr_adamw
        start_step = ck["curr_step"]
        print(f"[resume] {resume_path} -> continuing at step {start_step}", flush=True)
    else:
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=True)
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"[init] stage-1 weights from {cfg.init_from} (fresh optimizer)", flush=True)

    if cfg.compile_blocks:
        st.install_block_compile(model)
        print("[model] attention blocks compiled (dynamic shapes, symbolic _logn)", flush=True)
    print(f"[model] {n_params/1e6:.2f}M params | stage 2 | wd={cfg.wd_mode} {cfg.weight_decay} "
          f"| lr {cfg.lr_muon}/{cfg.lr_adamw} | warmup {cfg.warmup}", flush=True)

    host, _, port = cfg.remote_gen.partition(":")
    fetch = RobustFetcher(host, int(port), start_index=start_step)

    # double-buffer prefetch: hide fetch+deserialize (~0.2-0.4s) behind GPU compute
    import queue as _queue
    import threading as _threading
    _q = _queue.Queue(maxsize=2)
    def _prefetch():
        while True:
            Xs, ys, d, sl, ts = fetch.fetch()
            # pin in the background thread (overlaps GPU compute) so the
            # per-micro .to(cuda, non_blocking=True) is a true async DMA
            _q.put((Xs, ys, d, sl, ts))
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
        acc_num = 0
        acc_den = 0
        ROW_BUDGET = cfg.row_budget
        with amp:
            i = 0
            while i < n_ds:
                n_i = int(sl[i].item())
                j = i + 1   # extend over the gen group (consecutive equal seq len)
                while j < n_ds and int(sl[j].item()) == n_i:
                    j += 1
                # balanced split: g datasets into k=ceil(g/per) near-equal micros
                # (6+2 -> 4+4: the tiny remainder kernel wastes launch overhead)
                g = j - i
                per = max(1, ROW_BUDGET // n_i)
                k = -(-g // per)
                per = -(-g // k)
                for a in range(i, j, per):
                    b = min(a + per, j)
                    dd = int(d[a:b].max().item())
                    ts = int(ts_all[a].item())          # equal within group (same n, 80%)
                    mb_X = torch.stack([Xs[t][:, :dd] for t in range(a, b)]).to(dev, non_blocking=True)
                    mb_y = torch.stack([ys[t] for t in range(a, b)]).to(dev, non_blocking=True)
                    pred = model(mb_X, mb_y[:, :ts], None)
                    tgt = mb_y[:, ts:].long().flatten()
                    loss = F.cross_entropy(pred.flatten(end_dim=-2), tgt)
                    w = (b - a) / n_ds                  # equal weight per dataset
                    with torch.autocast(device_type="cuda", enabled=False):
                        (loss * w).backward()
                    loss_acc += loss.detach().float() * w
                    acc_num += (pred.argmax(-1).flatten() == tgt).sum()
                    acc_den += tgt.numel()
                i = j
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
