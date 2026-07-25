"""Faithful TabICLv2 Stage-1 pretraining (short 1k-step de-risk run).

Matches the paper's Stage 1 recipe as closely as the code allows:
  - Prior: graph_scm (v2) with ExtraTrees + graph-ancestry filtering ON.
  - Data: 1024 samples/dataset, 30-90% train (sampled per micro-batch), up to
    100 features, batch size 64.
  - Model: v2 TabICL defaults (27.5M params), classification head (max_classes=10).
  - Optimizer: Muon (Newton-Schulz, Moonlight 0.2*sqrt(max(n,m)) LR scaling) for
    2D hidden weights + AdamW for embeddings/heads/norms, cautious weight decay 0.01.
  - Max LR 8e-4 (Muon), cosine schedule w/ warmup, gradient clipping 10, AMP bf16.

Paper points intentionally deviating / underspecified are flagged inline as [DEVIATION].
"""
from __future__ import annotations
import math
import time
import queue
import functools
import threading
import argparse
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F

import tabicl.prior._dataset as _pds
import tabicl._model.ssmax as _ssmax
from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import PriorDataset
from tabicl.prior.graph_lib._config import PriorConfig


def _logn_symbolic(n, device=None, dtype=None):
    """Bit-exact replacement for ssmax._logn (verified identical for all
    n in 1..61440): computes log via a torch float64 op (same libm) instead of
    math.log, which keeps `n` SYMBOLIC under torch.compile — without this,
    dynamo burns train_size into a per-value guard (615 graph variants)."""
    t = torch.scalar_tensor(max(n, 1), dtype=torch.float64).log()
    return t.to(device=device, dtype=dtype)


def icl0_multiplier(model, n=512):
    """Health telemetry: the icl-block-0 attention-score multiplier — the
    component whose Muon-driven runaway caused the step-24K divergence.
    Released model's converged value: ~12.3 (max). Pathological: >20."""
    sd = {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()}
    p = "icl_predictor.tf_icl.blocks.0.attn.ssmax_layer.base_mlp"
    x = torch.tensor([[math.log(n)]], device=sd[p + ".0.weight"].device)
    out = F.gelu(x @ sd[p + ".0.weight"].float().T + sd[p + ".0.bias"].float()) \
          @ sd[p + ".2.weight"].float().T + sd[p + ".2.bias"].float()
    return out.abs().max().item()


def save_checkpoint(model, opt, step, cfg):
    """Save in the repo checkpoint format ({config, state_dict, ...}) so the
    result loads directly into tabicl.TabICLClassifier for evaluation.
    Strips torch.compile's '._orig_mod' prefixes so keys match vanilla TabICL."""
    import os
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    state = {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()}
    ckpt = {
        "config": {"max_classes": 0 if cfg.regression else 10},
        "state_dict": state,
        "optimizer_state": opt.state_dict(),
        "curr_step": step,
    }
    path = os.path.join(cfg.checkpoint_dir, f"step-{step}.ckpt")
    torch.save(ckpt, path)
    return path


def install_fp32_ssmax():
    """Run QASSMaxMLP in fp32 (forward + its backward). The bf16 backward of
    this op overflowed on saturated attention states (nan grads localized to
    col block-2 ssmax at step ~20K), and its spiky gradients are the source of
    the 10-50x grad-norm bursts. Output is cast back to the ambient dtype, so
    downstream computation is unchanged; internals compute at full precision.
    Must be installed BEFORE install_block_compile (compile captures it)."""
    import tabicl._model.ssmax as _sm
    if hasattr(_sm.QASSMaxMLP, "_orig_fwd"):
        return
    _sm.QASSMaxMLP._orig_fwd = _sm.QASSMaxMLP.forward

    def fwd32(self, q, n):
        with torch.autocast(device_type="cuda", enabled=False):
            return _sm.QASSMaxMLP._orig_fwd(self, q.float(), n).to(q.dtype)

    _sm.QASSMaxMLP.forward = fwd32


def install_fp32_col_attention(model):
    """ESCALATION: run every nn.MultiheadAttention inside col_embedder fully in
    fp32 (weights matmuls + SDPA + backward). Evidence: nan grads persist with
    fp32-SSMax alone -> the bf16 cuDNN SDPA backward is the remaining overflow
    source; ssmax was merely downstream in the backward chain. Cost ~+80ms/step
    (col attention at 2x bytes). Install BEFORE install_block_compile."""
    import torch.nn as nn

    def wrap(mha):
        orig = mha.forward

        def fwd32(*args, **kw):
            dt = args[0].dtype
            with torch.autocast(device_type="cuda", enabled=False):
                a32 = [x.float() if torch.is_tensor(x) and x.is_floating_point() else x
                       for x in args]
                res = orig(*a32, **kw)
            def back(r):
                return r.to(dt) if torch.is_tensor(r) and r.is_floating_point() else r
            return tuple(back(r) for r in res) if isinstance(res, tuple) else back(res)

        mha.forward = fwd32

    n = 0
    for mod in model.col_embedder.modules():
        if isinstance(mod, nn.MultiheadAttention):
            wrap(mod)
            n += 1
    print(f"[model] fp32 col-attention: wrapped {n} MultiheadAttention modules")


def install_skip_path_dtype_fix():
    """The partial-skip path in InducedSelfAttentionBlock.forward index-puts the
    attention output (fp32 under autocast: final LayerNorm + fp32 col-attn) into
    a torch.empty_like(src) buffer that inherits src's bf16 dtype ->
    'Index put requires the source and destination dtypes match'. First trip:
    run-7 step ~71.7K, the first batch with a partially-skipped column group.
    Fix: allocate the buffer with the result's dtype. The no-skip and all-skip
    paths are untouched -> bit-exact with every previously completed step.
    Install BEFORE install_block_compile so compiled blocks capture it."""
    from tabicl._model.layers import InducedSelfAttentionBlock

    def forward(self, src, train_size=None):
        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if skip_mask.any():
            if skip_mask.all():
                out = torch.full_like(src, self.skip_value)
            else:
                res = self.induced_attention(src[~skip_mask], train_size)
                out = torch.empty(src.shape, dtype=res.dtype, device=src.device)
                out[~skip_mask] = res
                out[skip_mask] = self.skip_value
            return out
        return self.induced_attention(src, train_size)

    InducedSelfAttentionBlock.forward = forward
    print("[model] skip-path dtype fix installed (InducedSelfAttentionBlock)")


def install_block_compile(model):
    """Compile the attention blocks of all three components with dynamic
    shapes. With _logn_symbolic installed, each block family needs ~2 graph
    variants total (batch-dim + ts fully symbolic). Numerics: epsilon-class
    (kernel fusion), same class as the accepted cuDNN nondeterminism."""
    _ssmax._logn = _logn_symbolic
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    for i, blk in enumerate(model.col_embedder.tf_col.blocks):
        model.col_embedder.tf_col.blocks[i] = torch.compile(blk, dynamic=True)
    for i, blk in enumerate(model.row_interactor.tf_row.blocks):
        model.row_interactor.tf_row.blocks[i] = torch.compile(blk, dynamic=True)
    for i, blk in enumerate(model.icl_predictor.tf_icl.blocks):
        model.icl_predictor.tf_icl.blocks[i] = torch.compile(blk, dynamic=True)


_GEN_POOL = None


def _seeded_call(payload):
    """Run one generation task under a parent-assigned seed (executes in worker).

    REPRODUCIBILITY FIX: upstream generate_dataset has no per-task seeding, so
    dataset content depended on OS scheduling of task->worker assignment (worker
    RNG streams diverge). Seeding every task from the parent's seeded RNG makes
    generation deterministic regardless of which worker runs it. The generator's
    sampling logic is untouched — only the RNG stream is pinned."""
    import random as _random
    seed, func, arg = payload
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return func(arg)


_ON_DISPATCHED = None   # optional callback fired right after tasks are submitted
                        # (used by PipelinedPrefetcher to release the RNG ticket)


def _worker_init(worker_cores):
    """Pool worker init: single-threaded torch, pinned to the worker core set,
    slightly niced so the GPU-feeding main thread wins the scheduler."""
    torch.set_num_threads(1)
    try:
        if worker_cores:
            import os
            os.sched_setaffinity(0, worker_cores)
            os.nice(5)
    except Exception:
        pass  # affinity is a perf hint only


def install_persistent_pool(n_jobs: int, worker_cores=None):
    """PROFILING FIX: create the generation pool ONCE (must run before CUDA/model
    init so workers fork from a small parent) and reuse it for every batch.

    Replaces tabicl.prior._dataset.run_parallel, which forks a fresh Pool per
    get_batch() call — fork cost grows with parent RSS and caused the observed
    gen-wait degradation (0.1s -> 13s over 1000 steps)."""
    global _GEN_POOL
    _GEN_POOL = mp.get_context("fork").Pool(
        processes=n_jobs, initializer=functools.partial(_worker_init, worker_cores))

    def run_parallel_seeded(func, args, n_jobs=-1):
        # Seeds drawn from the parent's (seeded) np.random stream -> deterministic.
        # Data content is invariant to worker count / scheduling / pipelining.
        seeds = np.random.randint(0, 2**31 - 1, size=len(args))
        async_res = _GEN_POOL.map_async(_seeded_call, [(int(s), func, a) for s, a in zip(seeds, args)])
        if _ON_DISPATCHED is not None:
            _ON_DISPATCHED()   # parent RNG no longer needed for this batch
        return async_res.get()

    _pds.run_parallel = run_parallel_seeded


# --------------------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalization) + AdamW aux, with cautious weight decay
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration to orthogonalize G (Keller Jordan coeffs)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()   # reference impl (Schaipp/Moonlight) runs NS in fp32; bf16 caused divergent iterations on spiky grads
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def _newton_schulz5_batched(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched quintic Newton-Schulz over (k, n, m) — same per-matrix math as
    _newton_schulz5 (all matrices in a batch share one shape, so the transpose
    decision and reduction dims match the single-matrix path exactly)."""
    assert G.ndim == 3
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()   # fp32, matching reference impl
    transpose = X.size(1) > X.size(2)
    if transpose:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.transpose(1, 2)
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2D hidden weights, AdamW for the rest, with cautious weight decay.

    Cautious weight decay [Chen 2025]: decay a coordinate only when the applied
    update and the parameter share sign (interpreted here as ``update * param > 0``).
    """

    def __init__(self, muon_params, adamw_params, lr_muon=8e-4, lr_adamw=3e-4,
                 momentum=0.95, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr_muon=lr_muon, lr_adamw=lr_adamw, momentum=momentum,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(
            [{"params": muon_params, "use_muon": True},
             {"params": adamw_params, "use_muon": False}], defaults)
        self.lr_scale = 1.0  # set by the LR scheduler each step

    fast = True       # foreach elementwise ops (bit-exact per element)
    ns_group = True   # shape-grouped batched Newton-Schulz (validated bit-exact vs loop)
    wd_mode = "cautious"  # 'cautious' (paper text) | 'plain' (reference impl default;
                          # creates a weight-norm equilibrium that CWD cannot)

    @torch.no_grad()
    def step(self, closure=None):
        if self.fast:
            return self._step_fast()
        return self._step_reference()

    @torch.no_grad()
    def _step_reference(self, closure=None):
        for group in self.param_groups:
            wd = group["weight_decay"]
            if group["use_muon"]:
                lr = group["lr_muon"] * self.lr_scale
                mu = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "mom" not in st:
                        st["mom"] = torch.zeros_like(p)
                    buf = st["mom"]
                    buf.mul_(mu).add_(p.grad)
                    g = p.grad.add(buf, alpha=mu)             # Nesterov
                    u = _newton_schulz5(g)
                    u.mul_(0.2 * math.sqrt(max(p.shape)))     # Moonlight RMS match
                    if wd > 0:                                 # cautious weight decay
                        mask = (u * p > 0).to(p.dtype)
                        p.mul_(1 - lr * wd * mask)
                    p.add_(u, alpha=-lr)
            else:
                lr = group["lr_adamw"] * self.lr_scale
                b1, b2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["step"] += 1
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]
                    bc2 = 1 - b2 ** st["step"]
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    upd = (m / bc1) / denom
                    if wd > 0:                                 # cautious weight decay
                        mask = (upd * p > 0).to(p.dtype)
                        p.mul_(1 - lr * wd * mask)
                    p.add_(upd, alpha=-lr)

    @torch.no_grad()
    def _step_fast(self):
        """Same arithmetic as _step_reference, batched:
        - elementwise state updates via torch._foreach_* (identical per-element ops)
        - Newton-Schulz grouped by weight shape via bmm (validated bit-exact)."""
        for group in self.param_groups:
            wd = group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            if group["use_muon"]:
                lr = group["lr_muon"] * self.lr_scale
                mu = group["momentum"]
                bufs = []
                for p in params:
                    st = self.state[p]
                    if "mom" not in st:
                        st["mom"] = torch.zeros_like(p)
                    bufs.append(st["mom"])
                torch._foreach_mul_(bufs, mu)
                torch._foreach_add_(bufs, grads)                       # buf = mu*buf + g
                gs = torch._foreach_add(grads, bufs, alpha=mu)         # Nesterov
                # Newton-Schulz, grouped by shape
                if self.ns_group:
                    updates = [None] * len(gs)
                    by_shape = {}
                    for i, g in enumerate(gs):
                        by_shape.setdefault(tuple(g.shape), []).append(i)
                    for shape, idxs in by_shape.items():
                        if len(idxs) == 1:
                            updates[idxs[0]] = _newton_schulz5(gs[idxs[0]])
                        else:
                            out = _newton_schulz5_batched(torch.stack([gs[i] for i in idxs]))
                            for j, i in enumerate(idxs):
                                updates[i] = out[j]
                else:
                    updates = [_newton_schulz5(g) for g in gs]
                for u, p in zip(updates, params):
                    u.mul_(0.2 * math.sqrt(max(p.shape)))
                if wd > 0:
                    if self.wd_mode == "plain":
                        torch._foreach_mul_(params, 1 - lr * wd)       # reference-style decay
                    else:
                        for u, p in zip(updates, params):              # cautious wd
                            mask = (u * p > 0).to(p.dtype)
                            p.mul_(1 - lr * wd * mask)
                torch._foreach_add_(params, updates, alpha=-lr)
            else:
                lr = group["lr_adamw"] * self.lr_scale
                b1, b2 = group["betas"]
                eps = group["eps"]
                ms, vs = [], []
                for p in params:
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["step"] += 1
                    ms.append(st["m"])
                    vs.append(st["v"])
                k = self.state[params[0]]["step"]                      # same for all
                torch._foreach_mul_(ms, b1)
                torch._foreach_add_(ms, grads, alpha=1 - b1)
                torch._foreach_mul_(vs, b2)
                torch._foreach_addcmul_(vs, grads, grads, value=1 - b2)
                bc1 = 1 - b1 ** k
                bc2 = 1 - b2 ** k
                denom = torch._foreach_sqrt(vs)                        # v.sqrt()
                torch._foreach_div_(denom, math.sqrt(bc2))             # / sqrt(bc2)
                torch._foreach_add_(denom, eps)                        # .add_(eps)
                upd = torch._foreach_div(ms, bc1)                      # m / bc1
                torch._foreach_div_(upd, denom)                        # / denom
                if wd > 0:
                    if self.wd_mode == "plain":
                        torch._foreach_mul_(params, 1 - lr * wd)
                    else:
                        for u, p in zip(upd, params):                  # cautious wd
                            mask = (u * p > 0).to(p.dtype)
                            p.mul_(1 - lr * wd * mask)
                torch._foreach_add_(params, upd, alpha=-lr)


def split_params(model):
    """2D hidden weights -> Muon; embeddings/heads/norms/biases/degenerate -> AdamW."""
    muon, adamw = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d = p.ndim == 2 and min(p.shape) > 1
        is_embed_or_head = any(k in n for k in
                               ["in_linear", "ind_vectors", "y_encoder", "decoder.2",
                                "ssmax", "cls"])   # modulators/tokens: AdamW (validated
                                                    # fix for the Muon-driven scale runaway)
        if is_2d and not is_embed_or_head:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def cosine_lr(step, total, warmup, floor=0.0):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))


# --------------------------------------------------------------------------------------
# Prior (Stage 1) with a prefetching worker so generation overlaps GPU compute
# --------------------------------------------------------------------------------------
def build_prior(cfg):
    prior_config = PriorConfig(
        filter_unpredictable_datasets=True,   # ExtraTrees bootstrap test (paper)
        filter_unpredictable_graphs=True,     # x/y common-ancestor filter (paper)
    )
    stage = getattr(cfg, "stage", 1)
    if stage == 2:
        # v2 paper stage 2: 400-10,240 samples log-uniform, 80% train
        # exact 80% train: validator requires min<max; epsilon never crosses the
        # next integer after int(0.8*n) for n<=10240 (eps*n < 1e-3)
        seq_kw = dict(min_seq_len=400, max_seq_len=10240, log_seq_len=True,
                      min_train_size=0.8, max_train_size=0.8 + 1e-8)
    elif stage == 3:
        # v2 paper stage 3: 400-60,000 samples log-uniform, 80% train
        seq_kw = dict(min_seq_len=400, max_seq_len=60000, log_seq_len=True,
                      min_train_size=0.8, max_train_size=0.8 + 1e-8)
    else:
        seq_kw = dict(min_seq_len=1024, max_seq_len=1025,       # ~1024 samples (Stage 1)
                      min_train_size=0.3, max_train_size=0.9)   # 30-90% train (Stage 1)
    return PriorDataset(
        prior_type="graph_scm",
        regression=cfg.regression,
        batch_size=cfg.batch_size,
        batch_size_per_gp=cfg.micro_batch_size,   # one group == one micro-batch
        min_features=2, max_features=100,
        max_classes=10,
        seq_len_per_gp=True,                      # per-group seq len + train/test split
        n_jobs=cfg.prior_jobs,
        device="cpu",
        config=prior_config,
        **seq_kw,
    )


class Prefetcher:
    """Background thread that generates prior batches into a queue, overlapping
    CPU generation with GPU compute. The prior's multiprocessing pool works fine
    from a (non-daemon) main-process thread."""

    def __init__(self, dataset, depth=3):
        self.dataset = dataset
        self.q = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        while True:
            self.q.put(self.dataset.get_batch())

    def next(self):
        return self.q.get()


class RemoteFetcher:
    """Fetch pre-generated batches from a CPU-node gen_server (gen_server.py).

    The server generates ahead into its own buffer; this client keeps a small
    local queue filled by a background thread, so the train loop waits on
    neither generation nor the network. Batches arrive already padded, in
    seeded generation order -> identical stream to local generation.
    start_index supports --resume: the stream re-positions to exactly the
    batch the resumed step needs (server replay cache / snapshot rewind)."""

    def __init__(self, host: str, port: int = 29700, depth: int = 4, start_index: int = 0):
        from gen_client import RobustFetcher
        self.f = RobustFetcher(host, port, start_index)
        self.q = queue.Queue(maxsize=depth)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            self.q.put(self.f.fetch())

    def next(self):
        return self.q.get()


class PipelinedPrefetcher:
    """Keeps TWO get_batch() calls in flight on the shared pool so batch k+1's
    tasks fill workers idled by batch k's straggler tail (max-of-chains bubble).

    Determinism: parent-RNG consumption (param sampling + per-task seeds) is
    serialized in strict batch order via a ticket; run_parallel_seeded fires
    _ON_DISPATCHED right after submitting tasks, releasing the ticket while the
    batch still executes. Assembly after dispatch is RNG-free (verified), so the
    generated data is bit-identical to the sequential Prefetcher. Batches are
    delivered to the trainer in ticket order."""

    def __init__(self, dataset, depth=8, n_flight=2, snapshot_cb=None):
        global _ON_DISPATCHED
        self.dataset = dataset
        self.q = queue.Queue(maxsize=depth)
        self._cv = threading.Condition()
        self._next_ticket = 0    # next ticket to hand out
        self._sample_turn = 0    # ticket allowed to consume parent RNG
        self._deliver_turn = 0   # ticket allowed to enqueue its result
        self._snapshot_cb = snapshot_cb   # called as cb(ticket) under RNG turn,
                                          # BEFORE the batch consumes parent RNG
        _ON_DISPATCHED = self._on_dispatched
        for _ in range(n_flight):
            threading.Thread(target=self._worker, daemon=True).start()

    def _on_dispatched(self):
        with self._cv:
            self._sample_turn += 1
            self._cv.notify_all()

    def _worker(self):
        while True:
            with self._cv:
                t = self._next_ticket
                self._next_ticket += 1
                while self._sample_turn != t:
                    self._cv.wait()
            # our turn for the parent RNG; the ticket is released inside
            # run_parallel_seeded via _ON_DISPATCHED once tasks are submitted
            if self._snapshot_cb is not None:
                self._snapshot_cb(t)      # RNG state is exact for batch t here
            batch = self.dataset.get_batch()
            with self._cv:
                while self._deliver_turn != t:
                    self._cv.wait()
            self.q.put(batch)
            with self._cv:
                self._deliver_turn += 1
                self._cv.notify_all()

    def next(self):
        return self.q.get()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--micro_batch_size", type=int, default=8)
    ap.add_argument("--lr_muon", type=float, default=8e-4)   # Stage 1 max LR (paper)
    ap.add_argument("--lr_adamw", type=float, default=1e-4)  # v1 precedent; validated with ssmax routing fix (single-lr 8e-4 drove modulator runaway)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--wd_mode", default="cautious", choices=["cautious", "plain"],
                    help="cautious (paper text) | plain (reference default; run-7 fix — "
                         "counterfactual-validated norm equilibrium, see wdtest.tsv)")
    ap.add_argument("--grad_clip", type=float, default=10.0)  # stages 1&2 (paper)
    ap.add_argument("--warmup", type=int, default=50)         # [DEVIATION] short warmup for 1k-step run
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--prior_jobs", type=int, default=24)  # re-tuned with persistent pool + affinity isolation
    ap.add_argument("--pipeline", type=int, default=1, help="1: two batches in flight (bit-identical data); 0: sequential")
    ap.add_argument("--queue_depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--remote_gen", default=None,
                    help="host[:port] of a gen_server; skips all local generation "
                         "(no pool, no affinity split — every core goes to training)")
    ap.add_argument("--compile_blocks", type=int, default=0,
                    help="1: torch.compile the attention blocks (epsilon-class numerics)")
    ap.add_argument("--fp32_ssmax", type=int, default=1,
                    help="1: run QASSMax in fp32 (stability fix for bf16 backward overflow)")
    ap.add_argument("--fp32_col_attn", type=int, default=0,
                    help="1: run col_embedder attention fully in fp32 (escalation)")
    ap.add_argument("--checkpoint_dir", default=None,
                    help="directory for step-tagged checkpoints (repo-compatible format)")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", default=None,
                    help="'auto' (latest ckpt in checkpoint_dir) or an explicit path; "
                         "restores model+optimizer+step and re-positions the data stream")
    cfg = ap.parse_args()

    if cfg.remote_gen is None:
        # Must precede any CUDA/model allocation: fork the pool from a small parent.
        # CPU split: main thread (GPU feeding + torch CPU ops) on low cores,
        # gen workers pinned to the rest and niced. Perf hint only; data unaffected.
        import os as _os
        try:
            n_cores = len(_os.sched_getaffinity(0))
            main_cores = set(range(0, 8)) if n_cores >= 16 else None
            worker_cores = set(range(8, n_cores)) if n_cores >= 16 else None
        except Exception:
            main_cores = worker_cores = None
        install_persistent_pool(cfg.prior_jobs, worker_cores=worker_cores)
        if main_cores:
            try:
                _os.sched_setaffinity(0, main_cores)
            except Exception:
                pass

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Model (v2 defaults). NOTE: resume-load must happen BEFORE compile so the
    # checkpoint's vanilla keys match; compile wraps the loaded weights.
    model = TabICL(max_classes=0 if cfg.regression else 10).to(cfg.device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    muon_p, adamw_p = split_params(model)
    print(f"[model] {n_params/1e6:.2f}M params | Muon tensors={len(muon_p)} "
          f"AdamW tensors={len(adamw_p)} | task={'regression' if cfg.regression else 'classification'}")

    opt = MuonAdamW(muon_p, adamw_p, lr_muon=cfg.lr_muon, lr_adamw=cfg.lr_adamw,
                    weight_decay=cfg.weight_decay)
    opt.wd_mode = cfg.wd_mode

    start_step = 0
    if cfg.resume:
        import os as _os2, glob as _glob
        path = cfg.resume
        if path == "auto":
            cands = sorted(_glob.glob(_os2.path.join(cfg.checkpoint_dir or ".", "step-*.ckpt")),
                           key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
            path = cands[-1] if cands else None
        if path:
            ck = torch.load(path, map_location=cfg.device, weights_only=True)
            model.load_state_dict(ck["state_dict"], strict=True)
            opt.load_state_dict(ck["optimizer_state"])
            for g in opt.param_groups:   # loaded state restores saved wd/lr; CLI wins
                g["weight_decay"] = cfg.weight_decay
                g["lr_muon"] = cfg.lr_muon
                g["lr_adamw"] = cfg.lr_adamw
            start_step = ck["curr_step"]
            print(f"[resume] {path} -> continuing at step {start_step}")
        else:
            print("[resume] no checkpoint found, starting fresh")

    if cfg.fp32_ssmax:
        install_fp32_ssmax()
        print("[model] QASSMax running in fp32 (stability fix)")
    if cfg.fp32_col_attn:
        install_fp32_col_attention(model)
    install_skip_path_dtype_fix()
    if cfg.compile_blocks:
        install_block_compile(model)
        print("[model] attention blocks compiled (dynamic shapes, symbolic _logn)")

    # Data source: remote gen_server (CPU node) or local prior with prefetch.
    # Stream position = start_step (batch i is consumed at step i).
    if cfg.remote_gen:
        host, _, port = cfg.remote_gen.partition(":")
        prefetcher = RemoteFetcher(host, int(port or 29700), depth=4, start_index=start_step)
        print(f"[data] remote generation from {host}:{port or 29700} at batch {start_step}")
    else:
        dataset = build_prior(cfg)
        if cfg.pipeline:
            prefetcher = PipelinedPrefetcher(dataset, depth=cfg.queue_depth)
        else:
            prefetcher = Prefetcher(dataset, depth=cfg.queue_depth)

    # Regression quantile levels (for pinball loss)
    if cfg.regression:
        Q = model.num_quantiles
        alpha = torch.linspace(0.0, 1.0, Q + 2, device=cfg.device)[1:-1]

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    n_micro = cfg.batch_size // cfg.micro_batch_size

    ema_loss = None
    t0 = time.time()
    for step in range(start_step, cfg.steps):
        opt.lr_scale = cosine_lr(step, cfg.steps, cfg.warmup)

        tg = time.time()
        X, y, d, seq_lens, train_sizes = prefetcher.next()
        X = X.to_padded_tensor(0.0) if X.is_nested else X
        y = y.to_padded_tensor(0.0) if y.is_nested else y
        gen_t = time.time() - tg

        opt.zero_grad(set_to_none=True)
        # BIT-EXACT SPEEDUPS (validated): accumulate metrics on-GPU (one sync per
        # step instead of 8x .item()), and keep ONE autocast context across the
        # micro-batch loop so the weight bf16-cast cache survives (same cast
        # values; backward runs with autocast disabled, per torch guidance).
        loss_acc = torch.zeros((), device=cfg.device)
        metric_acc = torch.zeros((), device=cfg.device)

        with amp_ctx:
            for g in range(n_micro):
                sl = slice(g * cfg.micro_batch_size, (g + 1) * cfg.micro_batch_size)
                mb_X, mb_y = X[sl], y[sl]
                ts = int(train_sizes[sl][0].item())     # CPU tensors: no GPU sync
                dd = int(d[sl].max().item())
                mb_X = mb_X[:, :, :dd].to(cfg.device, non_blocking=True)   # uniform real features -> d=None
                mb_y = mb_y.to(cfg.device, non_blocking=True)
                y_tr, y_te = mb_y[:, :ts], mb_y[:, ts:]

                if cfg.regression:
                    mu = y_tr.mean(1, keepdim=True)
                    sd = y_tr.std(1, keepdim=True).clamp_min(1e-6)
                    y_tr_n, y_te_n = (y_tr - mu) / sd, (y_te - mu) / sd
                    pred = model(mb_X, y_tr_n, None)
                    diff = y_te_n.unsqueeze(-1) - pred
                    loss = torch.maximum(alpha * diff, (alpha - 1) * diff).mean()
                    with torch.no_grad():
                        med = pred[..., pred.shape[-1] // 2]
                        ss = ((y_te_n - med) ** 2).sum()
                        tot = ((y_te_n - y_te_n.mean()) ** 2).sum().clamp_min(1e-8)
                        metric = (1 - ss / tot).float()
                else:
                    pred = model(mb_X, y_tr, None)
                    loss = F.cross_entropy(pred.flatten(end_dim=-2), y_te.long().flatten())
                    with torch.no_grad():
                        metric = (pred.argmax(-1) == y_te.long()).float().mean()

                with torch.autocast(device_type="cuda", enabled=False):
                    (loss / n_micro).backward()
                loss_acc += loss.detach().float() / n_micro
                metric_acc += metric / n_micro

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # NON-FINITE GUARD (added after step-19700 NaN incident): a single inf
        # gradient makes the global clip norm non-finite, which poisons ALL
        # gradients and destroys the model in one update. Skip such steps
        # entirely; bit-exact behavior on every healthy step.
        step_loss = loss_acc.item()      # single host sync per step
        step_metric = metric_acc.item()
        norm_val = total_norm.item()
        if math.isfinite(norm_val) and math.isfinite(step_loss):
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            print(f"[guard] NON-FINITE at step {step} (grad_norm={norm_val}, "
                  f"loss={step_loss}) — optimizer step SKIPPED", flush=True)

        if cfg.checkpoint_dir and ((step + 1) % cfg.save_every == 0 or step == cfg.steps - 1):
            path = save_checkpoint(model, opt, step + 1, cfg)
            print(f"[ckpt] saved {path}", flush=True)

        ema_loss = step_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * step_loss
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            sps = (step + 1) / (time.time() - t0)
            mname = "R2" if cfg.regression else "acc"
            mult = icl0_multiplier(model)
            print(f"step {step:4d} | loss {step_loss:.4f} (ema {ema_loss:.4f}) | "
                  f"{mname} {step_metric:.4f} | lr {opt.lr_scale*cfg.lr_muon:.2e} | "
                  f"gen {gen_t:.2f}s | {sps:.2f} step/s | mult {mult:.2f}")

    print(f"\nDone {cfg.steps} steps in {(time.time()-t0)/60:.1f} min. "
          f"final loss {step_loss:.4f}, {('R2' if cfg.regression else 'acc')} {step_metric:.4f}")


if __name__ == "__main__":
    main()
