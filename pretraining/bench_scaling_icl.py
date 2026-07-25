"""Find the largest Stage-1-trainable model under ~21 H100-days.

Strategy under test: freeze the col/row front-end at 1x dims (it's 5% of params
and its cost scales with rows*features, not with ICL size) and scale ONLY the
ICL transformer via row_num_cls (icl_dim = 128*cls), depth, ff_factor.
icl_nhead is set to icl_dim//128 (head_dim 128, flash-friendly).

For each config: try micro 8 plain; if OOM, micro 8 with per-block activation
checkpointing on the ICL blocks; then micro 4 plain; micro 4 ckpt.
Report best step time. proj_days = 7.27 * step/base_step (anchored to the
measured 1x production run). Budget line: step <= 7.56s -> < 21 days.
"""
import statistics, time
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import stage1_train as st
from tabicl._model.tabicl import TabICL

DEV="cuda"; SEQ,FEAT,TS,TOTAL=1024,100,512,64

# (tag, cls, blocks, ff)
CASES=[
 ("1x-anchor",   4, 12, 2),
 ("1.46B c20b22f3", 20, 22, 3),
 ("1.53B c20b23f3", 20, 23, 3),
 ("1.44B c16b34f3", 16, 34, 3),
]

def build(cls_, blocks, ff):
    st.install_fp32_ssmax()
    icl_dim = 128 * cls_
    m = TabICL(max_classes=10, embed_dim=128, col_nhead=8, col_num_inds=128,
               row_num_cls=cls_, row_nhead=8, icl_num_blocks=blocks,
               icl_nhead=max(8, icl_dim // 128), ff_factor=ff).to(DEV)
    m.train()
    st.install_fp32_col_attention(m)
    st.install_skip_path_dtype_fix()
    return m

def enable_icl_ckpt(model):
    blocks = model.icl_predictor.tf_icl.blocks
    for blk in blocks:
        orig = blk.forward
        def make(f):
            def fwd(*a, **k):
                return checkpoint(f, *a, use_reentrant=False, **k)
            return fwd
        blk.forward = make(orig)

def step(model, opt, amp, per):
    micro = TOTAL // per
    opt.zero_grad(set_to_none=True)
    with amp:
        for _ in range(micro):
            X = torch.randn(per, SEQ, FEAT, device=DEV)
            y = torch.randint(0, 10, (per, SEQ), device=DEV).float()
            pred = model(X, y[:, :TS], None)
            loss = F.cross_entropy(pred.flatten(end_dim=-2), y[:, TS:].long().flatten())
            with torch.autocast(device_type="cuda", enabled=False):
                (loss / micro).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()

def timeit(model, opt, amp, per):
    for _ in range(3): step(model, opt, amp, per)
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.time(); step(model, opt, amp, per); torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return statistics.median(ts)

amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
print(f"{'tag':>16} {'params(M)':>10} {'variant':>10} {'step(s)':>9} {'peak(GiB)':>10} {'proj_days':>10}")
base_t = None
for tag, cls_, blocks, ff in CASES:
    results = []
    for use_ckpt in (False, True):
        for per in (8, 4):
            try:
                torch.cuda.empty_cache()
                model = build(cls_, blocks, ff)
                if use_ckpt:
                    enable_icl_ckpt(model)
                npar = sum(p.numel() for p in model.parameters())
                mu, ad = st.split_params(model)
                opt = st.MuonAdamW(mu, ad, lr_muon=8e-4, lr_adamw=1e-4, weight_decay=0.1)
                opt.lr_scale = 1.0
                torch.cuda.reset_peak_memory_stats()
                t = timeit(model, opt, amp, per)
                peak = torch.cuda.max_memory_allocated() / 2**30
                results.append((t, per, use_ckpt, peak, npar))
                del model, opt
                torch.cuda.empty_cache()
                break   # first fitting micro for this ckpt-mode is the fastest
            except torch.cuda.OutOfMemoryError:
                try: del model, opt
                except Exception: pass
                torch.cuda.empty_cache()
                continue
    if not results:
        print(f"{tag:>16} {'--':>10} {'ALL OOM':>10}", flush=True)
        continue
    t, per, ck, peak, npar = min(results)
    if base_t is None:
        base_t = t
    variant = f"m{per}{'+ckpt' if ck else ''}"
    print(f"{tag:>16} {npar/1e6:>10.1f} {variant:>10} {t:>9.3f} {peak:>10.1f} {7.27*t/base_t:>10.1f}", flush=True)
