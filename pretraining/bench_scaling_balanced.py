"""Refine the TabFM-style balanced frontier under 21 days: candidates 530M-800M.
Balanced = embed/cls/depth/ff all grow with the same scale factor (TabFM ratios)."""
import statistics, time
import torch, torch.nn.functional as F
import stage1_train as st
from tabicl._model.tabicl import TabICL
DEV="cuda"; SEQ,FEAT,TS,TOTAL=1024,100,512,64
# (tag, embed, cls, blocks, ff)  — balanced proportions around s=1.73-1.95
CASES=[
 ("1x-anchor",128,4,12,2),
 ("~554M e224c7b22f3",224,7,22,3),
 ("~580M e224c7b23f3",224,7,23,3),
 ("~605M e224c7b24f3",224,7,24,3),
]
def build(embed,cls_,blocks,ff):
    st.install_fp32_ssmax()
    icl=embed*cls_
    m=TabICL(max_classes=10,embed_dim=embed,col_nhead=8,col_num_inds=embed,row_num_cls=cls_,
             row_nhead=8,icl_num_blocks=blocks,icl_nhead=8,ff_factor=ff).to(DEV)
    m.train(); st.install_fp32_col_attention(m); st.install_skip_path_dtype_fix(); return m
def step(model,opt,amp,per):
    micro=TOTAL//per; opt.zero_grad(set_to_none=True)
    with amp:
        for _ in range(micro):
            X=torch.randn(per,SEQ,FEAT,device=DEV); y=torch.randint(0,10,(per,SEQ),device=DEV).float()
            pred=model(X,y[:,:TS],None); loss=F.cross_entropy(pred.flatten(end_dim=-2),y[:,TS:].long().flatten())
            with torch.autocast(device_type="cuda",enabled=False): (loss/micro).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
amp=torch.autocast(device_type="cuda",dtype=torch.bfloat16)
print(f"{'tag':>18} {'params(M)':>10} {'micro':>6} {'step(s)':>9} {'peak(GiB)':>10} {'proj_days':>10}")
base=None
for tag,embed,cls_,blocks,ff in CASES:
    used=None
    for per in (8,4):
        try:
            torch.cuda.empty_cache()
            model=build(embed,cls_,blocks,ff); npar=sum(p.numel() for p in model.parameters())
            mu,ad=st.split_params(model); opt=st.MuonAdamW(mu,ad,lr_muon=8e-4,lr_adamw=1e-4,weight_decay=0.1); opt.lr_scale=1.0
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3): step(model,opt,amp,per)
            torch.cuda.synchronize(); ts=[]
            for _ in range(5):
                t0=time.time(); step(model,opt,amp,per); torch.cuda.synchronize(); ts.append(time.time()-t0)
            used=(per,statistics.median(ts),torch.cuda.max_memory_allocated()/2**30,npar)
            del model,opt; torch.cuda.empty_cache(); break
        except torch.cuda.OutOfMemoryError:
            try: del model,opt
            except Exception: pass
            torch.cuda.empty_cache(); continue
    if used:
        per,stime,peak,npar=used
        if base is None: base=stime
        print(f"{tag:>18} {npar/1e6:>10.1f} {per:>6} {stime:>9.3f} {peak:>10.1f} {7.27*stime/base:>10.1f}",flush=True)
    else:
        print(f"{tag:>18}  OOM at micro 4",flush=True)
