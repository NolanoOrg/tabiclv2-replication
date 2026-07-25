# Single-GPU TabICLv2 pretraining harness

The scripts that trained the replication checkpoints at
[huggingface.co/ayushkaushal4/tabiclv2-replication](https://huggingface.co/ayushkaushal4/tabiclv2-replication),
built on the pre-release TabICLv2 training code from
[PR #111](https://github.com/soda-inria/tabicl/pull/111) (branch `prior_v2`).

Stage 1 runs at 1.27 s/step on one H100 (~7 days for the 500K-step recipe).

## Files

| file | what |
|---|---|
| `stage1_train.py` | Stage 1: 500K steps, 1,024 rows/dataset, LR 8e-4. Also hosts the shared model patches and prior utilities imported by stages 2-3. |
| `stage2_train.py` | Stage 2: 40K steps, 400-10,240 rows log-uniform, LR 1e-4. Row-budget micro-batch packing for variable lengths. |
| `stage3_train.py` | Stage 3: 10K steps, 400-60,000 rows log-uniform, LR 2e-5, grad clip 1.0. Adds activation checkpointing above 20K rows. |
| `gen_server.py` / `gen_client.py` | Prior generation on a separate CPU node (`--stage {1,2,3}`); deterministic stream with seek/replay, so restarts consume bit-identical data. |
| `watchdog_*.sh` | Auto-restart with resume plus health halting (loss-EMA / ssmax-multiplier / NaN). |
| `eval_suites.py`, `eval_context_scaling.py` | Evaluation used for the model card: small/large OpenML suites (stage-2 vs stage-3 vs released v2), and accuracy vs in-context rows on large datasets. |
| `bench_scaling_icl.py`, `bench_scaling_balanced.py` | Step-time profiling (not training) of larger stage-1 configs. Projected to 500K steps on one H100: ~1.5B ICL-only scaling (`row_num_cls 20`, 23 ICL blocks, `ff_factor 3`, front-end at base dims, activation checkpointing) ~20.6 days; ~0.6B proportional scaling (`embed_dim 224`, `col_num_inds 224`, `row_num_cls 7`, 24 ICL blocks, `ff_factor 3`) ~19.9 days. |

## Running

```bash
# on the CPU node
PYTHONHASHSEED=0 python pretraining/gen_server.py --stage 1 --port 29700 --seed 42 --jobs 60

# on the GPU node
PYTHONHASHSEED=0 python -u pretraining/stage1_train.py \
    --steps 500000 --batch_size 64 --micro_batch_size 8 \
    --compile_blocks 1 --fp32_ssmax 1 --fp32_col_attn 1 \
    --wd_mode plain --weight_decay 0.1 \
    --remote_gen <cpu-host>:29700 \
    --checkpoint_dir <dir> --save_every 5000 --resume auto
```

Drop `--remote_gen` to generate locally instead (uses a persistent pooled
generator; `--prior_jobs` sets the worker count). Stages 2 and 3 take
`--init_from <previous stage checkpoint>`. Paths default to this directory;
override with the `TABICL_RUN_DIR` (Python) or `RUN_DIR` / `REPO_DIR` (shell)
environment variables. The generator host defaults are LAN addresses from our
setup - pass your own.

## Notes

- The model-level changes are runtime patches applied by `stage1_train.py`. `src/` is unmodified, so importing `tabicl` directly gives stock behavior.
- Classification only (`max_classes=10`); the regressor was not trained.
- Hyperparameters deviate from the official recipe in several places - see the model card for the list.

