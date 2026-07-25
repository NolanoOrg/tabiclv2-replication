"""Prefetching batch-generation server v2 (runs on the CPU node).

Generation runs CONTINUOUSLY ahead of demand into a buffer. Requests pop
pre-generated batches — zero generation wait on the request path.

Crash-safe, stream-exact resume:
  - np.random state is snapshotted at every batch boundary (under the ticket
    lock, so the state for batch i is exact) into an on-disk ring
    (~2.5KB/entry, last 20000 kept).
  - a replay cache holds the last REPLAY_N served batches in RAM for instant
    small rewinds (trainer blips).
  - big rewinds re-exec the server with --resume_state/--resume_idx: clean
    slate, RNG restored, stream regenerated from exactly batch idx.

Protocol (TCP, length-prefixed):
  client -> b"SEEK" + uint64 idx        (once, after connect)
  server -> b"OKAY"                     (ready at idx; may take time on rewind)
  client -> b"GETB"                     (repeat)
  server -> uint64 len + torch.save bytes of (X, y, d, seq_lens, train_sizes)
  client -> b"QUIT"

Usage:
  PYTHONHASHSEED=0 python gen_server.py --port 29700 --seed 42 --jobs 60 --depth 16
"""
import argparse
import io
import os
import pickle
import socket
import struct
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

import stage1_train as st

REPLAY_N = 64          # batches kept in RAM for instant rewind
SNAP_KEEP = 600000     # keep ALL of stage 1 (~1.5GB RAM; rewinds can span the whole run)
SNAP_FLUSH = 50        # persist snapshot ring to disk every N batches


class Cfg:
    regression = False
    batch_size = 64
    micro_batch_size = 8
    prior_jobs = 60
    stage = 1


def pad_batch(b):
    if Cfg.stage >= 2:
        # variable seq lens: ship per-dataset tensors (no cross-dataset padding;
        # a padded [64, 10240, 100] batch would be ~1GB on the wire)
        X, y, d, sl, ts = b
        Xs = list(X.unbind()) if X.is_nested else list(X)
        ys = list(y.unbind()) if y.is_nested else list(y)
        return Xs, ys, d, sl, ts
    X, y, d, sl, ts = b
    X = X.to_padded_tensor(0.0) if X.is_nested else X
    y = y.to_padded_tensor(0.0) if y.is_nested else y
    return X, y, d, sl, ts


def serialize(batch):
    buf = io.BytesIO()
    torch.save(batch, buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=29700)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--state_file", default="/tmp/gen_state.pkl")
    ap.add_argument("--resume_idx", type=int, default=None)
    ap.add_argument("--resume_state", default=None)
    ap.add_argument("--resume_from_statefile", action="store_true",
                    help="crash recovery: resume at the newest snapshot in --state_file "
                         "(clients SEEK to what they need; older indices trigger rewind)")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                    help="prior curriculum stage (2: 400-10240 rows log-uniform, "
                         "80%% train, micro group=1, list serialization)")
    cfg = ap.parse_args()
    Cfg.stage = cfg.stage
    if cfg.stage >= 2:
        # groups of 8 share a sampled seq len -> trainer can batch datasets
        # within a group without row padding (splits long groups by row budget)
        Cfg.micro_batch_size = 8

    if cfg.resume_from_statefile and os.path.exists(cfg.state_file):
        with open(cfg.state_file, "rb") as f:
            ring = pickle.load(f)["snapshots"]
        idx = max(ring)
        tmp = f"/tmp/gen_recover_{idx}.pkl"
        with open(tmp, "wb") as f:
            pickle.dump(ring[idx], f)
        cfg.resume_idx, cfg.resume_state = idx, tmp
        print(f"[server] crash recovery from statefile: batch {idx}", flush=True)

    Cfg.prior_jobs = cfg.jobs
    st.install_persistent_pool(cfg.jobs)

    snapshots = OrderedDict()   # batch_idx -> np.random state tuple
    if os.path.exists(cfg.state_file):
        try:
            with open(cfg.state_file, "rb") as f:
                for k, v in sorted(pickle.load(f)["snapshots"].items()):
                    snapshots[k] = v     # inherit prior ring: keeps old rewind targets
        except Exception:
            pass
    base_idx = 0

    if cfg.resume_idx is not None and cfg.resume_state:
        with open(cfg.resume_state, "rb") as f:
            state = pickle.load(f)
        np.random.set_state(state)
        base_idx = cfg.resume_idx
        print(f"[server] RESUMED stream at batch {base_idx}", flush=True)
    else:
        np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ds = st.build_prior(Cfg())

    # snapshot hook: called under the ticket lock, right before batch t's
    # params/seeds are drawn -> state is exact for regenerating batch t
    def snapshot_cb(ticket):
        idx = base_idx + ticket
        snapshots[idx] = np.random.get_state()
        while len(snapshots) > SNAP_KEEP:
            snapshots.popitem(last=False)
        if idx % SNAP_FLUSH == 0:
            tmp = cfg.state_file + ".tmp"
            disk_ring = {k: v for k, v in snapshots.items() if k % SNAP_FLUSH == 0}
            with open(tmp, "wb") as f:
                pickle.dump({"snapshots": disk_ring, "seed": cfg.seed}, f)
            os.replace(tmp, cfg.state_file)

    pf = st.PipelinedPrefetcher(ds, depth=cfg.depth, snapshot_cb=snapshot_cb)
    print(f"[server] generating from batch {base_idx}: jobs={cfg.jobs} depth={cfg.depth} seed={cfg.seed}", flush=True)
    t0 = time.time()
    while pf.q.qsize() < min(cfg.depth, 8) and time.time() - t0 < 180:
        time.sleep(1)
    print(f"[server] buffer primed: {pf.q.qsize()}/{cfg.depth}", flush=True)

    replay = OrderedDict()      # batch_idx -> serialized bytes
    next_idx = base_idx         # index of next batch to pop from pf

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", cfg.port))
    srv.listen(4)
    print(f"[server] listening on :{cfg.port}", flush=True)

    def rewind_reexec(idx):
        """Re-exec the server to serve from batch idx (state from snapshot)."""
        state = snapshots.get(idx)
        if state is None:   # nearest snapshot at-or-before idx (client fast-forwards the gap)
            cands = [k for k in snapshots if k <= idx]
            if cands:
                idx = max(cands)
                state = snapshots[idx]
        if state is None:
            print(f"[server] FATAL: no snapshot at or before batch {idx}", flush=True)
            sys.exit(2)
        tmp = f"/tmp/gen_rewind_{idx}.pkl"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        print(f"[server] REWIND to batch {idx}: re-exec", flush=True)
        os.execv(sys.executable, [sys.executable, "-u", os.path.abspath(__file__),
                                  "--port", str(cfg.port), "--seed", str(cfg.seed),
                                  "--jobs", str(cfg.jobs), "--depth", str(cfg.depth),
                                  "--stage", str(cfg.stage),
                                  "--state_file", cfg.state_file,
                                  "--resume_idx", str(idx), "--resume_state", tmp])

    n_served = 0
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            hdr = conn.recv(12, socket.MSG_WAITALL)
            if len(hdr) != 12 or hdr[:4] != b"SEEK":
                conn.close()
                continue
            (want,) = struct.unpack("!Q", hdr[4:])
            print(f"[server] client {addr} SEEK {want} (next={next_idx})", flush=True)
            if want > next_idx:
                # fast-forward: generate and discard
                while next_idx < want:
                    replay[next_idx] = serialize(pad_batch(pf.next()))
                    while len(replay) > REPLAY_N:
                        replay.popitem(last=False)
                    next_idx += 1
            elif want < next_idx and want not in replay:
                conn.close()
                rewind_reexec(want)     # never returns
            conn.sendall(b"OKAY")
            serve_idx = want
            while True:
                cmd = conn.recv(4, socket.MSG_WAITALL)
                if cmd != b"GETB":
                    break
                if serve_idx in replay:
                    data = replay[serve_idx]
                else:
                    assert serve_idx == next_idx, (serve_idx, next_idx)
                    data = serialize(pad_batch(pf.next()))
                    replay[next_idx] = data
                    while len(replay) > REPLAY_N:
                        replay.popitem(last=False)
                    next_idx += 1
                conn.sendall(struct.pack("!Q", len(data)))
                conn.sendall(data)
                serve_idx += 1
                n_served += 1
                if n_served % 500 == 0:
                    print(f"[server] served {n_served} (next={next_idx}, buffer={pf.q.qsize()})", flush=True)
        except (ConnectionResetError, BrokenPipeError, AssertionError) as e:
            print(f"[server] client error: {e}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print(f"[server] client {addr} disconnected (served {n_served} total)", flush=True)


if __name__ == "__main__":
    main()
