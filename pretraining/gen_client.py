"""Client for the CPU-node batch server v2 (SEEK protocol + auto-reconnect).

RobustFetcher tracks the next batch index; on any connection failure it
reconnects and SEEKs to exactly that index — the stream continues bit-exact
across server restarts, rewinds, and network blips.
"""
import hashlib
import io
import socket
import struct
import time

import torch

REF_MD5 = [
    "be29bd3afd348ca67130929837929bfe",
    "fd94aed3a6b84ac8e4c8c5c30d0d6e0e",
    "91613e85e835b9f41f67b67e2553a9a3",
]


def _connect_seek(host, port, idx, timeout_s=900):
    """Connect and SEEK, retrying until the server is up (covers re-exec/rewind
    regeneration windows)."""
    deadline = time.time() + timeout_s
    delay = 2.0
    while True:
        try:
            s = socket.create_connection((host, port), timeout=30)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(600)   # a fetch may wait on a rewind regeneration
            s.sendall(b"SEEK" + struct.pack("!Q", idx))
            ok = s.recv(4, socket.MSG_WAITALL)
            if ok == b"OKAY":
                return s
            s.close()
        except OSError:
            pass
        if time.time() > deadline:
            raise ConnectionError(f"gen server unreachable for {timeout_s}s")
        time.sleep(delay)
        delay = min(delay * 1.5, 20)


def _recv_batch(s):
    hdr = s.recv(8, socket.MSG_WAITALL)
    if len(hdr) != 8:
        raise ConnectionError("short header")
    (n,) = struct.unpack("!Q", hdr)
    chunks, got = [], 0
    while got < n:
        c = s.recv(min(1 << 22, n - got))
        if not c:
            raise ConnectionError("server closed")
        chunks.append(c)
        got += len(c)
    return torch.load(io.BytesIO(b"".join(chunks)), weights_only=True)


class RobustFetcher:
    """Sequential batch stream from a gen_server, resumable at any index."""

    def __init__(self, host, port=29700, start_index=0):
        self.host, self.port = host, port
        self.next_idx = start_index
        self.sock = _connect_seek(host, port, start_index)

    def fetch(self):
        for attempt in range(100):
            try:
                self.sock.sendall(b"GETB")
                b = _recv_batch(self.sock)
                self.next_idx += 1
                return b
            except (OSError, ConnectionError):
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = _connect_seek(self.host, self.port, self.next_idx)
        raise ConnectionError("fetch failed after 100 reconnects")

    def close(self):
        try:
            self.sock.sendall(b"QUIT")
            self.sock.close()
        except OSError:
            pass


# compatibility shims for older callers/tests
def connect(host, port=29700, start_index=0):
    return _connect_seek(host, port, start_index)


def fetch_batch(s):
    s.sendall(b"GETB")
    return _recv_batch(s)


def md5_batch(b):
    h = hashlib.md5()
    for t in b:
        h.update(t.numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.0.0.7")
    ap.add_argument("--port", type=int, default=29700)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    f = RobustFetcher(a.host, a.port, a.start)
    for i in range(a.n):
        t = time.time()
        b = f.fetch()
        idx = a.start + i
        note = f" md5={md5_batch(b)}"
        if idx < len(REF_MD5):
            note += " MATCH" if md5_batch(b) == REF_MD5[idx] else " *** MISMATCH ***"
        print(f"batch {idx}: {time.time()-t:.2f}s{note}", flush=True)
    f.close()
