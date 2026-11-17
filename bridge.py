#!/usr/bin/env python3
"""
Laptop <-> Arduino Nano ESP32 bridge. Run next to sid.py in speaker_demo.
The laptop records and makes the log-mel; the BOARD runs the neural network.

  pip install pyserial
  python bridge.py check     --port COM5     # board vs Python embeddings + speed
  python bridge.py confusion --port COM5     # held-out clips through the board
  python bridge.py live      --port COM5 --device 1
"""
import argparse, struct, sys, time
from pathlib import Path
import numpy as np
import sid


class Board:
    def __init__(self, port, baud=921600):
        import serial
        self.ser = serial.Serial(port, baud, timeout=15)
        t0 = time.time()
        while True:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == "READY":
                break
            if line.startswith("INIT FAIL"):
                sys.exit("Board says: " + line)
            if time.time() - t0 > 20:
                sys.exit("No READY from board. Close Serial Monitor, check the port, press the reset button.")
        print("Board ready.")

    def embed(self, window):
        feats = sid.log_mel(sid.normalise_level(window)).astype("<f4")
        self.ser.reset_input_buffer()
        self.ser.write(b"ST" + np.ascontiguousarray(feats).tobytes())
        head = self.ser.read(2)
        if head != b"OK":
            raise RuntimeError(f"Board reply was {head!r}, expected b'OK'")
        us = struct.unpack("<I", self.ser.read(4))[0]
        e = np.frombuffer(self.ser.read(256), dtype="<f4").astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9), us


def cmd_check(a):
    board, py = Board(a.port), sid.Embedder()
    cos, lat = [], []
    for f in sorted(Path("data/test").rglob("*.wav")):
        for w in sid.make_windows(sid.load_audio(f))[: a.per_clip]:
            eb, us = board.embed(w)
            cos.append(float(eb @ py(w)))
            lat.append(us / 1000)
    cos, lat = np.array(cos), np.array(lat)
    print(f"\nwindows compared: {len(cos)}")
    print(f"cosine(board, Python): mean {cos.mean():.5f}  min {cos.min():.5f}")
    print(f"board inference time per window: mean {lat.mean():.1f} ms  max {lat.max():.1f} ms")
    print("Real-time budget: under 600 ms (the window hop).")


def cmd_confusion(a):
    board = Board(a.port)
    names, C = sid.load_db()
    root = sid.DATA / a.set
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    rows, cols = [p.name for p in folders], names + ["UNKNOWN"]
    cm = np.zeros((len(rows), len(cols)), dtype=int)
    times = []
    for ri, p in enumerate(folders):
        for f in sid.wavs_in(p):
            wins = sid.make_windows(sid.load_audio(f))
            if not wins:
                continue
            res = [board.embed(w) for w in wins]
            times += [us / 1000 for _, us in res]
            scores = C @ sid.unit(np.mean([e for e, _ in res], axis=0))
            label, _ = sid.decide(scores, names, a.threshold, a.margin)
            cm[ri, cols.index(label)] += 1
    w = max(len(x) for x in rows + cols) + 2
    print("\n" + "true \\ pred".ljust(w) + "".join(c.rjust(w) for c in cols))
    for ri, r in enumerate(rows):
        print(r.ljust(w) + "".join(str(v).rjust(w) for v in cm[ri]))
    tot = ok = 0
    for ri, r in enumerate(rows):
        want = r if r in names else "UNKNOWN"
        n, c = int(cm[ri].sum()), int(cm[ri, cols.index(want)])
        tot += n; ok += c
        print(f"{r:<14} {c}/{n} correct")
    print(f"\nOverall with the board in the loop: {ok}/{tot} ({ok / max(tot, 1):.1%})")
    print(f"Board inference per window: mean {np.mean(times):.1f} ms")


def cmd_live(a):
    import sounddevice as sd
    board = Board(a.port)
    names, C = sid.load_db()
    print("Listening (1.2 s at a time). Ctrl+C to stop.")
    try:
        while True:
            audio = sd.rec(sid.WIN, samplerate=sid.SR, channels=1, dtype="float32", device=a.device)
            sd.wait()
            w = audio[:, 0]
            if sid.dbfs(w) < a.gate:
                print("(silence)")
                continue
            e, us = board.embed(w)
            scores = C @ e
            label, best = sid.decide(scores, names, a.threshold, a.margin)
            print(f"{label:<10} best={best:.2f}  board {us / 1000:.0f} ms   " +
                  "  ".join(f"{n}={s:.2f}" for n, s in zip(names, scores)))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("check", cmd_check), ("confusion", cmd_confusion), ("live", cmd_live)]:
        p = sub.add_parser(name)
        p.add_argument("--port", required=True, help="e.g. COM5")
        p.set_defaults(fn=fn)
        if name == "check":
            p.add_argument("--per-clip", type=int, default=3)
        if name == "confusion":
            p.add_argument("--set", default="test")
            p.add_argument("--threshold", type=float, default=0.0)
            p.add_argument("--margin", type=float, default=0.0)
        if name == "live":
            p.add_argument("--device", type=int, default=None)
            p.add_argument("--threshold", type=float, default=0.60)
            p.add_argument("--margin", type=float, default=0.0)
            p.add_argument("--gate", type=float, default=-45.0)
    a = ap.parse_args()
    a.fn(a)
