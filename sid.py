#!/usr/bin/env python3
"""
Speaker fingerprint (voiceprint) demo using the pretrained tiny model from
github.com/tduffy000/ccny-masters-thesis  (no training, only enrollment).

Pipeline:
  audio -> 1.2 s windows -> level-normalise -> log-mel (40 x 121)
        -> tiny TFLite model -> 64-d embedding -> L2-normalise
        -> average per person = fingerprint -> cosine similarity at runtime

Commands (run from the folder that contains this file and model.tflite):
  python sid.py devices
  python sid.py record  --name shashank --session 1            # enrollment audio
  python sid.py record  --name shashank --session 2 --set test # held-out audio
  python sid.py enroll
  python sid.py live
  python sid.py test --name shashank --genuine data/test/shashank --impostor data/test/nischal
  python sid.py confusion                                      # who-is-who matrix over data/test/*
  python sid.py export                                         # C arrays for the Arduino
"""
import argparse
import json
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- settings
# These MUST match how the model was trained (see the repo's tensorflow/features.py)
SR = 16000
WIN = int(1.2 * SR)          # 19200 samples -> 121 frames with hop 160
HOP = int(0.6 * SR)          # 50 % overlap between windows
N_FFT, WIN_LEN, HOP_LEN, N_MELS = 512, 400, 160, 40
TARGET_DBFS = -26.0          # every window is scaled to this RMS level
REL_SILENCE_DB = 20.0        # drop windows this far below the loudest one
ABS_FLOOR_DBFS = -60.0       # drop windows that are basically digital silence

MODEL_PATH = Path("model.tflite")           # copy quant_model.tflite here
DATA = Path("data")
ENROLL_FILE = Path("fingerprints.json")

DEFAULT_THRESHOLD = 0.70     # placeholder: calibrate with the `test` command
DEFAULT_MARGIN = 0.05        # best must beat runner-up by this much

PROMPTS = [
    ("normal",   "Speak NORMALLY. Read anything aloud or talk about your day."),
    ("loud",     "Speak LOUDLY, like calling someone across a room."),
    ("soft",     "Speak SOFTLY, like it is late at night."),
    ("fast",     "Speak FAST, like you are in a hurry."),
    ("slow",     "Speak SLOWLY and clearly."),
    ("happy",    "Sound HAPPY / excited, like sharing good news."),
    ("sad",      "Sound SAD or TIRED, low energy."),
    ("question", "Ask lots of QUESTIONS out loud (rising tone)."),
]


# ---------------------------------------------------------------- audio utils
def load_audio(path):
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def dbfs(x):
    return 20 * np.log10(rms(x) + 1e-12)


def make_windows(y):
    """Cut a recording into 1.2 s windows, drop silent ones."""
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)), mode="edge")
    wins = [y[s:s + WIN] for s in range(0, len(y) - WIN + 1, HOP)]
    levels = np.array([dbfs(w) for w in wins])
    keep = (levels > levels.max() - REL_SILENCE_DB) & (levels > ABS_FLOOR_DBFS)
    return [w for w, k in zip(wins, keep) if k]


def normalise_level(w):
    return w * (10 ** (TARGET_DBFS / 20) / rms(w))


def log_mel(w):
    import librosa
    m = librosa.feature.melspectrogram(
        y=w, sr=SR, n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_LEN,
        n_mels=N_MELS, power=2.0)
    return np.log10(m + 1e-6).astype(np.float32)      # (40, 121)


# ---------------------------------------------------------------- model
class Embedder:
    def __init__(self, path=MODEL_PATH):
        if not Path(path).exists():
            sys.exit(f"Model not found: {path}\n"
                     "Copy tensorflow/frozen_models/tiny/1605647312/quant_model.tflite "
                     "from the repo into this folder and rename it model.tflite")
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        shape = tuple(self.inp["shape"][1:])
        if shape != (N_MELS, 121):
            sys.exit(f"Unexpected model input shape {shape}, expected (40, 121). "
                     "Wrong model file?")

    def __call__(self, window):
        x = log_mel(normalise_level(window))[None].astype(np.float32)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        e = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


def embed_file(emb, path):
    """All window embeddings of one file -> array (n_windows, dim)."""
    wins = make_windows(load_audio(path))
    if not wins:
        return np.zeros((0, 64), dtype=np.float32)
    return np.stack([emb(w) for w in wins])


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def wavs_in(folder):
    return sorted(Path(folder).rglob("*.wav"))


# ---------------------------------------------------------------- commands
def cmd_devices(_):
    import sounddevice as sd
    print(sd.query_devices())
    print("\nUse the index of your microphone with --device N")


def cmd_record(a):
    import sounddevice as sd
    import soundfile as sf
    out = DATA / a.set / a.name / f"session{a.session}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Recording for '{a.name}', session {a.session} -> {out}")
    print(f"{len(PROMPTS)} prompts x {a.seconds} s. Keep the mic where you will use it later.\n")
    for label, text in PROMPTS:
        while True:
            print(f"[{label}] {text}")
            for n in (3, 2, 1):
                print(f"  starting in {n}...", end="\r")
                time.sleep(1)
            print("  RECORDING...        ")
            audio = sd.rec(int(a.seconds * SR), samplerate=SR, channels=1,
                           dtype="float32", device=a.device)
            sd.wait()
            audio = audio[:, 0]
            peak = float(np.max(np.abs(audio)))
            print(f"  level {dbfs(audio):.1f} dBFS, peak {peak:.2f}")
            if peak < 0.02:
                print("  Too quiet / mic not picking up. Check device or move closer.")
            elif peak > 0.98:
                print("  Clipping! Move back from the mic or lower input volume.")
            if (peak < 0.02 or peak > 0.98) and input("  Redo? [Y/n] ").strip().lower() != "n":
                continue
            sf.write(out / f"{label}.wav", audio, SR, subtype="PCM_16")
            break
    print("\nDone. Saved to", out)


def cmd_enroll(_):
    emb = Embedder()
    root = DATA / "enroll"
    people = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not people:
        sys.exit(f"No folders in {root}. Run `record` first.")
    db = {"model": str(MODEL_PATH), "people": {}}
    for p in people:
        vecs = []
        for f in wavs_in(p):
            e = embed_file(emb, f)
            if len(e):
                vecs.append(e)
        if not vecs:
            print(f"{p.name}: no usable audio, skipped")
            continue
        E = np.concatenate(vecs)
        centroid = unit(E.mean(axis=0))
        agree = E @ centroid
        db["people"][p.name] = {"centroid": centroid.tolist()}
        print(f"{p.name:<12} windows={len(E):<4} agreement with fingerprint: "
              f"mean={agree.mean():.3f} min={agree.min():.3f}")
    ENROLL_FILE.write_text(json.dumps(db))
    print(f"\nSaved fingerprints -> {ENROLL_FILE}")


def load_db():
    if not ENROLL_FILE.exists():
        sys.exit("No fingerprints.json. Run `python sid.py enroll` first.")
    people = json.loads(ENROLL_FILE.read_text())["people"]
    names = list(people)
    C = np.stack([np.array(people[n]["centroid"], dtype=np.float32) for n in names])
    return names, C


def decide(scores, names, thr, margin):
    order = np.argsort(scores)[::-1]
    best = float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else -1.0
    if best < thr or (best - second) < margin:
        return "UNKNOWN", best
    return names[order[0]], best


def cmd_live(a):
    import sounddevice as sd
    emb = Embedder()
    names, C = load_db()
    q = queue.Queue()
    recent = deque(maxlen=a.smooth)
    buf = np.zeros(0, dtype=np.float32)

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    print(f"Enrolled: {', '.join(names)} | threshold={a.threshold} margin={a.margin}")
    print("Listening... speak normally. Ctrl+C to stop.\n")
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=a.device, callback=cb):
            while True:
                buf = np.concatenate([buf, q.get()])
                if len(buf) < WIN:
                    continue
                w, buf = buf[:WIN], buf[HOP:]
                if dbfs(w) < a.gate:
                    recent.clear()
                    print(f"{'(silence)':<10} level {dbfs(w):.0f} dBFS")
                    continue
                recent.append(C @ emb(w))
                avg = np.mean(recent, axis=0)
                label, best = decide(avg, names, a.threshold, a.margin)
                detail = "  ".join(f"{n}={s:.2f}" for n, s in zip(names, avg))
                print(f"{label:<10} best={best:.2f}   {detail}")
    except KeyboardInterrupt:
        print("\nStopped.")


def file_scores(emb, folder, centroid):
    """One score per file = cosine(mean window embedding, centroid)."""
    scores = []
    for f in wavs_in(folder):
        e = embed_file(emb, f)
        if len(e):
            scores.append((f.name, float(unit(e.mean(axis=0)) @ centroid)))
    return scores


def cmd_test(a):
    emb = Embedder()
    names, C = load_db()
    if a.name not in names:
        sys.exit(f"'{a.name}' not enrolled. Enrolled: {names}")
    c = C[names.index(a.name)]

    gen = []
    for fol in a.genuine:
        gen += file_scores(emb, fol, c)
    imp = []
    for fol in a.impostor or []:
        imp += file_scores(emb, fol, c)
    if not gen:
        sys.exit("No genuine test clips found. Record a held-out session with --set test.")

    print(f"\nGENUINE ({a.name}) - {len(gen)} clips")
    for n, s in gen:
        print(f"  {s:.3f}  {n}")
    g = np.array([s for _, s in gen])
    print(f"  mean={g.mean():.3f} min={g.min():.3f}")
    if imp:
        print(f"\nIMPOSTOR - {len(imp)} clips")
        for n, s in imp:
            print(f"  {s:.3f}  {n}")
        i = np.array([s for _, s in imp])
        print(f"  mean={i.mean():.3f} max={i.max():.3f}")

        print("\nthreshold   FRR (you rejected)   FAR (stranger accepted)")
        best_gap, eer_t, eer = 9, 0, 0
        for t in np.arange(0.30, 0.96, 0.01):
            frr = float(np.mean(g < t))
            far = float(np.mean(i >= t))
            if abs(frr - far) < best_gap:
                best_gap, eer_t, eer = abs(frr - far), t, (frr + far) / 2
            if abs((t * 20) - round(t * 20)) < 1e-6:      # print every 0.05
                print(f"  {t:.2f}        {frr:6.1%}               {far:6.1%}")
        print(f"\nApprox EER = {eer:.1%} at threshold {eer_t:.2f}")
        print("Pick a threshold between impostor max and genuine min if they do not overlap.")
    else:
        print("\n(no --impostor folder given, only genuine scores shown)")


def cmd_confusion(a):
    """Classify every clip in data/<set>/<folder>/ and print a confusion matrix.
    A folder named after an enrolled speaker should be recognised as that person.
    Any other folder (a stranger) should come out as UNKNOWN."""
    emb = Embedder()
    names, C = load_db()
    root = DATA / a.set
    if not root.exists():
        sys.exit(f"No folder {root}. Record held-out audio with --set {a.set} first.")
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        sys.exit(f"{root} has no speaker folders.")
    rows = [p.name for p in folders]
    cols = names + ["UNKNOWN"]
    cm = np.zeros((len(rows), len(cols)), dtype=int)
    for ri, p in enumerate(folders):
        for f in wavs_in(p):
            e = embed_file(emb, f)
            if not len(e):
                continue
            scores = C @ unit(e.mean(axis=0))
            label, _ = decide(scores, names, a.threshold, a.margin)
            cm[ri, cols.index(label)] += 1

    w = max(len(x) for x in rows + cols + ["true \\ predicted"]) + 2
    print()
    print("true \\ predicted".ljust(w) + "".join(c.rjust(w) for c in cols))
    for ri, r in enumerate(rows):
        print(r.ljust(w) + "".join(str(v).rjust(w) for v in cm[ri]))
    print()
    tot = ok = 0
    for ri, r in enumerate(rows):
        want = r if r in names else "UNKNOWN"
        n = int(cm[ri].sum())
        c = int(cm[ri, cols.index(want)])
        tot += n
        ok += c
        if n:
            note = "" if r in names else "  [not enrolled: should be UNKNOWN]"
            print(f"{r:<14} {c}/{n} correct ({c / n:.0%}){note}")
    if tot:
        print(f"\nOverall: {ok}/{tot} correct ({ok / tot:.1%}) "
              f"at threshold {a.threshold}, margin {a.margin}")


def cmd_export(_):
    names, C = load_db()
    lines = ["// Generated by sid.py export. Fingerprints from the PC mic.",
             "// Re-enroll through the device mic before final use.",
             f"#define EMBED_LEN {C.shape[1]}",
             f"#define NUM_SPEAKERS {len(names)}", ""]
    for n, v in zip(names, C):
        ident = "".join(ch if ch.isalnum() else "_" for ch in n)
        vals = ", ".join(f"{x:.6f}f" for x in v)
        lines.append(f"const float ref_{ident}[EMBED_LEN] = {{{vals}}};")
    lines.append("")
    lines.append("const char* speaker_names[NUM_SPEAKERS] = {" +
                 ", ".join(f'"{n}"' for n in names) + "};")
    Path("fingerprints.h").write_text("\n".join(lines))
    print("Wrote fingerprints.h")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices").set_defaults(fn=cmd_devices)

    r = sub.add_parser("record")
    r.add_argument("--name", required=True)
    r.add_argument("--session", type=int, default=1)
    r.add_argument("--set", choices=["enroll", "test"], default="enroll")
    r.add_argument("--seconds", type=int, default=10)
    r.add_argument("--device", type=int, default=None)
    r.set_defaults(fn=cmd_record)

    sub.add_parser("enroll").set_defaults(fn=cmd_enroll)

    l = sub.add_parser("live")
    l.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    l.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    l.add_argument("--gate", type=float, default=-45.0,
                   help="below this dBFS the window counts as silence")
    l.add_argument("--smooth", type=int, default=3,
                   help="average scores over the last N windows")
    l.add_argument("--device", type=int, default=None)
    l.set_defaults(fn=cmd_live)

    t = sub.add_parser("test")
    t.add_argument("--name", required=True)
    t.add_argument("--genuine", nargs="+", required=True)
    t.add_argument("--impostor", nargs="*")
    t.set_defaults(fn=cmd_test)

    cf = sub.add_parser("confusion")
    cf.add_argument("--set", choices=["enroll", "test"], default="test")
    cf.add_argument("--threshold", type=float, default=0.80)
    cf.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    cf.set_defaults(fn=cmd_confusion)

    sub.add_parser("export").set_defaults(fn=cmd_export)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()  