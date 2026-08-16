# -*- coding: utf-8 -*-
"""
Step 3: plan the cuts, invert them to keep-spans, render with ffmpeg.

Dead air is found two ways, because neither alone is enough:

  audio silence   ffmpeg silencedetect at a strict -36 dB floor.
                  Catches pauses inside what whisper merged into one word.
                  Misses pauses masked by room tone, traffic, a shop fan.

  word gaps       the space between consecutive word timings.
                  Catches those masked pauses, because whisper knows nobody was
                  speaking even when the noise floor says otherwise.

Their union is the dead air. Add the filler spans from step 2, merge overlaps,
then invert: what is left is what you keep.

Asymmetric padding matters. PAD_LEAD (before speech resumes) is larger than
PAD_TAIL (after speech ends) because a clipped consonant onset is instantly
audible as a glitch, while a slightly early tail cut just sounds tight.

Usage:
    python cut.py --src clip.mov --work work/clip --out edited.mp4
    python cut.py --src clip.mov --work work/clip --out edited.mp4 --dry-run
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys


def probe_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path])
    return float(out.decode().strip())


def detect_silence(wav, floor_db, min_dur):
    """Parse ffmpeg silencedetect output into (start, end) pairs."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", wav, "-af",
         f"silencedetect=noise={floor_db}dB:d={min_dur}", "-f", "null", "-"],
        capture_output=True, text=True)
    spans, start = [], None
    for line in proc.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[0-9.]+)", line)
        if m:
            start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m and start is not None:
            spans.append((start, float(m.group(1))))
            start = None
    return spans


def build_cuts(silences, words, fillers, duration, args):
    cuts = []
    for s, e in silences:
        if e - s >= args.dead_min:
            a, b = s + args.pad_tail, e - args.pad_lead
            if b - a > 0.04:
                cuts.append([round(a, 3), round(b, 3), "silence"])
    for i in range(len(words) - 1):
        gap = words[i + 1]["t"] - words[i]["e"]
        if gap >= args.dead_min:
            a, b = words[i]["e"] + args.pad_tail, words[i + 1]["t"] - args.pad_lead
            if b - a > 0.04:
                cuts.append([round(a, 3), round(b, 3), "word gap"])
    for s, e in fillers:
        cuts.append([round(float(s), 3), round(float(e), 3), "filler"])

    last_speech = words[-1]["e"]
    if duration - (last_speech + args.tail_keep) > 0.1:
        cuts.append([round(last_speech + args.tail_keep, 3), round(duration, 3), "tail"])

    cuts.sort()
    merged = []
    for c in cuts:
        if merged and c[0] <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], c[1])
            if c[2] not in merged[-1][2]:
                merged[-1][2] += "+" + c[2]
        else:
            merged.append(c[:])
    return merged


def invert(cuts, duration, min_keep=0.03):
    keeps, pos = [], 0.0
    for a, b, _label in cuts:
        if a - pos > min_keep:
            keeps.append([round(pos, 3), round(a, 3)])
        pos = b
    if duration - pos > min_keep:
        keeps.append([round(pos, 3), round(duration, 3)])
    return keeps


def render(src, keeps, out, crf):
    """One trim+concat filtergraph. Re-encodes, so cuts land on exact frames."""
    v = [f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS[v{i}]" for i, (a, b) in enumerate(keeps)]
    a_ = [f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS[a{i}]" for i, (a, b) in enumerate(keeps)]
    labels = "".join(f"[v{i}][a{i}]" for i in range(len(keeps)))
    fc = ";".join(v + a_) + ";" + labels + f"concat=n={len(keeps)}:v=1:a=1[v][a]"
    cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", fc,
           "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-r", "30000/1001", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "160k", out]
    print(f"\nrendering {len(keeps)} segments -> {out}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        sys.exit(1)
    print("done ->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dead-min", type=float, default=0.34,
                    help="trim any pause at or above this many seconds")
    ap.add_argument("--pad-lead", type=float, default=0.10,
                    help="keep this much before speech resumes (protects consonant onsets)")
    ap.add_argument("--pad-tail", type=float, default=0.06,
                    help="keep this much after speech ends")
    ap.add_argument("--tail-keep", type=float, default=0.22,
                    help="keep this much after the final word, then cut")
    ap.add_argument("--silence-floor", type=float, default=-36.0,
                    help="dB floor for silencedetect; raise toward 0 for noisy rooms")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, render nothing")
    args = ap.parse_args()

    W = args.work
    duration = probe_duration(args.src)
    words = json.load(open(os.path.join(W, "timedwords.json"), encoding="utf-8"))
    fillers_path = os.path.join(W, "fillers.json")
    fillers = json.load(open(fillers_path, encoding="utf-8")) \
        if os.path.exists(fillers_path) else []
    silences = detect_silence(os.path.join(W, "audio16k.wav"),
                              args.silence_floor, 0.25)

    cuts = build_cuts(silences, words, fillers, duration, args)
    keeps = invert(cuts, duration)
    removed = sum(b - a for a, b, _ in cuts)
    new_dur = sum(b - a for a, b in keeps)

    json.dump({"keeps": keeps, "cuts": cuts,
               "src_dur": round(duration, 3), "new_dur": round(new_dur, 3)},
              open(os.path.join(W, "keep_spans.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"source {duration:.2f}s  ->  edit {new_dur:.2f}s "
          f"({removed:.2f}s removed, {removed / duration * 100:.0f}%)")
    print(f"{len(cuts)} cuts, {len(keeps)} keep-segments, {len(fillers)} filler spans")
    print("--- cuts ---")
    for a, b, label in cuts:
        print(f"  {a:7.2f} - {b:7.2f}  ({b - a:.2f}s)  {label}")

    if args.dry_run:
        print("\ndry run, nothing rendered")
        return
    render(args.src, keeps, args.out, args.crf)


if __name__ == "__main__":
    # UTF-8 stdout for Thai on Windows consoles; kept out of import time so
    # the module stays importable as a library.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
