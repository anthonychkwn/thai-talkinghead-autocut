# -*- coding: utf-8 -*-
"""
Step 2: find stutters, repeated phrases and drawn-out hesitation vowels.

Three detectors, all operating on the timings produced by transcribe.py:

  1. repeated single word   "ดู ดู ดู"        -> keep the last occurrence
  2. repeated n-gram        "9 คุม 9 คุม"     -> cut the earlier block
  3. drawn bare vowel       "อ่าาาา"           -> cut the sustain, keep the onset

(3) is the one that needs the second whisper pass. A hesitation vowel is a
single Thai vowel or vowel-plus-tone-mark held far longer than that syllable
ever is in real speech, so the rule is: token is one or two vowel characters
AND lasts longer than --drawl-min. The upper bound exists because anything
longer than a couple of seconds is more likely a mis-segmentation than a filler.

Cuts start 0.12s into the vowel rather than at its onset. Cutting from the very
start clips the consonant transition of the preceding word and you hear a click.

Writes fillers.json: a list of [start, end] spans on the ORIGINAL timeline.
Review it before running cut.py; it is a plain JSON list, meant to be edited.

Usage:
    python detect_fillers.py --work work/clip
"""
import argparse
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Thai vowels that get held during hesitation.
DRAWN_VOWELS = set("าอะเแโใไีืูึ")
# Tone marks and the mai-han-akat, stripped before the vowel test.
MARKS = "่้๊๋็ั"


def find_repeats(words, max_span):
    """Immediate repetition of the same word: keep the last one."""
    spans, i, n = [], 0, len(words)
    while i < n:
        j = i
        while j + 1 < n and words[j + 1]["w"] == words[i]["w"] and words[i]["w"]:
            j += 1
        run = j - i + 1
        wlen = len(words[i]["w"])
        # A 2x repeat of a 1-character word is usually real Thai, not a stutter
        # (ๆ-style reduplication), so require either 3+ repeats or a longer word.
        if run >= 2 and (run >= 3 or wlen >= 2) and wlen >= 1 \
                and (words[j]["t"] - words[i]["t"]) <= max_span:
            spans.append([round(words[i]["t"], 3), round(words[j]["t"] - 0.02, 3),
                          f"repeat x{run} '{words[i]['w']}'"])
        i = j + 1
    return spans


def find_ngram_repeats(words, max_span, sizes=(4, 3, 2)):
    """A whole phrase restarted: cut the first attempt, keep the second."""
    spans, n = [], len(words)
    for k in sizes:
        i = 0
        while i + 2 * k <= n:
            a = [words[x]["w"] for x in range(i, i + k)]
            b = [words[x]["w"] for x in range(i + k, i + 2 * k)]
            if a == b and all(a) and sum(len(x) for x in a) >= 3 \
                    and (words[i + k]["t"] - words[i]["t"]) <= max_span:
                spans.append([round(words[i]["t"], 3), round(words[i + k]["t"] - 0.02, 3),
                              f"phrase repeat x2 ({k} words)"])
                i += 2 * k
            else:
                i += 1
    return spans


def find_drawls(tokens, lo, hi, onset_keep):
    """Held hesitation vowels, from the no-VAD pass."""
    spans = []
    for t in tokens:
        base = t["w"].strip()
        dur = t["e"] - t["s"]
        core = "".join(c for c in base if c not in MARKS)
        if lo <= dur <= hi and core and len(core) <= 2 \
                and all(c in DRAWN_VOWELS for c in core):
            spans.append([round(t["s"] + onset_keep, 3), round(t["e"] - 0.02, 3),
                          f"drawl '{base}' {dur:.2f}s"])
    return spans


def merge(spans, tol=0.03, min_len=0.05):
    spans.sort()
    out = []
    for s in spans:
        if out and s[0] <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], s[1])
            out[-1][2] += " + " + s[2]
        else:
            out.append(s[:])
    return [s for s in out if s[1] - s[0] > min_len]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--max-repeat-span", type=float, default=4.0,
                    help="ignore 'repeats' further apart than this (they are not stutters)")
    ap.add_argument("--drawl-min", type=float, default=0.55,
                    help="a held vowel shorter than this is normal speech")
    ap.add_argument("--drawl-max", type=float, default=2.2,
                    help="longer than this is probably a mis-segmentation, not a filler")
    ap.add_argument("--onset-keep", type=float, default=0.12,
                    help="seconds of the vowel to keep, so the cut does not click")
    args = ap.parse_args()

    W = args.work
    timed = json.load(open(os.path.join(W, "timedwords.json"), encoding="utf-8"))
    tokens = json.load(open(os.path.join(W, "tokens3.json"), encoding="utf-8"))
    words = [{"w": re.sub(r"\s+", "", x["w"].strip()), "t": x["t"], "e": x["e"]}
             for x in timed]

    spans = merge(
        find_repeats(words, args.max_repeat_span)
        + find_ngram_repeats(words, args.max_repeat_span)
        + find_drawls(tokens, args.drawl_min, args.drawl_max, args.onset_keep)
    )

    out = os.path.join(W, "fillers.json")
    json.dump([[s[0], s[1]] for s in spans], open(out, "w"), indent=0)
    total = sum(s[1] - s[0] for s in spans)
    print(f"{len(spans)} filler/stutter spans, {total:.2f}s total -> {out}")
    for s in spans:
        print(f"  {s[0]:6.2f}-{s[1]:6.2f}  ({s[1] - s[0]:.2f}s)  {s[2]}")
    print("\nreview fillers.json before running cut.py; it is meant to be hand-edited")


if __name__ == "__main__":
    main()
