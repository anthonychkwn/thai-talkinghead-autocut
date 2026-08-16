# -*- coding: utf-8 -*-
"""
Step 4: build captions for the EDITED timeline and write an SRT.

Two problems this solves.

Remapping. Word timings from transcribe.py are on the original timeline. After
cut.py removes spans, every timestamp after the first cut is wrong. So each word
is projected through the keep-spans: words that fell inside a cut are dropped,
and the rest are shifted by the total duration removed before them.

Line breaking. Thai does not put spaces between words, so you cannot wrap on
whitespace, and breaking on character count alone splits words in half. Lines
are packed at PyThaiNLP word boundaries with two extra rules:

  lead-in words   คือ ถ้า แล้ว เพราะ ...   never end a line, they pull to the next
  enclitics       ครับ ค่ะ นะ ไหม เลย ...   never start a line, they pull to the previous

Both are about how the line reads on screen: a viewer reading a line ending in
"เพราะ" is left hanging, and a line beginning with "ครับ" reads as a fragment.

Usage:
    python captions.py --work work/clip --out captions.srt
    python captions.py --work work/clip --out captions.srt --max-chars 20
"""
import argparse
import io
import json
import os
import sys

# True connectives only. Subject pronouns (ผม เรา) are NOT lead-ins: ending a
# line on the subject and starting the next on the predicate is a natural Thai
# break, and treating pronouns as unbreakable makes some sentences unsolvable.
LEAD_IN = set("คือ ถ้า แล้ว เพราะ ก็ แต่ และ ที่ ซึ่ง กับ หรือ พอ จน ว่า ใน ของ ให้".split())
ENCLITIC = set("ไหม มั้ย นะ น่ะ ครับ คับ ค่ะ คะ ล่ะ หรอ เหรอ สิ ซิ แหละ ไง ด้วย เลย กัน อีก".split())


def visible_len(s):
    return len(s.replace(" ", ""))


def tokenize(s):
    from pythainlp.tokenize import word_tokenize
    return [t for t in word_tokenize(s.strip(), engine="newmm") if t.strip()]


def remap(words, keeps):
    """
    Project original-timeline words onto the edited timeline.

    Each keep-span [a, b) maps to [offset, offset + (b - a)) in the output, so a
    word at t inside that span lands at offset + (t - a).
    """
    out, offset = [], 0.0
    for a, b in keeps:
        for w in words:
            if w["t"] >= a and w["e"] <= b:
                out.append({"w": w["w"].strip(),
                            "t": round(offset + (w["t"] - a), 3),
                            "e": round(offset + (w["e"] - a), 3)})
        offset += b - a
    out.sort(key=lambda w: w["t"])
    return out


def pack(tokens, max_chars):
    """Greedy-pack tokens into lines, then apply the lead-in / enclitic rules.

    The rules run to a fixed point: moving a word can expose a new violation
    (splitting "ผมว่าเพราะ" after moving เพราะ leaves ว่า stranded at the line
    end), so a single pass is not enough. The two rules cannot ping-pong the
    same word because the lead-in and enclitic sets are disjoint; the loop cap
    is just a backstop.
    """
    lines, cur = [], ""
    for t in tokens:
        if cur and visible_len(cur) + visible_len(t) > max_chars:
            # Peel trailing lead-ins off the full line so the break lands
            # before them, as long as they fit on the next line with t.
            ctoks = tokenize(cur)
            carry = []
            while ctoks and ctoks[-1] in LEAD_IN and \
                    sum(visible_len(x) for x in carry) + visible_len(ctoks[-1]) \
                    + visible_len(t) <= max_chars + 4:
                carry.insert(0, ctoks.pop())
            if ctoks:
                lines.append("".join(ctoks))
            cur = "".join(carry) + t
        else:
            cur += t
    if cur:
        lines.append(cur)

    for _ in range(10):
        lines = [ln for ln in lines if ln.strip()]
        changed = False

        # a line must not END on a lead-in word: push it to the next line
        for i in range(len(lines) - 1):
            toks = tokenize(lines[i])
            if len(toks) > 1 and toks[-1] in LEAD_IN:
                lines[i] = "".join(toks[:-1])
                lines[i + 1] = toks[-1] + lines[i + 1]
                changed = True

        # a line must not START on an enclitic: pull it back to the previous
        # line. A line made ONLY of enclitics is pulled unconditionally; a
        # slightly over-long line beats an orphaned "กัน" on screen.
        for i in range(1, len(lines)):
            toks = tokenize(lines[i])
            if not toks or not lines[i - 1]:
                continue
            fits = visible_len(lines[i - 1]) + visible_len(toks[0]) <= max_chars + 4
            all_enclitic = all(x in ENCLITIC for x in toks)
            if toks[0] in ENCLITIC and (fits or all_enclitic):
                lines[i - 1] += toks[0]
                lines[i] = "".join(toks[1:])
                changed = True

        if not changed:
            break

    return [ln for ln in lines if ln.strip()]


def group_words(words, max_chars, max_gap):
    """
    Walk the remapped words, starting a new caption when the line is full or a
    pause opens up. Timing comes from the words themselves, so captions stay
    locked to the voice instead of drifting.
    """
    caps, cur = [], None
    for w in words:
        if cur is None:
            cur = {"t": w["t"], "e": w["e"], "words": [w["w"]]}
            continue
        too_long = visible_len("".join(cur["words"]) + w["w"]) > max_chars
        big_gap = w["t"] - cur["e"] > max_gap
        if too_long or big_gap:
            caps.append(cur)
            cur = {"t": w["t"], "e": w["e"], "words": [w["w"]]}
        else:
            cur["words"].append(w["w"])
            cur["e"] = w["e"]
    if cur:
        caps.append(cur)

    # Re-break each caption at real word boundaries; distribute its window by
    # character share so a split caption keeps sensible timing.
    out = []
    for c in caps:
        text = "".join(c["words"])
        parts = pack(tokenize(text), max_chars)
        total = sum(visible_len(p) for p in parts) or 1
        span = c["e"] - c["t"]
        acc = 0
        for p in parts:
            t0 = c["t"] + span * (acc / total)
            acc += visible_len(p)
            t1 = c["t"] + span * (acc / total)
            out.append({"t": round(t0, 3), "e": round(max(t1, t0 + 0.4), 3), "text": p})
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True, help="output .srt path")
    ap.add_argument("--max-chars", type=int, default=16,
                    help="maximum Thai characters per caption line")
    ap.add_argument("--max-gap", type=float, default=0.5,
                    help="start a new caption when the pause exceeds this")
    args = ap.parse_args()

    W = args.work
    words = json.load(open(os.path.join(W, "timedwords.json"), encoding="utf-8"))
    keeps = json.load(open(os.path.join(W, "keep_spans.json"), encoding="utf-8"))["keeps"]

    remapped = remap(words, keeps)
    caps = group_words(remapped, args.max_chars, args.max_gap)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, c in enumerate(caps, 1):
            f.write(f"{i}\n{srt_time(c['t'])} --> {srt_time(c['e'])}\n{c['text']}\n\n")

    dropped = len(words) - len(remapped)
    print(f"{len(words)} words, {dropped} fell inside cuts, {len(caps)} captions")
    print(f"longest line: {max(visible_len(c['text']) for c in caps)} chars")
    print("wrote ->", args.out)
    for c in caps[:8]:
        print(f"  {c['t']:6.2f}  {c['text']}")


if __name__ == "__main__":
    # UTF-8 stdout for Thai on Windows consoles; kept out of import time so
    # the module stays importable as a library.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
