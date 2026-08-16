# -*- coding: utf-8 -*-
"""
Step 1: transcribe a Thai talking-head clip to word-level timings.

Runs faster-whisper twice over the same audio, because the two passes are good
at different things:

  pass A  VAD on, clean audio   -> accurate words and boundaries
  pass B  VAD off, denoised     -> keeps the "เอ่อ / อ่า / อือ" that VAD eats

VAD exists to throw away non-speech, and a drawn-out hesitation vowel looks
exactly like non-speech to it. If you only run pass A you cannot cut fillers,
because they are not in the transcript. Pass B is deliberately noisy and is only
ever used as a filler oracle, never as the transcript.

Outputs into --work:
    audio16k.wav      16 kHz mono, extracted from the source
    audio16k_dn.wav   the same, denoised
    words.json        pass A raw whisper tokens with probabilities
    timedwords.json   pass A tokens regrouped into whole words
    tokens3.json      pass B raw tokens
    transcript.txt    human-readable pass A segments

Usage:
    python transcribe.py --src clip.mov --work work/clip
    python transcribe.py --src clip.mov --work work/clip \
        --prompt "domain words that whisper keeps mishearing"
"""
import argparse
import io
import json
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def extract_audio(src, work):
    """16 kHz mono for whisper, plus a denoised copy for the filler pass."""
    clean = os.path.join(work, "audio16k.wav")
    denoised = os.path.join(work, "audio16k_dn.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", clean],
        check=True,
    )
    # afftdn is enough here; the goal is not clean audio, it is to stop room
    # tone from being transcribed as words during a hesitation.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", clean,
         "-af", "afftdn=nf=-25", denoised],
        check=True,
    )
    return clean, denoised


def group_into_words(tokens):
    """
    Whisper emits sub-word tokens and marks a new word with a LEADING SPACE.
    Regroup them so downstream code reasons about words, not fragments.
    """
    words, cur = [], None
    for t in tokens:
        starts_word = t["w"].startswith(" ")
        text = t["w"].strip()
        if cur is None or starts_word:
            if cur:
                words.append(cur)
            cur = {"w": text, "s": t["s"], "e": t["e"]}
        else:
            cur["w"] += text
            cur["e"] = t["e"]
    if cur:
        words.append(cur)
    return [w for w in words if w["w"].strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source video or audio file")
    ap.add_argument("--work", required=True, help="working directory for intermediates")
    ap.add_argument("--model", default="large-v3-turbo", help="faster-whisper model")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--compute-type", default="int8",
                    help="int8 for cpu, float16 for cuda")
    ap.add_argument("--prompt", default="",
                    help="domain vocabulary to bias pass A (names, jargon, product words)")
    ap.add_argument("--gap-report", type=float, default=0.34,
                    help="report word gaps at or above this many seconds")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    clean, denoised = extract_audio(args.src, args.work)

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    # ---- pass A: VAD on, clean audio, domain prompt -> the real transcript ----
    seg_a, _ = model.transcribe(
        clean, language="th", word_timestamps=True,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=350),
        beam_size=8, best_of=5,
        initial_prompt=args.prompt or None,
        condition_on_previous_text=False, temperature=0.0,
    )
    raw, lines = [], []
    for s in seg_a:
        lines.append(f"[{s.start:6.2f}-{s.end:6.2f}] {s.text.strip()}")
        for w in (s.words or []):
            raw.append({"w": w.word, "s": round(w.start, 3),
                        "e": round(w.end, 3), "p": round(w.probability, 3)})

    words = group_into_words(raw)
    for i, w in enumerate(words):
        w["i"] = i
        w["t"] = round(w["s"], 3)
        w["e"] = round(w["e"], 3)

    W = args.work
    json.dump(raw, open(os.path.join(W, "words.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump([{"i": w["i"], "w": w["w"], "t": w["t"], "e": w["e"]} for w in words],
              open(os.path.join(W, "timedwords.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    open(os.path.join(W, "transcript.txt"), "w", encoding="utf-8").write("\n".join(lines))

    # ---- pass B: VAD off, denoised -> filler oracle only ----
    seg_b, _ = model.transcribe(
        denoised, language="th", word_timestamps=True,
        vad_filter=False, beam_size=5,
        condition_on_previous_text=False, temperature=0.0,
        initial_prompt="พูดมีเสียงเอ่อ อ่า อือ เอิ่ม แทรก",
    )
    tokens3 = [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
                "p": round(w.probability, 3)}
               for s in seg_b for w in (s.words or [])]
    json.dump(tokens3, open(os.path.join(W, "tokens3.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    print("=== TRANSCRIPT ===")
    print("\n".join(lines))
    print(f"\n=== WORD-GAP DEAD AIR (>= {args.gap_report}s) ===")
    for i in range(len(words) - 1):
        gap = words[i + 1]["t"] - words[i]["e"]
        if gap >= args.gap_report:
            print(f"  {gap:.2f}s at {words[i]['e']:.2f}-{words[i + 1]['t']:.2f}")
    print(f"\nwords {len(words)}   raw tokens {len(raw)}   filler tokens {len(tokens3)}")
    print(f"wrote -> {W}")


if __name__ == "__main__":
    main()
