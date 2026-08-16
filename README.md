# thai-talkinghead-autocut

Automatically tighten a Thai talking-head video: remove dead air, stutters and hesitation fillers, then generate captions that stay locked to the voice.

Built for the case where someone records themselves speaking Thai straight to camera, one long take, no script, and the raw file is 30-40% pauses and "เอ่อ".

```bash
python transcribe.py     --src raw.mov --work work/clip
python detect_fillers.py --work work/clip          # review work/clip/fillers.json
python cut.py            --src raw.mov --work work/clip --out edited.mp4
python captions.py       --work work/clip --out captions.srt
```

Requires Python 3.9+, `ffmpeg` and `ffprobe` on your PATH, and `pip install -r requirements.txt`.

## Why it is built this way

### Two whisper passes, not one

Voice activity detection exists to throw away non-speech. A drawn-out hesitation vowel is, acoustically, indistinguishable from non-speech. So a VAD-filtered transcript is clean and accurate and contains **none of the fillers you want to cut**.

So `transcribe.py` runs the model twice:

| pass | audio | VAD | used for |
|---|---|---|---|
| A | clean | on | the transcript, word timings, captions |
| B | denoised | off | filler detection only |

Pass B is deliberately noisy. It is never used as a transcript, only asked "was a held vowel sitting here".

### Two dead-air detectors, not one

`silencedetect` at a strict -36 dB floor finds true silence, including pauses *inside* what whisper merged into a single word. It misses any pause masked by room tone, traffic, or a shop fan.

Word-gap analysis finds those masked pauses, because whisper knows nobody was speaking even when the noise floor disagrees.

Neither is sufficient. `cut.py` takes the union.

### Asymmetric padding

`--pad-lead` (kept before speech resumes, default 0.10s) is deliberately larger than `--pad-tail` (kept after speech ends, default 0.06s). Clipping a consonant onset is instantly audible as a glitch; cutting the tail slightly early just sounds tight. Filler cuts also keep the first 0.12s of the vowel, so the cut does not land on the transition from the preceding word and click.

### Thai line breaking

Thai has no spaces between words, so captions cannot wrap on whitespace, and wrapping on character count alone splits words in half. `captions.py` packs lines at PyThaiNLP word boundaries, plus two readability rules:

- **lead-in words** (คือ ถ้า แล้ว เพราะ ก็ แต่ ...) never end a line, they pull to the next
- **enclitics** (ครับ ค่ะ นะ ไหม เลย ...) never start a line, they pull to the previous

### Caption remapping

Word timings come from the original timeline. After cutting, every timestamp past the first cut is wrong. `captions.py` projects each word through the keep-spans, dropping words that fell inside a cut and shifting the rest by the duration removed before them. Caption timing therefore comes from the words themselves and cannot drift.

## What each step writes

| step | output |
|---|---|
| `transcribe.py` | `audio16k.wav`, `audio16k_dn.wav`, `words.json`, `timedwords.json`, `tokens3.json`, `transcript.txt` |
| `detect_fillers.py` | `fillers.json` — `[[start, end], ...]` on the original timeline |
| `cut.py` | `keep_spans.json`, and the edited MP4 |
| `captions.py` | an SRT on the edited timeline |

`fillers.json` is a plain JSON list and is meant to be hand-edited. Detection is deliberately conservative; read the printed report, delete the false positives, add anything it missed, then run `cut.py`.

Run `cut.py --dry-run` to print the full cut plan without encoding anything.

## Tuning

| flag | default | when to change it |
|---|---|---|
| `--dead-min` | `0.34` | lower for a tighter edit, raise to keep the speaker's natural rhythm |
| `--silence-floor` | `-36` | raise toward `0` for a noisy room |
| `--drawl-min` | `0.55` | lower if obvious fillers are being missed |
| `--max-repeat-span` | `4.0` | how far apart two identical words can be and still count as a stutter |
| `--max-chars` | `16` | caption line length |
| `--prompt` | empty | domain vocabulary whisper keeps mishearing (names, jargon, product words) |

## Known limits

- Detection is tuned for one speaker in a single continuous take. Two people talking over each other will confuse the word-gap detector.
- `cut.py` re-encodes rather than stream-copying, so cuts land on exact frames instead of the nearest keyframe. On a long clip this is the slow step.
- Reduplicated Thai words written out in full (ๆ expanded) can read as a stutter. That is why a 2x repeat of a single-character word is ignored by default.
- Rendering an SRT into video through libass drops stacked Thai tone marks, which is a separate bug in the renderer rather than in the subtitle file. If you need burned-in Thai captions, render the text as images and overlay them; see [thai-text-render](https://github.com/anthonychkwn/thai-text-render).

## License

MIT
