# -*- coding: utf-8 -*-
"""
Unit tests for the pure-logic parts of the pipeline: filler detection, cut
planning, timeline inversion, caption remapping and Thai line packing.

No audio, no whisper, no ffmpeg. Run:  pytest -q
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import captions
import cut
import detect_fillers


def w(word, t, e):
    return {"w": word, "t": t, "e": e}


# ---------------------------------------------------------------- detect_fillers

def test_repeated_word_is_cut_keeping_the_last():
    words = [w("ดู", 1.0, 1.2), w("ดู", 1.3, 1.5), w("ดู", 1.6, 1.8), w("หนัง", 1.9, 2.2)]
    spans = detect_fillers.find_repeats(words, max_span=4.0)
    assert len(spans) == 1
    start, end, _label = spans[0]
    # cut covers the first two, the final ดู at 1.6 survives
    assert start == 1.0 and end < 1.6


def test_single_2x_short_word_is_not_a_stutter():
    # a 2x repeat of a 1-char word is usually real Thai reduplication
    words = [w("ๆ", 1.0, 1.1), w("ๆ", 1.2, 1.3), w("ไป", 1.4, 1.6)]
    assert detect_fillers.find_repeats(words, max_span=4.0) == []


def test_distant_repeat_is_not_a_stutter():
    words = [w("ผม", 1.0, 1.2), w("ผม", 9.0, 9.2)]
    assert detect_fillers.find_repeats(words, max_span=4.0) == []


def test_repeated_phrase_cuts_the_first_attempt():
    words = [w("เก้า", 1.0, 1.2), w("คุม", 1.3, 1.5),
             w("เก้า", 1.6, 1.8), w("คุม", 1.9, 2.1), w("เนอะ", 2.2, 2.4)]
    spans = detect_fillers.find_ngram_repeats(words, max_span=4.0)
    assert len(spans) == 1
    assert spans[0][0] == 1.0 and spans[0][1] < 1.6


def test_drawled_vowel_keeps_its_onset():
    tokens = [{"w": "อ่า", "s": 5.0, "e": 6.0, "p": 0.4}]
    spans = detect_fillers.find_drawls(tokens, lo=0.55, hi=2.2, onset_keep=0.12)
    assert len(spans) == 1
    assert spans[0][0] == 5.12          # onset preserved so the cut does not click
    assert spans[0][1] == 5.98


def test_short_vowel_is_normal_speech():
    tokens = [{"w": "อ่า", "s": 5.0, "e": 5.3, "p": 0.4}]
    assert detect_fillers.find_drawls(tokens, lo=0.55, hi=2.2, onset_keep=0.12) == []


def test_merge_joins_overlapping_spans():
    spans = [[1.0, 2.0, "a"], [1.9, 3.0, "b"], [5.0, 6.0, "c"]]
    merged = detect_fillers.merge(spans)
    assert [(s[0], s[1]) for s in merged] == [(1.0, 3.0), (5.0, 6.0)]


# ------------------------------------------------------------------------- cut

ARGS = types.SimpleNamespace(dead_min=0.34, pad_lead=0.10, pad_tail=0.06, tail_keep=0.22)


def test_word_gap_becomes_a_cut_with_asymmetric_padding():
    words = [w("หนึ่ง", 0.5, 1.0), w("สอง", 2.0, 2.5)]
    cuts = cut.build_cuts([], words, [], duration=3.0, args=ARGS)
    gap_cuts = [c for c in cuts if c[2] == "word gap"]
    assert len(gap_cuts) == 1
    a, b, _ = gap_cuts[0]
    assert a == 1.06        # speech end + pad_tail
    assert b == 1.90        # next word - pad_lead (lead pad is the bigger one)


def test_small_gap_is_left_alone():
    words = [w("หนึ่ง", 0.5, 1.0), w("สอง", 1.2, 1.7)]
    cuts = cut.build_cuts([], words, [], duration=2.0, args=ARGS)
    assert [c for c in cuts if c[2] == "word gap"] == []


def test_invert_produces_ordered_disjoint_keeps():
    words = [w("a", 0.5, 1.0), w("b", 2.0, 2.5), w("c", 4.0, 4.5)]
    cuts = cut.build_cuts([], words, [[3.0, 3.5]], duration=5.0, args=ARGS)
    keeps = cut.invert(cuts, 5.0)
    prev = 0.0
    for a, b in keeps:
        assert 0.0 <= a < b <= 5.0
        assert a >= prev
        prev = b
    # keeps + cuts must tile the full duration
    total = sum(b - a for a, b in keeps) + sum(b - a for a, b, _ in cuts)
    assert abs(total - 5.0) < 0.02


# -------------------------------------------------------------------- captions

def test_remap_shifts_words_and_drops_cut_ones():
    words = [w("หนึ่ง", 0.0, 1.0), w("ตัด", 1.5, 2.5), w("สอง", 3.0, 4.0)]
    keeps = [[0.0, 1.2], [2.8, 4.2]]        # the middle word falls inside the cut
    out = captions.remap(words, keeps)
    assert [x["w"] for x in out] == ["หนึ่ง", "สอง"]
    assert out[0]["t"] == 0.0
    assert abs(out[1]["t"] - (1.2 + (3.0 - 2.8))) < 1e-6  # offset of 2nd keep + local position


def test_pack_respects_max_chars_at_word_boundaries():
    text = "เพราะภาษาไทยไม่มีช่องว่างระหว่างคำ"
    lines = captions.pack(captions.tokenize(text), max_chars=12)
    assert len(lines) >= 2
    for ln in lines:
        assert captions.visible_len(ln) <= 12 + 4    # enclitic tolerance


def test_lead_in_never_ends_a_line():
    for ln in captions.pack(captions.tokenize("ผมว่าเพราะเราไม่เคยลองทำแบบนี้มาก่อนเลย"), 10)[:-1]:
        toks = captions.tokenize(ln)
        assert toks[-1] not in captions.LEAD_IN, f"line ends on lead-in: {ln}"


def test_enclitic_never_starts_a_line():
    lines = captions.pack(captions.tokenize("ขอบคุณมากเลยนะครับทุกคนที่เข้ามาดูกัน"), 8)
    for ln in lines[1:]:
        toks = captions.tokenize(ln)
        assert toks[0] not in captions.ENCLITIC, f"line starts on enclitic: {ln}"


def test_srt_time_format():
    assert captions.srt_time(0.0) == "00:00:00,000"
    assert captions.srt_time(3661.5) == "01:01:01,500"
