"""Project predicted sentences back onto the original words.

Generative segmenters (LLMs) may drop, add or alter words. ``project`` returns one boundary
label per original word: if the predicted words match exactly the labels are read off directly,
otherwise a character-edit-distance (CED) word alignment maps predicted sentence ends onto the
original words. Documents with >= 5000 words are aligned in overlapping 1000-word windows.

The alignment code is a port of ArabicSeg/Notebooks/{aligner,ced_alignment,utils}.py.
"""
from __future__ import annotations

import logging
import re
import unicodedata

import editdistance

from utils.data import labels_to_segments, sentence_labels

log = logging.getLogger(__name__)

CHUNKED_MIN_WORDS = 5000
CHUNK_WORDS = 1000
CHUNK_OVERLAP = 100
CHUNK_PRED_MARGIN = 120


def project(pred_sentences: list[str], words: list[str]) -> list[int]:
    """Boundary labels (len == len(words)) for ``words`` given a model's predicted sentences."""
    pred_words, labels = sentence_labels(pred_sentences)
    if pred_words == words:
        return labels
    return align_predictions(pred_sentences, [" ".join(words)])


def snap(pred_sentences: list[str], words: list[str]) -> list[str]:
    """Rewrite predicted sentences so they consist exactly of ``words``."""
    return labels_to_segments(words, project(pred_sentences, words))


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ALEF = re.compile("[إأٱآ]")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _norm_pnx_nums(s: str) -> str:
    s = "".join("PNX" + c if _is_punct(c) else c for c in s.strip())
    s = s.translate(_DIGITS)
    lines = [" ".join(w.replace("ـ", "") if len(w) > 1 else w for w in line.split())
             for line in s.split("\n")]
    return " \n ".join(lines)


def _norm_alef_ya_teh(s: str) -> str:
    return _ALEF.sub("ا", s).replace("ى", "ي").replace("ة", "ه")


def _tokenize(text: str) -> list[str]:
    tokens = []
    for part in re.split(r"(\n)", text):
        tokens.extend([part] if part == "\n" else part.split())
    return tokens


def _align_words(s1: str, s2: str) -> list[tuple]:
    """Weighted edit-distance alignment of the words of s1 (pred) and s2 (gold)."""
    t1, t2 = _tokenize(_norm_pnx_nums(s1)), _tokenize(_norm_pnx_nums(s2))
    m, n = len(t1), len(t2)
    cost = [[0.0] * (n + 1) for _ in range(m + 1)]
    op = [["n"] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        cost[i][0], op[i][0] = i, "d"
    for j in range(1, n + 1):
        cost[0][j], op[0][j] = j, "i"
    for i in range(m):
        for j in range(n):
            a, b = t1[i], t2[j]
            if a == b:
                best = (cost[i][j], "n")
            else:
                best = (cost[i][j] + editdistance.eval(a, b) * 2 / max(len(a), len(b)), "s")
            best = min([(cost[i][j + 1] + 1, "d"), (cost[i + 1][j] + 1, "i"), best], key=lambda t: t[0])
            cost[i + 1][j + 1], op[i + 1][j + 1] = best

    out = []
    i, j = m, n
    while i or j:
        o = op[i][j]
        if o in ("n", "s"):
            out.append((i, j, t1[i - 1], t2[j - 1], "KEEP" if o == "n" else "REPLACE", "", ""))
            i, j = i - 1, j - 1
        elif o == "i":
            out.append(("None", j, t2[j - 1], "INSERT", "", t2[j - 1]))
            j -= 1
        else:
            out.append((i, "None", t1[i - 1], "DELETE", t1[i - 1], ""))
            i -= 1
    return out[::-1]


def _capture_bugs(alignment) -> list[tuple[int, int, list]]:
    """Spans of >1 consecutive DELETE/REPLACE/INSERT ops: (start, end, ops)."""
    spans, i = [], 0
    while i < len(alignment):
        start, ops = i - 1, []
        while i < len(alignment) and (
            "DELETE" in alignment[i][3] or "REPLACE" in alignment[i][4] or "INSERT" in alignment[i][3]
        ):
            ops.append(alignment[i])
            i += 1
        if len(ops) > 1:
            spans.append((start, i, ops))
        elif not ops:
            i += 1
    return spans


def _bug_src_tgt(ops) -> tuple[list[str], list[str]]:
    src, tgt = [], []
    for v in ops:
        if v[3] == "DELETE":
            src.append(v[-2]); tgt.append("NIL")
        elif v[3] == "INSERT":
            src.append("NIL"); tgt.append(v[-1])
        elif v[4] == "REPLACE":
            src.append(v[2] if v[-2] == "" else v[-2]); tgt.append(v[3] if v[-1] == "" else v[-1])
    return src, tgt


def _clean_src_tgt(align) -> tuple[list[str], list[str]]:
    src, tgt = [], []
    for x in align:
        if x[4] in ("KEEP", "REPLACE"):
            src.append(x[2] if x[-2] == "" else x[-2]); tgt.append(x[3] if x[-1] == "" else x[-1])
        elif x[3] == "INSERT":
            src.append(""); tgt.append(x[-1])
        elif x[3] == "DELETE":
            src.append(x[-2]); tgt.append("")
    return src, tgt


def _edit(src: list[str], tgt: list[str]) -> float:
    total = 0.0
    for s, t in zip(src, tgt):
        s, t = _norm_alef_ya_teh(s), _norm_alef_ya_teh(t)
        total += editdistance.eval(s.replace("PNX", "") if s != "NIL" else "",
                                   t.replace("PNX", "") if t != "NIL" else "")
    return total


def _split_merge(src, tgt, i, mode, src_first):
    """Try merging the unmatched token at i into its neighbour (prepend/append)."""
    def result(s, t, e):
        return {"src": s, "tgt": t, "edit": e} if src_first else {"src": t, "tgt": s, "edit": e}

    if mode == "prepend":
        j = i - 1
        while j >= 0 and tgt[j] == "NIL":
            j -= 1
        if j >= 0:
            src_ = src[:i] + src[i + 1:]
            tgt_ = tgt[:j] + [tgt[j] + " " + tgt[i]] + [tgt[x] for x in range(j + 1, len(tgt)) if x != i]
            if len(src_) == len(tgt_):
                e_all, e_ns = _edit(src_, tgt_), _edit(src_, [x.replace(" ", "") for x in tgt_])
                return result(src_, tgt_, e_ns + 0.1 * (e_all - e_ns))
    else:
        j = i + 1
        while j < len(tgt) and tgt[j] == "NIL":
            j += 1
        if j < len(tgt):
            src_ = src[:i] + src[i + 1:]
            tgt_ = tgt[:i] + [tgt[i] + " " + tgt[j]] + [tgt[x] for x in range(i + 1, len(tgt)) if x != j]
            if len(src_) == len(tgt_):
                e_all, e_ns = _edit(src_, tgt_), _edit(src_, [x.replace(" ", "") for x in tgt_])
                return result(src_, tgt_, e_ns + 0.1 * (e_all - e_ns))
    return result(src, tgt, _edit(src, tgt))


def _best(*cands):
    if cands[0]["edit"] == cands[1]["edit"] == cands[2]["edit"]:
        return cands[0]
    return min(cands, key=lambda c: c["edit"])


def _perfect_align(src, tgt):
    best = {"src": src, "tgt": tgt, "edit": _edit(src, tgt)}
    i = 0
    while i < len(src):
        if src[i] == "NIL" and tgt[i] != "NIL":
            best = _best(best, _split_merge(src, tgt, i, "prepend", True), _split_merge(src, tgt, i, "append", True))
        elif src[i] != "NIL" and tgt[i] == "NIL":
            best = _best(best, _split_merge(tgt, src, i, "prepend", False), _split_merge(tgt, src, i, "append", False))
        if best["src"] != src or best["tgt"] != tgt:
            src, tgt, i = best["src"], best["tgt"], 0
            continue
        i += 1
    return best["src"], best["tgt"]


def _is_single_punct(tok: str) -> bool:
    return len(tok) == 1 and _is_punct(tok)


def _reduce_span(span):
    inserts = [x for x in span if x[0] == "" and x[1] != ""]
    deletes = [x for x in span if x[0] != "" and x[1] == ""]
    ii = di = 0
    out = []
    while ii < len(inserts) and di < len(deletes):
        ins, dele = inserts[ii], deletes[di]
        tgt, src = ins[1], dele[0]
        tp, sp = _is_single_punct(tgt), _is_single_punct(src)
        if (tp and len(src) == 1) or (sp and len(tgt) == 1) or (not tp and not sp) or (sp and tp):
            out.append((src, tgt)); ii += 1; di += 1
        elif len(deletes) > len(inserts):
            out.append(dele); di += 1
        else:
            out.append(ins); ii += 1
    out += inserts[ii:] + deletes[di:]
    return out


def _reduce_inserts_deletions(src, tgt):
    src = [x.replace("PNX", "").replace("NIL", "") for x in src]
    tgt = [x.replace("PNX", "").replace("NIL", "") for x in tgt]
    out, i = [], 0
    while i < len(src):
        if src[i] == "" and tgt[i] != "":
            s = i
            while s < len(src) and src[s] == "" and tgt[s] != "":
                s += 1
            d = s
            while d < len(src) and src[d] != "" and tgt[d] == "":
                d += 1
            if d != s:
                out += _reduce_span(list(zip(src[i:d], tgt[i:d])))
                i = d
                continue
        out.append((src[i], tgt[i]))
        i += 1
    return [x[0] for x in out], [x[1] for x in out]


def _post_process(align):
    bugs = _capture_bugs(align)
    if not bugs:
        src, tgt = _clean_src_tgt(align)
    else:
        src, tgt, start = [], [], 0
        for b_start, b_end, ops in bugs:
            s, t = _clean_src_tgt(align[start: b_start + 1])
            src += s; tgt += t
            s, t = _perfect_align(*_bug_src_tgt(ops))
            src += s; tgt += t
            start = b_end
        s, t = _clean_src_tgt(align[start:])
        src += s; tgt += t
    return _reduce_inserts_deletions(src, tgt)


def _labels_from_alignment(src, tgt, n_gold: int, force_last: bool) -> list[int]:
    """A gold word is a boundary if the next non-empty pred token after it is a newline."""
    labels = []
    for i, tok in enumerate(tgt):
        if tok in ("", "\n"):
            continue
        boundary = 0
        if src[i] not in ("", "\n"):
            for j in range(i + 1, len(src)):
                if src[j] == "":
                    continue
                boundary = int(src[j] == "\n")
                break
        toks = tok.split()
        labels += [0] * (len(toks) - 1) + [boundary]
    if len(labels) != n_gold:
        raise ValueError(f"alignment produced {len(labels)} labels for {n_gold} gold words")
    if force_last:
        labels[-1] = 1
    return labels


def _align_single(pred_sentences, gold_sentences, n_gold, force_last):
    align = _align_words(" \n ".join(pred_sentences), " \n ".join(gold_sentences))
    src, tgt = _post_process(align)
    return _labels_from_alignment(src, tgt, n_gold, force_last)


def _pred_slice(pred_sentences, w0, w1):
    """Consecutive predicted sentences overlapping pred word range [w0, w1)."""
    counts = [len(s.split()) for s in pred_sentences]
    total = sum(counts)
    w0 = max(0, min(w0, total - 1))
    w1 = max(w0 + 1, min(w1, total))
    acc, first = 0, 0
    for k, c in enumerate(counts):
        if acc + c > w0:
            first = k
            break
        acc += c
    acc, last = 0, len(pred_sentences) - 1
    for k, c in enumerate(counts):
        acc += c
        if acc >= w1:
            last = k
            break
    return pred_sentences[first: last + 1]


def _align_chunked(pred_sentences, gold_words):
    n = len(gold_words)
    scale = sum(len(s.split()) for s in pred_sentences) / max(1, n)
    final = [0] * n
    start = 0
    while start < n:
        end = min(start + CHUNK_WORDS, n)
        chunk = gold_words[start:end]
        preds = _pred_slice(pred_sentences, int(start * scale) - CHUNK_PRED_MARGIN,
                            int(end * scale) + CHUNK_PRED_MARGIN) or pred_sentences
        try:
            labels = _align_single(preds, [" ".join(chunk)], len(chunk), force_last=end >= n)
        except Exception as e:
            log.warning("chunk alignment failed [%d:%d]: %s", start, end, e)
            labels = [0] * len(chunk)
        hi = end if (start == 0 or end == n or end - start <= 2 * CHUNK_OVERLAP) else end - CHUNK_OVERLAP
        final[start:hi] = labels[: hi - start]
        if end == n:
            break
        start += CHUNK_WORDS - CHUNK_OVERLAP
    final[-1] = 1
    return final


def align_predictions(pred_sentences: list[str], gold_sentences: list[str]) -> list[int]:
    """Per-gold-word boundary labels from predicted sentences via CED alignment."""
    gold_sentences = [s for s in gold_sentences if s.strip()]
    pred_sentences = [s for s in pred_sentences if s.strip()]
    gold_words = [w for s in gold_sentences for w in s.split()]
    if not gold_words:
        return []
    if not pred_sentences:
        return [0] * len(gold_words)
    try:
        if len(gold_words) >= CHUNKED_MIN_WORDS:
            return _align_chunked(pred_sentences, gold_words)
        return _align_single(pred_sentences, gold_sentences, len(gold_words), force_last=True)
    except Exception as e:
        log.warning("alignment failed (%s); predicting no boundaries", e)
        return [0] * len(gold_words)
