# cvxnn_wer_plot.py
# Plot WER (%) vs ADMM iteration for a binary convex NN (cvxNN) trained with CRONOS.
# Drop this file into your repo and import run_cvxnn_wer_plot in your training script.
# Seaborn is used for a clean line plot; cached transcripts avoid re-decoding Whisper each iteration.
#
# Usage (minimal):
#   from cvxnn_wer_plot import run_cvxnn_wer_plot, example_predict_lang_fn
#   df, fig = run_cvxnn_wer_plot(
#       admm_fn=admm,                 # your ADMM entry point
#       model=model,                  # your cvxNN model instance
#       cronos_params=cronos_params,  # dict used by your solver
#       X_val=X_val_feats,            # Whisper encoder features (val), shaped as your cvxNN expects
#       ref_texts=val_refs,           # list[str] reference transcripts
#       ref_langs=val_ref_langs,      # list[str] language code per utterance (e.g., "en", "zh")
#       val_audio=val_audio,          # list[np.ndarray] or any audio handles your decoder expects
#       languages=["en", "zh"],       # candidate language codes
#       decode_fn=decode_whisper,     # callable(audio, language) -> str transcript
#       predict_lang_fn=example_predict_lang_fn,  # adapter: model + X_val -> ["en"/"zh", ...]
#       out_dir="results",            # where to save PNG/CSV
#       eval_every=1,                 # compute WER every ADMM iteration
#       title="WER vs ADMM iteration — Whisper + cvxNN (binary)"
#   )
#
# If your ADMM doesn't accept a `callback=` kwarg yet, add inside the iteration loop:
#   if callback is not None: callback({"iter": k})
#
# Background: Evaluating WER at each ADMM iteration mirrors the CLD setup (encoder features
# + convex language detector + language-conditioned decoding) and matches the WER curves shown
# in the project report (see Fig. 4–5).

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------
# Text normalization + WER
# -------------------------

def _normalize_text(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)  # keep CJK for CER
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize_for_lang(s: str, lang: str) -> List[str]:
    s = _normalize_text(s)
    if str(lang).lower() in {"zh", "zh-cn", "zh-tw", "zh_hans", "zh_hant"}:
        return list(s.replace(" ", ""))  # CER for Chinese
    return s.split()  # word-level for others

def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        ai = a[i-1]
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (ai != b[j-1])
            )
    return dp[m][n]

def corpus_wer_mixed_lang(refs: Sequence[str], hyps: Sequence[str], langs: Sequence[str]) -> float:
    """
    Mixed-language WER: word-WER for non-CJK languages, character error rate (CER) for CJK.
    Returns a fraction in [0,1].
    """
    assert len(refs) == len(hyps) == len(langs), "refs/hyps/langs must align"
    total_edits, total_tokens = 0, 0
    for r, h, lang in zip(refs, hyps, langs):
        rt = _tokenize_for_lang(r, lang)
        ht = _tokenize_for_lang(h, lang)
        total_edits += _edit_distance(rt, ht)
        total_tokens += max(len(rt), 1)
    return total_edits / max(total_tokens, 1)


# ------------------------------------------
# Core: build transcripts cache per language
# ------------------------------------------

def build_transcripts_cache(
    languages: Sequence[str],
    val_audio: Sequence[Any],
    decode_fn: Callable[[Any, str], str],
) -> Dict[str, List[str]]:
    """
    Precompute one transcript per (utterance, language). This makes per-iteration WER O(N).
    """
    cache: Dict[str, List[str]] = {}
    for lang in languages:
        cache[str(lang).lower()] = [decode_fn(a_i, str(lang)) for a_i in val_audio]
    return cache


# -----------------------------------------
# Prediction adapter for a binary cvxNN
# -----------------------------------------

def example_predict_lang_fn(model: Any, X_val: np.ndarray, lang_map: Optional[Dict[int, str]] = None) -> List[str]:
    """
    Try common model APIs to get binary labels and map to language codes.
    lang_map maps label->code; default {0:"en", 1:"zh"}.
    """
    if lang_map is None:
        lang_map = {0: "en", 1: "zh"}

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X_val))
        y = probs.argmax(axis=1)
    elif hasattr(model, "predict"):
        out = np.asarray(model.predict(X_val))
        if out.ndim == 2 and out.shape[1] == 2:
            y = out.argmax(axis=1)
        else:
            # scores or labels
            y = (out > 0).astype(int)
    else:
        raise AttributeError("Model needs a predict or predict_proba method. Provide a custom predict_lang_fn if different.")

    return [str(lang_map[int(k)]).lower() for k in y]


# ----------------------------------------------------
# Runner: ADMM training with per-iteration WER record
# ----------------------------------------------------

def run_cvxnn_wer_plot(
    admm_fn: Callable[..., Any],
    model: Any,
    cronos_params: Dict[str, Any],
    X_val: np.ndarray,
    ref_texts: Sequence[str],
    ref_langs: Sequence[str],
    val_audio: Sequence[Any],
    languages: Sequence[str],
    decode_fn: Callable[[Any, str], str],
    predict_lang_fn: Callable[[Any, np.ndarray], Sequence[str]] = example_predict_lang_fn,
    out_dir: str = "results",
    eval_every: int = 1,
    title: str = "WER vs ADMM iteration — Whisper + cvxNN (binary)",
    csv_name: str = "wer_cvxnn.csv",
    png_name: str = "wer_cvxnn.png",
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Runs ADMM with a callback that computes WER at each iteration using cached transcripts.
    Saves a PNG and CSV, and returns (df, fig).
    """
    os.makedirs(out_dir, exist_ok=True)
    transcripts_cache = build_transcripts_cache(languages, val_audio, decode_fn)

    iters: List[int] = []
    wers: List[float] = []

    # Local callback
    step = {"k": 0}
    def _cb(_state=None):
        step["k"] += 1
        k = step["k"]
        if k % eval_every != 0:
            return {}
        pred_langs = list(predict_lang_fn(model, X_val))
        hyps = [transcripts_cache[pred_langs[i]][i] for i in range(len(ref_texts))]
        wer = 100.0 * corpus_wer_mixed_lang(ref_texts, hyps, ref_langs)
        iters.append(k)
        wers.append(wer)
        return {"val_wer": wer}

    # Run ADMM; try callback kwarg
    try:
        _ = admm_fn(model, cronos_params, callback=_cb)
    except TypeError:
        # No callback support; compute only a final point
        _ = admm_fn(model, cronos_params)
        pred_langs = list(predict_lang_fn(model, X_val))
        hyps = [transcripts_cache[pred_langs[i]][i] for i in range(len(ref_texts))]
        wer = 100.0 * corpus_wer_mixed_lang(ref_texts, hyps, ref_langs)
        iters.append(0)
        wers.append(wer)

    # Plot with seaborn
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame({"ADMM iteration": iters, "WER (%)": wers})
    ax = sns.lineplot(data=df, x="ADMM iteration", y="WER (%)", marker="o")
    ax.set_title(title)
    ax.set_xlabel("ADMM iteration")
    ax.set_ylabel("WER (%)")
    fig = ax.get_figure()
    fig.tight_layout()

    # Save
    csv_path = os.path.join(out_dir, csv_name)
    png_path = os.path.join(out_dir, png_name)
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    return df, fig

