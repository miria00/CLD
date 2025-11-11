import argparse
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import wandb
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

from solve.utils.whisper_dataloader import load_data
from solve.models.cvx_relu_mlp import CVX_ReLU_MLP
from solve.optimizers.admm import admm
from wer_plot import run_cvxnn_wer_plot
import jax


def make_decode_fn(model_name: str):
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception:
        # Fallback to default openai checkpoint
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small",
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    model.config.forced_decoder_ids = None
    device = next(iter(model.parameters())).device

    def _decode(audio: Any, language: str) -> str:
        # Accept HF audio dicts or raw arrays
        if isinstance(audio, dict) and "array" in audio:
            arr = audio["array"]
            sr = audio.get("sampling_rate", 16000)
        else:
            arr = np.asarray(audio)
            sr = 16000

        inputs = processor(arr, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=model.dtype)
        forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        with torch.no_grad():
            pred_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_ids,
            )
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return text

    return _decode


def build_predict_lang_fn(target_lang: str, languages: Sequence[str]):
    other_lang = languages[1] if languages[0] == target_lang else languages[0]

    def _predict_lang(model: Any, X_val: np.ndarray) -> Sequence[str]:
        # Ensure weights exist (ADMM updates these each iter)
        if getattr(model, "theta1", None) is None or getattr(model, "theta2", None) is None:
            # Fallback: predict target for all until weights available
            return [str(target_lang)] * int(X_val.shape[0])
        W1 = np.asarray(model.theta1)
        w2 = np.asarray(model.theta2)
        scores = np.maximum(0.0, X_val @ W1) @ w2  # shape (n,)
        # Scores > 0 => target_lang (per dataloader label convention), else other_lang
        return [str(target_lang) if s > 0 else str(other_lang) for s in np.ravel(scores)]

    return _predict_lang


def main():
    p = argparse.ArgumentParser()
    # Mirror cronos_trainer.py interface
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--target_lang', type=str, required=False, default='en')
    p.add_argument('--output_dir', type=str, required=True)
    # WER/Whisper-specific
    p.add_argument('--lang1', type=str, default="en")
    p.add_argument('--lang2', type=str, default="zh")
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--title', type=str, default="WER vs ADMM iteration — Whisper + cvxNN (binary)")
    args = p.parse_args()

    model_name = args.model_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Match cronos_trainer defaults
    cronos_params: Dict[str, Any] = dict(
        rank=20, beta=0.001, rho=0.0001,
        gamma_ratio=1, admm_iters=6, pcg_iters=32,
        check_opt=False
    )
    adamW_params: Dict[str, Any] = dict(optimizer='AdamW', gamma=10**-4, n_epoch=30, batch_size=1024)

    # wandb setup
    wandb.init(
        project="CLD",
        name=f"cronos_wer_plot_{model_name}",
        config={
            "model_name": model_name,
            "cronos_params": cronos_params,
            "adamW_params": adamW_params,
            "output_dir": output_dir,
            "rank": cronos_params["rank"],
            "beta": cronos_params["beta"],
            "rho": cronos_params["rho"],
            "gamma_ratio": cronos_params["gamma_ratio"],
            "admm_iters": cronos_params["admm_iters"],
            "pcg_iters": cronos_params["pcg_iters"],
            "optimizer": adamW_params["optimizer"],
            "learning_rate": adamW_params["gamma"],
            "n_epoch": adamW_params["n_epoch"],
            "batch_size": adamW_params["batch_size"],
            "model_name": args.model_name,
            "lang1": args.lang1,
            "lang2": args.lang2,
            "target_lang": args.target_lang,
            "eval_every": args.eval_every,
        }
    )

    # Load features for train/valid (as in defrun)
    X_tr, y_tr = load_data(args.data_dir, args.target_lang, caller_script="defrun", dataset_split="train")
    X_val, y_val = load_data(args.data_dir, args.target_lang, caller_script="defrun", dataset_split="valid")

    # Build cvxNN model
    model = CVX_ReLU_MLP(X_tr, y_tr, cronos_params.get('P_S', 10), cronos_params['beta'], cronos_params['rho'], seed=jax.random.PRNGKey(0))
    model.init_model()
    model.Xtst = X_val
    model.ytst = y_val

    # Prepare validation audio/text/langs for WER
    ds = load_from_disk(args.data_dir)
    val_ds = ds["valid"]
    ref_texts: List[str] = [ex["text"] for ex in val_ds]
    ref_langs: List[str] = [str(ex["lang"]).lower() for ex in val_ds]
    val_audio: List[Any] = [ex["audio"] for ex in val_ds]  # HF audio dicts
    languages: List[str] = [str(args.lang1).lower(), str(args.lang2).lower()]

    # Build decode and predict adapters
    decode_fn = make_decode_fn(args.model_name)
    predict_lang_fn = build_predict_lang_fn(args.target_lang.lower(), languages)

    # Run ADMM with per-iteration WER callback
    start_time = time.time()
    df, fig = run_cvxnn_wer_plot(
        admm_fn=admm,
        model=model,
        cronos_params=cronos_params,
        X_val=np.asarray(X_val),
        ref_texts=ref_texts,
        ref_langs=ref_langs,
        val_audio=val_audio,
        languages=languages,
        decode_fn=decode_fn,
        predict_lang_fn=predict_lang_fn,
        out_dir=output_dir,
        eval_every=args.eval_every,
        title=args.title,
        csv_name="wer_cvxnn.csv",
        png_name="wer_cvxnn.png",
    )
    elapsed = time.time() - start_time

    # Log to wandb
    if not df.empty:
        # Log series as a table and last point as a scalar
        table = wandb.Table(dataframe=df)
        wandb.log({"wer_vs_iter": table})
        wandb.log({"final/wer_percent": float(df["WER (%)"].iloc[-1])})
    wandb.log({"training_time": elapsed})
    wandb.save(os.path.join(output_dir, "wer_cvxnn.csv"))
    wandb.save(os.path.join(output_dir, "wer_cvxnn.png"))
    wandb.finish()


if __name__ == "__main__":
    main()

