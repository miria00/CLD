## jaxcld

`jaxcld` is a lightweight language-detection module for multilingual ASR models (Whisper / MMS). It provides an `ASRModel` wrapper plus pluggable language detection heads you can attach at inference time. This package also ships our novel CVXNN detection head, implemented in JAX and optimized via ADMM for high performance and low latency.

## Install

```bash
pip install jaxcld
```

If you are developing from source:

```bash
pip install -e .
```

## Using the package (minimal inference example)

```python
import numpy as np

from jaxcld import ASRModel, CVXNNLangDetectHead, NNLangDetectHead, SVMLangDetectHead

# 1) Load the base ASR model
languages = ["en", "hi", "id", "ms", "zh"]
asr = ASRModel.from_pretrained("openai/whisper-small", config={"languages": languages})

# 2) Load a language detection head artifact (choose ONE)
# head = CVXNNLangDetectHead.load("path/to/whisper-small_trained_cvx_mlp.pkl", asr)
# head = NNLangDetectHead.load("path/to/openai_whisper-small_nn_head.pkl", asr)
# head = SVMLangDetectHead.load("path/to/openai_whisper-small_linear_svm.pkl", asr)

# 3) Attach head and run inference
asr.set_lang_detect_head(head)

audio_16k_mono: np.ndarray = ...  # shape (T,), sampling rate 16kHz
pred_langs, pred_texts = asr.predict(audio_16k_mono)
print(pred_langs[0], pred_texts[0])
```

## Notes

- Head artifacts (`*.pkl`) are produced by training scripts in the source repository; this pip README intentionally focuses only on **package usage**.

