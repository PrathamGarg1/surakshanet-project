"""Modal GPU runner for FINAL SURAKSHANET HINGLISH — minimal delta from original notebook.

Usage:
  modal run aws_train/modal_train.py
"""

from __future__ import annotations

import modal

app = modal.App("surakshanet-hinglish")

vol = modal.Volume.from_name("surakshanet-notebook", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "scikit-learn", "sentencepiece")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "transformers==4.44.2",
        "datasets",
        "accelerate",
        "evaluate",
        "indic-transliteration",
    )
    .pip_install("onnx", "onnxruntime", "optimum[onnxruntime]==1.23.3")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/vol": vol},
)
def train():
    import json
    import re
    import shutil
    from pathlib import Path

    import numpy as np
    from datasets import concatenate_datasets, load_dataset
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    # --- same constants as FINAL SURAKSHANET.ipynb ---
    SEED = 42
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ID2LABEL = {0: "abusive", 1: "non-abusive"}
    LABEL2ID = {"abusive": 0, "non-abusive": 1}
    OUTPUT_DIR = "/vol/checkpoints/pt_hinglish"
    ONNX_FP32_DIR = "/vol/checkpoints/onnx_fp32_hinglish"
    ONNX_INT8_DIR = "/vol/checkpoints/onnx_int8_hinglish"
    HINGLISH_CACHE = Path("/vol/checkpoints/hinglish_cache")

    for p in (OUTPUT_DIR, ONNX_FP32_DIR, ONNX_INT8_DIR, str(HINGLISH_CACHE)):
        Path(p).mkdir(parents=True, exist_ok=True)

    macd_train = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_train.csv",
    )["train"]
    macd_val = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_val.csv",
    )["train"]
    macd_test = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_test.csv",
    )["train"]

    davidson = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
    )["train"]

    def convert_labels(example):
        return {"label": 0 if example["class"] in [0, 1] else 1}

    davidson = davidson.map(convert_labels)
    davidson = davidson.rename_column("tweet", "text")
    rem = [c for c in davidson.column_names if c not in ["text", "label"]]
    davidson = davidson.remove_columns(rem)
    davidson = davidson.train_test_split(test_size=0.10, seed=SEED)
    davidson_train = davidson["train"]
    davidson_holdout = davidson["test"]

    # --- minimal Hinglish add ---
    DEVANAGARI = re.compile(r"[\u0900-\u097F]+")
    _diacritic = re.compile(r"[\u0300-\u036f]")

    def to_hinglish(text: str) -> str:
        if not text:
            return text

        def repl(m):
            roman = transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.ITRANS)
            return _diacritic.sub("", roman)

        return DEVANAGARI.sub(repl, text)

    def add_hinglish(example):
        return {"text": to_hinglish(example["text"]), "label": example["label"]}

    def load_or_build(name, ds):
        cache = HINGLISH_CACHE / f"{name}.csv"
        if cache.exists() and cache.stat().st_size > 0:
            print("cache hit", cache, flush=True)
            return load_dataset("csv", data_files=str(cache))["train"]
        print("transliterating", name, "n=", len(ds), flush=True)
        out = ds.map(add_hinglish)
        out.to_csv(str(cache), index=False)
        return out

    macd_train_h = load_or_build("train", macd_train)
    macd_val_h = load_or_build("val", macd_val)
    macd_test_h = load_or_build("test", macd_test)
    print("sample hi:", macd_train[0], flush=True)
    print("sample hinglish:", macd_train_h[0], flush=True)

    dataset = concatenate_datasets(
        [davidson_train, macd_train, macd_train_h]
    ).shuffle(seed=SEED)
    macd_val_both = concatenate_datasets([macd_val, macd_val_h]).shuffle(seed=SEED)
    print("train", len(dataset), "val", len(macd_val_both), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_train = dataset.map(tokenize_function, batched=True)
    tokenized_val = macd_val_both.map(tokenize_function, batched=True)
    tokenized_macd_test = macd_test.map(tokenize_function, batched=True)
    tokenized_macd_test_h = macd_test_h.map(tokenize_function, batched=True)
    tokenized_davidson_holdout = davidson_holdout.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_predictions):
        logits, labels = eval_predictions
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }

    # --- EXACT same TrainingArguments style as FINAL SURAKSHANET.ipynb ---
    # (defaults: 3 epochs — do NOT override)
    training_args = TrainingArguments(
        "test-trainer",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("val:", trainer.evaluate(), flush=True)

    metrics = {
        "macd_devanagari_test": trainer.predict(tokenized_macd_test).metrics,
        "macd_hinglish_test": trainer.predict(tokenized_macd_test_h).metrics,
        "davidson_holdout": trainer.predict(tokenized_davidson_holdout).metrics,
    }
    print("METRICS", json.dumps(metrics, indent=2, default=str), flush=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_DIR) / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ONNX export (same as notebook)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR, export=True
    )
    ort_model.save_pretrained(ONNX_FP32_DIR)
    AutoTokenizer.from_pretrained(OUTPUT_DIR).save_pretrained(ONNX_FP32_DIR)

    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    quantizer.quantize(save_dir=ONNX_INT8_DIR, quantization_config=qconfig)
    AutoTokenizer.from_pretrained(OUTPUT_DIR).save_pretrained(ONNX_INT8_DIR)

    # extension-friendly copy
    ext = Path("/vol/checkpoints/custom-macd-model-hinglish")
    if ext.exists():
        shutil.rmtree(ext)
    shutil.copytree(ONNX_INT8_DIR, ext)
    (ext / "onnx").mkdir(exist_ok=True)
    for f in ext.glob("*.onnx"):
        shutil.copy2(f, ext / "onnx" / f.name)

    vol.commit()
    return metrics


@app.local_entrypoint()
def main():
    metrics = train.remote()
    print("DONE")
    print(json_dumps(metrics))


def json_dumps(obj):
    import json

    return json.dumps(obj, indent=2, default=str)
