#!/usr/bin/env python3
"""Local fast path (M1 MPS): same English+Hindi+Hinglish train + Hinglish test metrics + ONNX."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

SEED = 42
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ID2LABEL = {0: "abusive", 1: "non-abusive"}
LABEL2ID = {"abusive": 0, "non-abusive": 1}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "aws_train" / "local_out"
CACHE = ROOT / "aws_train" / "hinglish_cache"
PT_DIR = OUT / "pt"
ONNX_FP32 = OUT / "onnx_fp32"
ONNX_INT8 = OUT / "onnx_int8"
EXT_DEST = ROOT / "assets" / "models" / "custom-macd-model"


def main() -> None:
    import numpy as np
    import torch
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

    t0 = time.time()
    for p in (OUT, CACHE, PT_DIR, ONNX_FP32, ONNX_INT8):
        p.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device=", device, flush=True)

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

    davidson = davidson.map(convert_labels).rename_column("tweet", "text")
    rem = [c for c in davidson.column_names if c not in ["text", "label"]]
    davidson = davidson.remove_columns(rem).train_test_split(test_size=0.10, seed=SEED)
    davidson_train, davidson_holdout = davidson["train"], davidson["test"]

    diacritic = re.compile(r"[\u0300-\u036f]")

    def to_hinglish(text: str) -> str:
        if not text:
            return text
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        return diacritic.sub("", roman).replace("~N", "n").replace("R^i", "ri")

    def add_hinglish(example):
        return {"text": to_hinglish(example["text"]), "label": example["label"]}

    def load_or_build(name, ds):
        path = CACHE / f"{name}.csv"
        if path.exists() and path.stat().st_size > 0:
            print("cache", path, flush=True)
            return load_dataset("csv", data_files=str(path))["train"]
        print("transliterating", name, len(ds), flush=True)
        out = ds.map(add_hinglish, desc=f"hinglish-{name}")
        out.to_csv(str(path), index=False)
        return out

    macd_train_h = load_or_build("train", macd_train)
    macd_val_h = load_or_build("val", macd_val)
    macd_test_h = load_or_build("test", macd_test)
    print("sample", macd_train[0], "=>", macd_train_h[0], flush=True)

    train_ds = concatenate_datasets([davidson_train, macd_train, macd_train_h]).shuffle(seed=SEED)
    val_ds = concatenate_datasets([macd_val, macd_val_h]).shuffle(seed=SEED)
    print("train", len(train_ds), "val", len(val_ds), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(ex):
        return tokenizer(ex["text"], truncation=True)

    tokenized_train = train_ds.map(tok, batched=True)
    tokenized_val = val_ds.map(tok, batched=True)
    tokenized_hi = macd_test.map(tok, batched=True)
    tokenized_hinglish = macd_test_h.map(tok, batched=True)
    tokenized_en = davidson_holdout.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def compute_metrics(eval_predictions):
        logits, labels = eval_predictions
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        }

    use_fp16 = device == "cuda"
    args = TrainingArguments(
        str(OUT / "trainer_out"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=use_fp16,
        logging_steps=50,
        seed=SEED,
        dataloader_num_workers=0,
        use_mps_device=device == "mps",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()

    metrics = {
        "val_hi_plus_hinglish": trainer.evaluate(),
        "macd_devanagari_test": trainer.predict(tokenized_hi).metrics,
        "macd_hinglish_test": trainer.predict(tokenized_hinglish).metrics,
        "davidson_english_holdout": trainer.predict(tokenized_en).metrics,
    }
    print(json.dumps(metrics, indent=2), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    trainer.save_model(str(PT_DIR))
    tokenizer.save_pretrained(str(PT_DIR))

    ort_model = ORTModelForSequenceClassification.from_pretrained(str(PT_DIR), export=True)
    ort_model.save_pretrained(str(ONNX_FP32))
    AutoTokenizer.from_pretrained(str(PT_DIR)).save_pretrained(str(ONNX_FP32))
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    ORTQuantizer.from_pretrained(ort_model).quantize(
        save_dir=str(ONNX_INT8), quantization_config=qconfig
    )
    AutoTokenizer.from_pretrained(str(PT_DIR)).save_pretrained(str(ONNX_INT8))

    if EXT_DEST.exists():
        shutil.rmtree(EXT_DEST)
    shutil.copytree(ONNX_INT8, EXT_DEST)
    (EXT_DEST / "onnx").mkdir(exist_ok=True)
    for f in EXT_DEST.glob("*.onnx"):
        shutil.copy2(f, EXT_DEST / "onnx" / f.name)

    done = {"elapsed_sec": time.time() - t0, "metrics": metrics, "device": device}
    (OUT / "DONE.json").write_text(json.dumps(done, indent=2))
    print("DONE", done["elapsed_sec"], "s installed ->", EXT_DEST, flush=True)


if __name__ == "__main__":
    main()
