#!/usr/bin/env python3
"""SurakshaNet: train MiniLM on English + Hindi + Hinglish, export ONNX INT8.

Hinglish = Devanagari→Roman via indic_transliteration (ITRANS, WhatsApp-style ASCII).
Evaluates separately on MACD Devanagari test AND Hinglish test.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEED = 42
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ID2LABEL = {0: "abusive", 1: "non-abusive"}
LABEL2ID = {"abusive": 0, "non-abusive": 1}

WORK = Path(os.environ.get("WORK_DIR", "/opt/suraksha"))
OUT = Path(os.environ.get("OUTPUT_DIR", str(WORK / "output")))
CACHE = WORK / "hinglish_cache"
PT_DIR = OUT / "pt"
ONNX_FP32 = OUT / "onnx_fp32"
ONNX_INT8 = OUT / "onnx_int8"
S3_URI = os.environ.get("S3_OUTPUT_URI", "")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def install_deps() -> None:
    # torch may already be present on the AMI session
    try:
        import torch  # noqa: F401
    except Exception:
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy", "transformers", "datasets", "accelerate", "evaluate",
        "scikit-learn", "sentencepiece", "indic-transliteration",
        "pandas", "pyarrow", "optimum[onnxruntime]", "onnx", "onnxruntime", "boto3",
    ])


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for p in (PT_DIR, ONNX_FP32, ONNX_INT8):
        p.mkdir(parents=True, exist_ok=True)

    marker = WORK / "status.json"
    marker.write_text(json.dumps({"status": "installing", "t": time.time()}))
    install_deps()

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

    marker.write_text(json.dumps({"status": "loading_data", "t": time.time()}))

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

    marker.write_text(json.dumps({"status": "transliterating", "t": time.time()}))
    # ITRANS → ASCII Roman (WhatsApp-like). Strip leftover diacritic marks if any.
    diacritic = re.compile(r"[\u0300-\u036f]")

    def to_hinglish(text: str) -> str:
        if not text:
            return text
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        roman = diacritic.sub("", roman)
        # common WhatsApp-ish cleanup
        roman = roman.replace("~N", "n").replace("R^i", "ri").replace(".D", "d")
        return roman

    def add_hinglish(example):
        return {"text": to_hinglish(example["text"]), "label": example["label"]}

    def load_or_build(name, ds):
        path = CACHE / f"{name}.csv"
        if path.exists() and path.stat().st_size > 0:
            print(f"cache hit {path}", flush=True)
            return load_dataset("csv", data_files=str(path))["train"]
        print(f"transliterating {name} n={len(ds)}", flush=True)
        out = ds.map(add_hinglish, desc=f"hinglish-{name}")
        out.to_csv(str(path), index=False)
        return out

    # wipe old empty/partial cache from failed run
    for p in CACHE.glob("*.csv"):
        if p.stat().st_size == 0:
            p.unlink()

    macd_train_h = load_or_build("train", macd_train)
    macd_val_h = load_or_build("val", macd_val)
    macd_test_h = load_or_build("test", macd_test)
    print("sample hi:", macd_train[0], flush=True)
    print("sample hinglish:", macd_train_h[0], flush=True)

    train_ds = concatenate_datasets(
        [davidson_train, macd_train, macd_train_h]
    ).shuffle(seed=SEED)
    val_ds = concatenate_datasets([macd_val, macd_val_h]).shuffle(seed=SEED)
    print(f"sizes train={len(train_ds)} val={len(val_ds)}", flush=True)

    marker.write_text(json.dumps({"status": "training", "t": time.time()}))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    tokenized_macd_test = macd_test.map(tokenize_function, batched=True)
    tokenized_macd_test_h = macd_test_h.map(tokenize_function, batched=True)
    tokenized_davidson = davidson_holdout.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_predictions):
        logits, labels = eval_predictions
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "precision_macro": float(
                precision_score(labels, preds, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(labels, preds, average="macro", zero_division=0)
            ),
        }

    args = TrainingArguments(
        str(WORK / "trainer_out"),
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
        fp16=False,
        logging_steps=50,
        seed=SEED,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()

    metrics = {
        "val_hi_plus_hinglish": trainer.evaluate(),
        "macd_devanagari_test": trainer.predict(tokenized_macd_test).metrics,
        "macd_hinglish_test": trainer.predict(tokenized_macd_test_h).metrics,
        "davidson_english_holdout": trainer.predict(tokenized_davidson).metrics,
    }
    print("METRICS", json.dumps(metrics, indent=2, default=str), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    trainer.save_model(str(PT_DIR))
    tokenizer.save_pretrained(str(PT_DIR))

    marker.write_text(json.dumps({"status": "onnx", "t": time.time()}))
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(PT_DIR), export=True
    )
    ort_model.save_pretrained(str(ONNX_FP32))
    AutoTokenizer.from_pretrained(str(PT_DIR)).save_pretrained(str(ONNX_FP32))

    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    quantizer.quantize(save_dir=str(ONNX_INT8), quantization_config=qconfig)
    AutoTokenizer.from_pretrained(str(PT_DIR)).save_pretrained(str(ONNX_INT8))

    ext = OUT / "custom-macd-model"
    if ext.exists():
        shutil.rmtree(ext)
    shutil.copytree(ONNX_INT8, ext)
    onnx_dir = ext / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    for f in list(ext.glob("*.onnx")):
        shutil.copy2(f, onnx_dir / f.name)

    elapsed = time.time() - t0
    done = {"status": "done", "elapsed_sec": elapsed, "metrics": metrics}
    (OUT / "DONE.json").write_text(json.dumps(done, indent=2, default=str))
    marker.write_text(json.dumps({"status": "done", "elapsed_sec": elapsed}))
    print("DONE in", elapsed, "s", flush=True)

    if S3_URI:
        import boto3

        assert S3_URI.startswith("s3://")
        _, _, rest = S3_URI.partition("s3://")
        bucket, _, prefix = rest.partition("/")
        prefix = prefix.rstrip("/")
        s3 = boto3.client("s3")
        for path in OUT.rglob("*"):
            if path.is_file():
                key = (
                    f"{prefix}/{path.relative_to(OUT)}"
                    if prefix
                    else str(path.relative_to(OUT))
                )
                print("upload", key, flush=True)
                s3.upload_file(str(path), bucket, key)
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/_SUCCESS" if prefix else "_SUCCESS",
            Body=b"ok",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        work = Path(os.environ.get("WORK_DIR", "/opt/suraksha"))
        work.mkdir(parents=True, exist_ok=True)
        err = {"status": "failed", "error": repr(e)}
        (work / "status.json").write_text(json.dumps(err))
        s3_uri = os.environ.get("S3_OUTPUT_URI", "")
        if s3_uri.startswith("s3://"):
            try:
                import boto3

                _, _, rest = s3_uri.partition("s3://")
                bucket, _, prefix = rest.partition("/")
                prefix = prefix.rstrip("/")
                key = f"{prefix}/FAILED.json" if prefix else "FAILED.json"
                boto3.client("s3").put_object(
                    Bucket=bucket, Key=key, Body=json.dumps(err).encode()
                )
            except Exception:
                pass
        raise
