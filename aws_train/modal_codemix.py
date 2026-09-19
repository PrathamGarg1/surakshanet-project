"""Modal cloud runner for abuse-preserving Hindi→Hinglish code-mix conversion.

Requires:
  modal secret create aws-bedrock \\
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1 \\
    [AWS_SESSION_TOKEN=...]

Optional env on secret:
  S3_SCORER_URI=s3://bucket/prefix/custom-macd-model/   # ONNX scorer folder
  CODEMIX_MODEL=qwen.qwen3-32b
  CODEMIX_REGION=us-east-1

Usage:
  modal run aws_train/modal_codemix.py --pilot
  modal run aws_train/modal_codemix.py --full
  modal run aws_train/modal_codemix.py --full --split train --limit 500
"""

from __future__ import annotations

import modal

app = modal.App("surakshanet-codemix")

vol = modal.Volume.from_name("surakshanet-codemix", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3>=1.34",
        "botocore[crt]>=1.34",
        "openai>=1.40",
        "aws-bedrock-token-generator>=0.1",
        "pandas>=2.0",
        "numpy>=1.24",
        "onnxruntime>=1.16",
        "transformers>=4.40",
        "requests>=2.31",
        "awscrt",
    )
    .add_local_file(
        "aws_train/codemix_convert_bedrock.py",
        remote_path="/root/codemix_convert_bedrock.py",
    )
)


def _sync_scorer_from_s3(local_dir: str, s3_uri: str) -> None:
    import subprocess
    from pathlib import Path

    p = Path(local_dir)
    onnx = p / "model_quantized.onnx"
    if onnx.exists() or (p / "onnx" / "model_quantized.onnx").exists():
        return
    if not s3_uri:
        raise RuntimeError(
            "Scorer ONNX missing and S3_SCORER_URI not set. "
            "Upload custom-macd-model (with model_quantized.onnx) to S3 and set the URI."
        )
    p.mkdir(parents=True, exist_ok=True)
    print(f"syncing scorer from {s3_uri} -> {local_dir}", flush=True)
    subprocess.check_call(["aws", "s3", "sync", s3_uri.rstrip("/") + "/", local_dir])
    if not onnx.exists() and not (p / "onnx" / "model_quantized.onnx").exists():
        raise RuntimeError(f"After sync, no model_quantized.onnx under {local_dir}")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-bedrock")],
    volumes={"/vol": vol},
    timeout=60 * 60 * 6,
    cpu=4,
    memory=8192,
)
def convert_chunk(
    split: str,
    limit: int = 200,
    workers: int = 4,
    delta: float = 0.15,
    max_retries: int = 2,
    pilot: bool = False,
    pilot_n: int = 300,
) -> dict:
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    # Import after path setup
    import codemix_convert_bedrock as C

    out = Path("/vol/codemix_hinglish")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("/vol/macd_hindi_cache")
    scorer_dir = Path("/vol/scorer/custom-macd-model-hinglish")
    scorer_dir.mkdir(parents=True, exist_ok=True)

    # Prefer volume-cached scorer; else pull from S3
    s3_uri = os.environ.get("S3_SCORER_URI", "").strip()
    # Also try previous training output layout if unset
    if not s3_uri and os.environ.get("S3_OUTPUT_URI"):
        base = os.environ["S3_OUTPUT_URI"].rstrip("/")
        s3_uri = f"{base}/custom-macd-model"
    _sync_scorer_from_s3(str(scorer_dir), s3_uri)

    region = os.environ.get("CODEMIX_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    model = os.environ.get("CODEMIX_MODEL", "qwen.qwen3-32b")
    os.environ.setdefault("AWS_REGION", region)

    print(f"Loading scorer from {scorer_dir}", flush=True)
    scorer = C.AbuseScorer(scorer_dir)
    print(f"Mantle region={region} model={model}", flush=True)
    client = C.MantleClient(region, model)

    if pilot:
        frames = [C.load_macd_split("train", cache)]
        all_df = frames[0]
        sample = C.stratified_pilot(all_df, pilot_n, 42)
        report = C.run_conversion(
            sample,
            out / "pilot",
            client,
            scorer,
            delta,
            max_retries,
            0.05,
            tag="pilot",
            workers=workers,
        )
        vol.commit()
        return report

    part = C.load_macd_split(split, cache)
    if limit and limit > 0:
        done = C.load_done_ids(out / f"{split}.csv") | C.load_done_ids(
            out / f"{split}_quarantine.csv"
        )
        todo = part[~part["row_id"].astype(str).isin(done)].head(limit)
        part = todo.reset_index(drop=True)
        print(f"{split}: converting up to {len(part)} new rows", flush=True)

    report = C.run_conversion(
        part,
        out,
        client,
        scorer,
        delta,
        max_retries,
        0.05,
        tag=split,
        workers=workers,
    )
    vol.commit()
    return report


@app.local_entrypoint()
def main(
    full: bool = False,
    pilot: bool = False,
    split: str = "train",
    limit: int = 200,
    workers: int = 4,
    loops: int = 1,
):
    """Run one or more conversion chunks on Modal."""
    if pilot:
        print(convert_chunk.remote(split="train", pilot=True, workers=workers))
        return

    if not full and not split:
        raise SystemExit("Pass --pilot or --full (optionally --split/--limit/--loops)")

    splits = ["train", "val", "test"] if full and split == "all" else [split]
    if full and split in ("train", "val", "test"):
        splits = [split]

    # Default full = keep chunking one split until caller stops; loops controls batches
    for i in range(max(1, loops)):
        for s in splits:
            print(f"=== loop {i+1}/{loops} split={s} limit={limit} ===")
            report = convert_chunk.remote(
                split=s,
                limit=limit,
                workers=workers,
                pilot=False,
            )
            print(report)
