# Session: Hinglish Code-Mix Conversion (Abuse-Preserving)

## What was wrong

Previous “Hinglish” data in `hinglish_data/` was **ITRANS script transliteration** (Devanagari → Roman spelling like `kyoM`, `bhI.Da`), not natural Hindi–English **code-mixing**.

Example of the old method:  
`ये कुर्सी यहाँ पड़ी है` → `ye kursI yahAM pa.DI hai`

Target: WhatsApp-style mix, e.g. `Yeh chair yahan padi hai with sundarta` / `Ye kursi yahan padi hai looking sundar`.

## What we built

| Piece | Path |
|--------|------|
| Converter (Bedrock Mantle + gates) | [`aws_train/codemix_convert_bedrock.py`](aws_train/codemix_convert_bedrock.py) |
| Deps | [`aws_train/requirements_codemix.txt`](aws_train/requirements_codemix.txt) |
| Full-run supervisor (auto-resume) | [`aws_train/run_codemix_full.sh`](aws_train/run_codemix_full.sh) |
| Outputs | `codemix_hinglish/` (does **not** overwrite `hinglish_data/`) |

**Model:** `qwen.qwen3-32b` via **Bedrock Mantle** Chat Completions  
`https://bedrock-mantle.us-east-1.api.aws/v1`  
(`ap-south-1` was daily-token throttled; Mantle host open-weights there, not Llama.)

**Auth:** short-lived Bedrock token from `aws login` via `aws-bedrock-token-generator`.

## Abuse preservation (not prompt-only)

1. **Framing + few-shot** including abusive examples; retries inject required Roman swear tokens.  
2. **Hard rejects:** refusals, length collapse, leftover Devanagari, missing abuse lexicon stems.  
3. **Score gate:** same ONNX MACD MiniLM scorer on Hindi vs Hinglish; accept abusive rows only if  
   `hinglish_p_abuse >= hindi_p_abuse - 0.15`.  
   Failures → `*_quarantine.csv` (never silently accept sanitized text).

## Pilot results (300 stratified train rows)

After stronger quarantine reprocess:

- Accepted: **285 / 300** (95%)
- Abusive accept rate: **135 / 150 = 90%** (meets plan target)
- Mean score delta: slightly **negative** (Hinglish often scored as abusive as / more than Hindi)
- Residual quarantine: **15** (mostly model softening on short/masked abuse)

Artifacts: `codemix_hinglish/pilot/`

## Full MACD conversion (cloud)

Two ways to run (same converter + abuse gates):

### A) This Cursor cloud agent / any Linux host

```bash
# 1) AWS auth (browser device code)
aws login --remote
export AWS_REGION=us-east-1
export S3_SCORER_URI=s3://suraksha-hinglish-439446323592-20260908170949/run-20260908170949/output/custom-macd-model

# 2) deps once
python3 -m venv .venv_codemix && .venv_codemix/bin/pip install -r aws_train/requirements_codemix.txt

# 3) supervised chunked full run (resumes via row_id)
bash aws_train/run_codemix_full.sh
```

### B) Modal (durable remote workers)

```bash
modal secret create aws-bedrock \
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  AWS_DEFAULT_REGION=us-east-1 \
  S3_SCORER_URI=s3://.../custom-macd-model \
  CODEMIX_MODEL=qwen.qwen3-32b CODEMIX_REGION=us-east-1

modal run aws_train/modal_codemix.py --pilot
modal run aws_train/modal_codemix.py --full --split train --limit 500 --loops 40
```

Outputs land in `codemix_hinglish/` (agent) or Modal volume `surakshanet-codemix` at `/vol/codemix_hinglish`.

Monitor (agent):

```bash
tail -f codemix_hinglish/full_convert.log
wc -l codemix_hinglish/train_meta.jsonl   # train done ≈ 20183
```

Outputs when done: `codemix_hinglish/{train,val,test}.csv`, `*_quarantine.csv`, `*_report.json`.

**Note:** `*.onnx` is gitignored — scorer is pulled from `S3_SCORER_URI` on first run.

## Research hook (HinGE / Eval4NLP PDF)

Paper file `2021.eval4nlp-1.20.pdf` is **HinGE** (Srivastava & Singh, ACL 2020): code-mixing ≠ transliteration; matrix=Hindi, embed=English; prefer **DCM/RA** human ratings over BLEU on CM text.

**Next research step (not done this session):** retrain MiniLM on code-mix CSVs vs ITRANS baseline; report F1 + abuse-preservation rate on WhatsApp-style Hinglish.

## Out of scope this session

- Classifier retrain / ONNX re-export  
- Editing the old ITRANS cache  
- Custom Bedrock model deploy (used hosted Mantle open-weight)
