#!/usr/bin/env bash
# Cloud-agent / any Linux host supervisor for full MACD code-mix conversion.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/codemix_hinglish/full_convert.log"
PIDFILE="$ROOT/codemix_hinglish/full_convert.pid"
mkdir -p "$ROOT/codemix_hinglish"

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
# Previous hinglish train output (override if needed)
export S3_OUTPUT_URI="${S3_OUTPUT_URI:-s3://suraksha-hinglish-439446323592-20260908170949/run-20260908170949/output}"
export S3_SCORER_URI="${S3_SCORER_URI:-${S3_OUTPUT_URI%/}/custom-macd-model}"

if [[ -x "$ROOT/.venv_codemix/bin/python" ]]; then
  PY="$ROOT/.venv_codemix/bin/python"
else
  PY="${PYTHON:-python3}"
fi

echo "$$" > "$PIDFILE"
echo "[$(date -Iseconds)] cloud supervisor start region=$AWS_REGION scorer=$S3_SCORER_URI" >> "$LOG"

# Fail fast if no AWS creds
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] ERROR: no AWS credentials. Run: aws login --remote" >> "$LOG"
  echo "No AWS credentials. Run: aws login --remote" >&2
  exit 2
fi

done_split() {
  local s="$1"
  local meta="$ROOT/codemix_hinglish/${s}_meta.jsonl"
  local target="$2"
  [[ -f "$meta" ]] || return 1
  local n
  n=$(wc -l < "$meta" | tr -d ' ')
  [[ "$n" -ge "$target" ]]
}

while true; do
  if done_split train 20183 && done_split val 6728 && done_split test 6728; then
    echo "[$(date -Iseconds)] all splits complete — exiting" >> "$LOG"
    break
  fi
  for split in train val test; do
    case "$split" in
      train) target=20183 ;;
      val|test) target=6728 ;;
    esac
    if done_split "$split" "$target"; then
      continue
    fi
    echo "[$(date -Iseconds)] chunk $split" >> "$LOG"
    "$PY" -u aws_train/codemix_convert_bedrock.py --full --splits "$split" \
      --workers "${WORKERS:-4}" --sleep 0.05 --max-retries 2 --limit "${LIMIT:-200}" \
      --region "$AWS_REGION" >> "$LOG" 2>&1 || {
        echo "[$(date -Iseconds)] chunk failed (will retry)" >> "$LOG"
        sleep 15
      }
  done
  sleep 2
done
rm -f "$PIDFILE"
