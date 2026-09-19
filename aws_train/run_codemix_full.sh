#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/codemix_hinglish/full_convert.log"
PIDFILE="$ROOT/codemix_hinglish/full_convert.pid"
mkdir -p "$ROOT/codemix_hinglish"
export AWS_REGION="${AWS_REGION:-us-east-1}"
PY="$ROOT/.venv_codemix/bin/python"
echo "$$" > "$PIDFILE"
echo "[$(date -Iseconds)] supervisor start" >> "$LOG"

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
      --workers 3 --sleep 0.05 --max-retries 2 --limit 200 >> "$LOG" 2>&1 || {
        echo "[$(date -Iseconds)] chunk failed (will retry)" >> "$LOG"
        sleep 10
      }
  done
  sleep 2
done
rm -f "$PIDFILE"
