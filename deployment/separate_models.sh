#!/usr/bin/env bash
# Task 7 + 10 — serve the base and fine-tuned models as two distinct backends.
#
# Today both service definitions reference the same GGUF filename and the
# reconciliation script resolves both names to one file, so the Section 15
# comparison is a model measured against itself. Two files, two backends, one
# router: after this, changing the model name is genuinely the only variable.
#
# Run on CT 200.
set -euo pipefail

MODELS_DIR=${MODELS_DIR:-/models}
BASE_NAME=qwen2.5-3b-instruct-base-q4_k_m.gguf
FT_NAME=qwen2.5-3b-taskextract-ft-q4_k_m.gguf
BASE_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

echo "== disk before =="
df -h "$MODELS_DIR" | tail -1
echo "NOTE: the thin pool backs all containers. Run 'pct fstrim 200' from the host"
echo "      before a large download, or the pool fills and the container goes read-only."

cd "$MODELS_DIR"

# 1. the fine-tuned export must stop sharing the upstream filename
if [ -f qwen2.5-3b-instruct-q4_k_m.gguf ] && [ ! -f "$FT_NAME" ]; then
  echo "== renaming the fine-tuned export =="
  mv -v qwen2.5-3b-instruct-q4_k_m.gguf "$FT_NAME"
fi

# 2. fetch the genuine base model, if it is not already present
if [ ! -f "$BASE_NAME" ]; then
  echo "== downloading the base model =="
  curl -fL --retry 3 -o "$BASE_NAME.part" "$BASE_URL"
  mv "$BASE_NAME.part" "$BASE_NAME"
fi

# 3. they must not be the same bytes -- this is the check that was missing
echo "== distinctness =="
for f in "$BASE_NAME" "$FT_NAME"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
  printf '%-48s %12d bytes  %s\n' "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -c1-16)"
done
if [ "$(sha256sum < "$BASE_NAME")" = "$(sha256sum < "$FT_NAME")" ]; then
  echo "FAIL: both files have the same hash. They are one model under two names."
  exit 1
fi
echo "OK: two distinct model files."

# 4. one llama-server per model, then the router in front of both
write_unit () {
  local unit=$1 model=$2 alias=$3 port=$4 mem=$5
  cat > "/etc/systemd/system/${unit}.service" <<UNIT
[Unit]
Description=llama.cpp server - ${alias}
After=network-online.target

[Service]
Type=simple
Environment=LD_LIBRARY_PATH=/opt/llama/llama.cpp/build/bin
ExecStart=/opt/llama/llama.cpp/build/bin/llama-server \\
  --model ${MODELS_DIR}/${model} \\
  --alias ${alias} --host 127.0.0.1 --port ${port} \\
  --ctx-size 16384 --parallel 2 --jinja --threads 4 --no-webui --cache-reuse 256
Restart=on-failure
RestartSec=5
MemoryMax=${mem}

[Install]
WantedBy=multi-user.target
UNIT
}

echo "== writing units =="
write_unit llama-base  "$BASE_NAME" qwen3b-base  8082 4500M
write_unit llama-ft    "$FT_NAME"   qwen3b-ft    8083 4500M
systemctl daemon-reload
systemctl enable --now llama-base llama-ft
sleep 25

echo "== both backends answering =="
for p in 8082 8083; do
  printf 'port %s: ' "$p"; curl -s -m 20 "http://127.0.0.1:${p}/health" || echo unreachable
  echo
done

echo "== router =="
curl -s -m 15 http://127.0.0.1:8080/v1/models | python3 -m json.tool | head -30

cat <<'NEXT'

Evidence to capture for Task 7:
  curl -s http://127.0.0.1:8080/v1/models            # both names listed
  python3 3_test_chat_template_real.py /models/qwen2.5-3b-taskextract-ft-q4_k_m.gguf \
      --training-template chat_template.txt

Then re-run the Task 10 comparison through the router, changing only the model name.
With two genuinely different files, a difference between the arms is attributable to
fine-tuning. Until then it is not.
NEXT
