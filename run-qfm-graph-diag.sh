#!/usr/bin/env bash
# Diagnostics only. No compilation, installation, or global configuration edits.
set -eo pipefail
deps=${QFM_DEPS_PREFIX:-/mnt/qfm-ready.smywMV}
card=${1:-0}
[[ "$card" =~ ^[0-9]+$ ]] || { echo 'Usage: bash run-qfm-graph-diag.sh [physical-card]'; exit 2; }
proxy=${QFM_DOWNLOAD_PROXY:-http://10.155.96.5:8081}
base=https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/5518c439b6685535c9ffddd80c4f479d22191870/qfm_diagnostic
for vendor in qfm_stage1 qfm_meta_stage1; do
  test -f "$deps/opp/vendors/${vendor}_transformer/bin/set_env.bash" || { echo "Missing installed $vendor in $deps"; exit 2; }
done
test -d "$deps/python" || { echo 'Missing private Python installation'; exit 2; }
stage=$(mktemp -d /mnt/qfm-graph-diag.XXXXXX)
for file in qfm_probe.py planner.py run.py; do
  curl --insecure --fail --silent --show-error --location --proxy "$proxy" \
    --connect-timeout 15 --max-time 120 --retry 2 "$base/$file" -o "$stage/$file"
done
cd "$stage"
printf '%s\n' \
  'ac622f5f3a0bcbf1d7d1ee08d84c68be909a154d6c6de6e0ead0b7c8b29e09f6  qfm_probe.py' \
  '2ba6751e1207c7db2e8af10a08dcb4d0b34ab98f93ce5c083b139e66e91818eb  planner.py' \
  '7ce93220589ec9a41ce03cd1d05b259b31bb7e920b3f7f75fb3be758492f6b82  run.py' | sha256sum -c -
source "$deps/opp/vendors/qfm_stage1_transformer/bin/set_env.bash"
source "$deps/opp/vendors/qfm_meta_stage1_transformer/bin/set_env.bash"
export PYTHONPATH="$deps/python${PYTHONPATH:+:$PYTHONPATH}"
echo 'Only graph diagnostics; original FP32 accuracy failures remain failures.'
exec python3 run.py --card "$card" --suite graph --output-dir /mnt/qfm-graph-final-results
