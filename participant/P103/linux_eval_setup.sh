#!/usr/bin/env bash
# 在 Ubuntu 虚拟机中，从解压后的仓库根目录执行：bash participant/P103/linux_eval_setup.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ ! -f requirements-official.lock || ! -f scripts/evaluate_one.py ]]; then
  echo "请先解压完整 Linux 检验包，并从仓库中运行此脚本。" >&2
  exit 2
fi
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "当前 Linux 找不到 python3.12。请先提供 /etc/os-release 与 python3 --version 输出，再安装对应版本。" >&2
  cat /etc/os-release >&2
  python3 --version >&2 || true
  exit 2
fi

run_tag="$(date +%Y%m%d-%H%M%S)"
out_base="outputs/P103"
check_dir="$out_base/linux-check-$run_tag"
public_dir="$out_base/linux-public-E009-$run_tag"
v2_dir="$out_base/linux-v2-E009-$run_tag"
v3_dir="$out_base/linux-v3-E009-$run_tag"
for candidate in "$check_dir" "$public_dir" "$v2_dir" "$v3_dir"; do
  if [[ -e "$candidate" ]]; then
    echo "输出目录已存在，拒绝覆盖：$candidate" >&2
    exit 2
  fi
done
mkdir -p "$out_base"

{
  echo "=== OS ==="
  cat /etc/os-release
  echo "=== kernel ==="
  uname -a
  echo "=== Python ==="
  python3.12 --version
  echo "=== current submission artifact SHA-256 ==="
  sha256sum participant/P103/artifacts/policy.npz
} | tee "$out_base/linux-system-$run_tag.txt"

if [[ ! -x .venv-linux-p103/bin/python ]]; then
  python3.12 -m venv .venv-linux-p103
fi
python_bin="$repo_root/.venv-linux-p103/bin/python"
"$python_bin" -m pip install -r requirements-official.lock
"$python_bin" -m pip install --no-deps -e .
"$python_bin" -m pip check

"$python_bin" scripts/check_submission.py --submission participant/P103 --output "$check_dir"
"$python_bin" scripts/evaluate_one.py --submission participant/P103 \
  --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output "$public_dir"
"$python_bin" scripts/evaluate_one.py --submission participant/P103 \
  --suite participant/P103/validation-suite-v2.yaml --seeds participant/P103/validation-seeds-v2.json --output "$v2_dir"
"$python_bin" scripts/evaluate_one.py --submission participant/P103 \
  --suite participant/P103/validation-suite-v3.yaml --seeds participant/P103/validation-seeds-v3.json --output "$v3_dir"

"$python_bin" - "$check_dir" "$public_dir" "$v2_dir" "$v3_dir" <<'PY'
import json
from pathlib import Path
import sys

check, *evaluations = map(Path, sys.argv[1:])
audit = json.loads((check / "audit-report.json").read_text(encoding="utf-8"))
if audit.get("rejections"):
    raise SystemExit(f"提交预检存在硬拒绝：{audit['rejections']}")
for directory in evaluations:
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    episodes = sum(group["num_episodes"] for group in result["group_metrics"].values())
    print(f"{directory}: status={result['status']} provenance={result['provenance']} "
          f"episodes={episodes} score={result['performance_score']}")
    if result["status"] != "ok" or result["errors"]:
        raise SystemExit(f"评测失败：{directory}")
PY

echo "Linux 检查完成。结果目录见上面的路径；请保留原始 result.json 和 episodes.csv。"
