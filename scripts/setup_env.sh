#!/usr/bin/env bash
# One-shot environment setup for the SuperMimic reproduction.
#
#   bash scripts/setup_env.sh [ROOT]     # default ROOT=$HOME/supermimic
#
# Layout after setup:
#   $ROOT/mjlab            mjlab checkout pinned to the commit used for all results
#   $ROOT/mjlab/.venv      uv-managed venv (python 3.12, torch cu130, mujoco-warp)
#   $ROOT/supermimic       this repo, registered as an editable dependency of mjlab
#
# Notes for machines without GitHub access (bj27): clone mjlab elsewhere and
# rsync it to $ROOT/mjlab first; the script skips the clone if the dir exists.
# The PyPI index is switched to the Tencent mirror because pypi.org's CDN is
# unreachable/very slow from the Tencent VMs; drop that sed on other hosts.
set -euo pipefail
ROOT="${1:-$HOME/supermimic}"
MJLAB_COMMIT="8bb2bc9e187797689ec82c78c35019afd2fd0376"   # 2026-10-18
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$ROOT"

if ! command -v uv >/dev/null; then
  python3 -m pip install --user -q -i https://mirrors.cloud.tencent.com/pypi/simple/ uv
fi

if [ ! -d "$ROOT/mjlab" ]; then
  git clone https://github.com/mujocolab/mjlab.git "$ROOT/mjlab"
  git -C "$ROOT/mjlab" checkout "$MJLAB_COMMIT"
fi

cd "$ROOT/mjlab"
if grep -q "https://pypi.org/simple" pyproject.toml; then
  cp -n pyproject.toml pyproject.toml.orig
  sed -i "s#https://pypi.org/simple#https://mirrors.cloud.tencent.com/pypi/simple/#" pyproject.toml
  mv -n uv.lock uv.lock.orig || true
fi
# No --extra cu130: the default PyPI torch 2.14 wheel is already a cu130 build and
# the mirror serves it fast; download.pytorch.org is ~0.5 MB/s from here.
uv sync --python 3.12
if [ "$REPO_DIR" != "$ROOT/supermimic" ]; then
  mkdir -p "$ROOT/supermimic"
  rsync -a --exclude .git --exclude logs --exclude videos "$REPO_DIR/" "$ROOT/supermimic/"
fi
uv add --editable "$ROOT/supermimic"
uv run python -c "import torch, mjlab, supermimic; print('ok', torch.__version__, torch.cuda.get_device_name(0))"
echo "[setup] done. Train with:  cd $ROOT/mjlab && CUDA_VISIBLE_DEVICES=0 uv run python $ROOT/supermimic/scripts/train.py --motion_dir <npz dir> --run_name <name>"
