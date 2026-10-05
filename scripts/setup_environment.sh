#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX="$ROOT/.runtime"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
"${CONDA:-conda}" create -y -p "$PREFIX" --override-channels -c conda-forge python=3.11 pip cmake eigen=3.4 gtest cppzmq msgpack-cxx libboost-headers nlohmann_json git-lfs zlib mujoco=3.4 numpy scipy pyyaml matplotlib imageio imageio-ffmpeg huggingface_hub
export PATH="$PREFIX/bin:$PATH"
git -C "$ROOT/third_party/GR00T-WholeBodyControl" lfs install --local
git -C "$ROOT/third_party/GR00T-WholeBodyControl" lfs pull --include='gear_sonic_deploy/g1/meshes/**,decoupled_wbc/control/robot_model/model_data/g1/meshes/**,gear_sonic_deploy/reference/example/**,gear_sonic_deploy/thirdparty/unitree_sdk2/lib/x86_64/**,gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64/**'
"$PREFIX/bin/python" -m pip install --extra-index-url https://pypi.nvidia.com 'tensorrt-cu12==10.9.0.34' nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12
"$PREFIX/bin/python" "$ROOT/scripts/prepare_native.py"
"$PREFIX/bin/python" "$ROOT/third_party/GR00T-WholeBodyControl/download_from_hf.py" --sonic-v1-1
