#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX="$ROOT/.runtime"
DEPLOY="$ROOT/third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
export TensorRT_ROOT="$ROOT/.native/tensorrt"
export onnxruntime_ROOT="$ROOT/.native/onnxruntime-linux-x64-1.22.0"
CUDA_RUNTIME="$PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime"
CUDA_COMPILER="$PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc"
"$PREFIX/bin/python" "$ROOT/scripts/patch_sonic.py"
"$PREFIX/bin/cmake" -S "$DEPLOY" -B "$DEPLOY/build" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_CXX_FLAGS="-I$CUDA_RUNTIME/include -I$CUDA_COMPILER/include -I$PREFIX/include" \
  -DCUDA_RUNTIME_LIBRARY="$CUDA_RUNTIME/lib/libcudart.so.12" \
  -DCUDA_INCLUDE_DIR="$CUDA_RUNTIME/include" \
  -DTensorRT_nvinfer_LIBRARY="$TensorRT_ROOT/lib/libnvinfer.so" \
  -DTensorRT_nvinfer_plugin_LIBRARY="$TensorRT_ROOT/lib/libnvinfer_plugin.so" \
  -DTensorRT_nvonnxparser_LIBRARY="$TensorRT_ROOT/lib/libnvonnxparser.so" \
  -DTensorRT_nvparsers_LIBRARY="$TensorRT_ROOT/lib/libnvonnxparser.so"
"$PREFIX/bin/cmake" --build "$DEPLOY/build" --target g1_deploy_onnx_ref -j4
