"""Download native inference dependencies into the project runtime only."""
from pathlib import Path
import os
import subprocess
import sys
import tarfile
import urllib.request

root = Path(__file__).resolve().parents[1]
prefix = root / '.runtime'
native = root / '.native'
native.mkdir(exist_ok=True)
archive = native / 'onnxruntime.tgz'
urllib.request.urlretrieve('https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz', archive)
with tarfile.open(archive) as tf:
    tf.extractall(native, filter='data')
trt = native / 'tensorrt'
(trt / 'include').mkdir(parents=True, exist_ok=True)
(trt / 'lib').mkdir(exist_ok=True)
# Headers matching the TensorRT wheel. No system installation is performed.
for name in ['NvInfer.h', 'NvInferImpl.h', 'NvInferLegacyDims.h', 'NvInferPlugin.h', 'NvInferPluginBase.h', 'NvInferPluginUtils.h', 'NvInferRuntime.h', 'NvInferRuntimeBase.h', 'NvInferRuntimeCommon.h', 'NvInferRuntimePlugin.h', 'NvInferVersion.h', 'NvOnnxParser.h']:
    base = 'https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.9/include/'
    if name == 'NvOnnxParser.h':
        base = 'https://raw.githubusercontent.com/onnx/onnx-tensorrt/10.9-GA/'
    urllib.request.urlretrieve(base + name, trt / 'include' / name)
site = prefix / 'lib/python3.11/site-packages'
for lib in (site / 'tensorrt_libs').glob('*.so*'):
    target = trt / 'lib' / lib.name
    if not target.exists():
        target.symlink_to(lib)
    short = trt / 'lib' / (lib.name.split('.so')[0] + '.so')
    if not short.exists():
        short.symlink_to(lib)
dds = native / 'cyclonedds'
(dds / 'lib').mkdir(parents=True, exist_ok=True)
sdk = root / 'third_party/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty'
if not (dds / 'include').exists():
    (dds / 'include').symlink_to(sdk / 'include')
for lib in (sdk / 'lib/x86_64').glob('*'):
    target = dds / 'lib' / lib.name
    if not target.exists():
        target.symlink_to(lib)
full_dds = native / 'cyclonedds-full'
if not (full_dds / 'bin/idlc').exists():
    source = native / 'cyclonedds-source'
    if not source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', '0.10.2', 'https://github.com/eclipse-cyclonedds/cyclonedds.git', str(source)], check=True)
    # CycloneDDS 0.10.2 passes an incorrect remaining-buffer size to snprintf
    # in configuration logging. Modern fortified libc aborts on the second item.
    config_source = source / 'src/core/ddsi/src/ddsi_config.c'
    config_source.write_text(config_source.read_text().replace('snprintf (resp, 256,', 'snprintf (resp, sizeof (res) - (size_t) (resp - res),'))
    cmake = str(prefix / 'bin/cmake')
    build = native / 'cyclonedds-build'
    subprocess.run([cmake, '-S', str(source), '-B', str(build), f'-DCMAKE_INSTALL_PREFIX={full_dds}', '-DBUILD_EXAMPLES=OFF', '-DBUILD_TESTING=OFF'], check=True)
    subprocess.run([cmake, '--build', str(build), '-j4'], check=True)
    subprocess.run([cmake, '--install', str(build)], check=True)
env = dict(os.environ, CYCLONEDDS_HOME=str(full_dds))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(root / 'third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python')], env=env, check=True)
