# GPU environment for spacegen on this machine: source before `uv run spacegen ...`.
# gsplat compiles its CUDA kernels on first use; it needs a CUDA toolkit matching torch
# (12.x) and a writable, project-specific extensions folder.
export CUDA_HOME=${CUDA_HOME:-/home/utkarsh/namma-space-iitb-vyas/outputs/runtime/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_spacegen}
