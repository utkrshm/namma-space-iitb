# Source before running the reconstruction CLI or CUDA tests.
# gsplat compiles its CUDA kernels on first use; it needs a CUDA toolkit matching torch
# (12.x) and a writable, project-specific extensions folder.
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  else
    printf '%s\n' 'Set CUDA_HOME to your CUDA 12.x development toolkit.' >&2
    return 1
  fi
fi
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_namma}
