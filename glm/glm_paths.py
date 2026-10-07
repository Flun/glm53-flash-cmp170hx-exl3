"""Filesystem locations and environment flags shared by the GLM serving modules.

The repo layout is::

    <repo root>/
        glm/        serving code (this package)
        kernels/    k_hcfuse (mHC fused decode kernels)
        dflash2/    DFlash2 BF16 -> EXL3 6bpw conversion scripts
        scripts/    setup / serve launchers
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = REPO_ROOT / "kernels"
DFLASH2_DIR = REPO_ROOT / "dflash2"

# k_hcfuse.install() only runs when this env var is set (upstream 0xSero hook).
HCFUSE_ENV = "GLM53_K_HCFUSE"
# Optional bit-exactness shadow check; see kernels/k_hcfuse.py.
HCFUSE_CHECK_ENV = "GLM53_K_HCFUSE_CHECK"
