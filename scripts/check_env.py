"""Read-only environment diagnostics; installation is left to the user."""

import importlib.util
import logging
import platform
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Python: %s (%s)", platform.python_version(), sys.executable)
    logger.info("nvcc: %s", shutil.which("nvcc") or "absent (not required for this project)")
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        logger.info("NVIDIA: %s", (result.stdout or result.stderr).strip())
    else:
        logger.info("nvidia-smi: absent; NVIDIA driver utilities are not installed")
    if importlib.util.find_spec("torch") is None:
        logger.info("PyTorch: not installed in this Python environment")
        return 1
    import torch

    logger.info("PyTorch: %s; bundled CUDA: %s", torch.__version__, torch.version.cuda)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            logger.info("GPU %d: %s; VRAM=%.1f GiB", index, props.name, props.total_memory / 2**30)
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        (x @ x.T).square().mean().backward()
        torch.cuda.synchronize()
        logger.info("CUDA matrix multiplication + backward: OK")
    else:
        logger.info(
            "CPU training is available. For GPU: install the driver and CUDA PyTorch, then reboot."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
