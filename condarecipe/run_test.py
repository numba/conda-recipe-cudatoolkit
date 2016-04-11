import sys
from numba.cuda.cudadrv.libs import test
from numba.cuda.cudadrv.nvvm import NVVM


def run_test():
    if not test():
        return False
    nvvm = NVVM()
    print("NVVM version", nvvm.get_version())
    return True

sys.exit(0 if run_test() else 1)
