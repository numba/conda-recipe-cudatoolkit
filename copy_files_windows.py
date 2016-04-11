from __future__ import print_function

from copy_files_common import copy_files


def main():
    copy_files(cuda_lib_dir=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin',
               cuda_lib_fmt='{0}64_75.dll',
               nvvm_lib_dir=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\nvvm\bin',
               nvvm_lib_fmt='{0}64_30_0.dll',
               libdevice_lib_dir=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\nvvm\libdevice',
               libdevice_lib_fmt='libdevice.compute_{0}.bc')


if __name__ == '__main__':
    main()
