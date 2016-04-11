from __future__ import print_function

from copy_files_common import copy_files


def main():
    copy_files(cuda_lib_dir='/Developer/NVIDIA/CUDA-7.5/lib',
               cuda_lib_fmt='lib{0}.7.5.dylib',
               nvvm_lib_dir='/Developer/NVIDIA/CUDA-7.5/nvvm/lib',
               nvvm_lib_fmt='lib{0}.3.0.0.dylib',
               libdevice_lib_dir='/Developer/NVIDIA/CUDA-7.5/nvvm/libdevice',
               libdevice_lib_fmt='libdevice.compute_{0}.bc')


if __name__ == '__main__':
    main()
