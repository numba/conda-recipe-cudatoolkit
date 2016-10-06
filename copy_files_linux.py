from __future__ import print_function

from copy_files_common import copy_files


def main():
    copy_files(cuda_lib_dir='/home/sklam/cuda-8.0/lib64',
               cuda_lib_fmt='lib{0}.so.8.0',
               nvvm_lib_dir='/home/sklam/cuda-8.0/nvvm/lib64',
               nvvm_lib_fmt='lib{0}.so.3.1.0',
               libdevice_lib_dir='/home/sklam/cuda-8.0/nvvm/libdevice',
               libdevice_lib_fmt='libdevice.compute_{0}.bc')


if __name__ == '__main__':
    main()
