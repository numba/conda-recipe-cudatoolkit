from __future__ import print_function

import os
import shutil


cuda_libraries = [
    'cublas',
    'cudart',
    'cufft',
    'curand',
    'cusparse',
    'cusolver',
    'nppc',
    'nppi',
    'npps',
    'nvrtc-builtins',
    'nvrtc',
]

extras_libraries = [
    'cupti',
]

libdevice_versions = ['20.10', '30.10', '35.10', '50.10']


def get_paths(libraries, dirpath, template):
    pathlist = []
    for libname in libraries:
        filename = template.format(libname)
        path = os.path.join(dirpath, filename)
        assert os.path.isfile(path), 'missing {0}'.format(path)
        pathlist.append(path)
    return pathlist


def copy_files(cuda_lib_dir, cuda_lib_fmt, nvvm_lib_dir, nvvm_lib_fmt,
               libdevice_lib_dir, libdevice_lib_fmt, extras_lib_dir, extras_lib_fmt):
    filepaths = []
    filepaths += get_paths(cuda_libraries, cuda_lib_dir, cuda_lib_fmt)
    filepaths += get_paths(['nvvm'], nvvm_lib_dir, nvvm_lib_fmt)
    filepaths += get_paths(libdevice_versions, libdevice_lib_dir,
                           libdevice_lib_fmt)
    filepaths += get_paths(extras_libraries, extras_lib_dir, extras_lib_fmt)

    for fn in filepaths:
        print('copying', fn)
        shutil.copy(fn, 'src')
