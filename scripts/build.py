from __future__ import print_function
import fnmatch
import os
import sys
import shutil
import tarfile
import urllib.parse as urlparse
import yaml

from contextlib import contextmanager
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory as tempdir

from conda.exports import download, hashsum_file

config = {}
versions = ['7.5', '8.0', '9.0']
for v in versions:
    config[v] = {'linux': {}, 'windows': {}, 'osx': {}}


# The config dictionary looks like:
# config[cuda_version(s)...]
#
# and for each cuda_version the keys:
# base_url the base url for all downloads
# patch_url_ext the extra path needed to reach the patch directory from base_url
# installers_url_ext the extra path needed to reach the local installers directory
# md5_url the url for checksums
# cuda_libraries the libraries to copy in
# libdevice_versions the library device versions supported (.bc files)
# linux the linux platform config (see below)
# windows the windows platform config (see below)
# osx the osx platform config (see below)
#
# For each of the 3 platform specific dictionaries linux, windows, osx
# a dictionary containing keys:
# blob the name of the downloaded file, for linux this is the .run file
# patches a list of the patch files for the blob, they are applied in order
# cuda_lib_fmt string format for the cuda libraries
# nvvm_lib_fmt string format for the nvvm libraries
# libdevice_lib_fmt string format for the libdevice.compute bitcode file


######################
### CUDA 7.5 setup ###
######################

cu_75 = config['7.5']
cu_75['base_url'] = "http://developer.download.nvidia.com/compute/cuda/7.5/Prod/"
cu_75['patch_url_ext'] = ''
cu_75['installers_url_ext'] = 'local_installers/'
cu_75['md5_url'] = "http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/md5sum.txt"
cu_75['cuda_libraries'] = [
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
cu_75['libdevice_versions'] = ['20.10', '30.10', '35.10', '50.10']

cu_75['linux'] = {'blob': 'cuda_7.5.18_linux.run',
                  'patches': [],
                  'cuda_lib_fmt': 'lib{0}.so.7.5',
                  'nvvm_lib_fmt': 'lib{0}.so.3.0.0',
                  'libdevice_lib_fmt': 'libdevice.compute_{0}.bc'
                  }

cu_75['windows'] = {'blob': 'cuda_7.5.18_win10.exe',
                    'patches': [],
                    'cuda_lib_fmt': '{0}64_75.dll',
                    'nvvm_lib_fmt': '{0}64_30_0.dll',
                    'libdevice_lib_fmt': 'libdevice.compute_{0}.bc'
                    }

cu_75['osx'] = {'blob': 'cuda_7.5.27_mac.dmg',
                'patches': [],
                'cuda_lib_fmt': 'lib{0}.7.5.dylib',
                'nvvm_lib_fmt': 'lib{0}.3.0.0.dylib',
                'libdevice_lib_fmt': 'libdevice.compute_{0}.bc'
                }

######################
### CUDA 8.0 setup ###
######################

cu_8 = config['8.0']
cu_8['base_url'] = "https://developer.nvidia.com/compute/cuda/8.0/Prod2/"
cu_8['installers_url_ext'] = 'local_installers/'
cu_8['patch_url_ext'] = 'patches/2/'
cu_8['md5_url'] = "https://developer.nvidia.com/compute/cuda/8.0/Prod2/docs/sidebar/md5sum-txt"
cu_8['cuda_libraries'] = [
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
cu_8['libdevice_versions'] = ['20.10', '30.10', '35.10', '50.10']

cu_8['linux'] = {'blob': 'cuda_8.0.61_375.26_linux-run',
                 'patches': ['cuda_8.0.61.2_linux-run'],
                 'cuda_lib_fmt': 'lib{0}.so.8.0',
                 'nvvm_lib_fmt': 'lib{0}.so.3.1.0',
                 'libdevice_lib_fmt': 'libdevice.compute_{0}.bc'
                 }

cu_8['windows'] = {'blob': 'cuda_8.0.61_windows-exe',
                   'patches': ['cuda_8.0.61.2_windows-exe'],
                   'cuda_lib_fmt': '{0}64_80.dll',
                   'nvvm_lib_fmt': '{0}64_31_0.dll',
                   'libdevice_lib_fmt': 'libdevice.compute_{0}.bc'
                   }

cu_8['osx'] = {'blob': 'cuda_8.0.61_mac-dmg',
               'patches': ['cuda_8.0.61.2_mac-dmg'],
               'cuda_lib_fmt': 'lib{0}.8.0.dylib',
               'nvvm_lib_fmt': 'lib{0}.3.1.0.dylib',
               'libdevice_lib_fmt': 'libdevice.compute_{0}.bc'
               }


# CUDA 9.0 setup
# TODO


class Extractor(object):
    """Extractor base class, platform specific extractors should inherit
    from this class.
    """

    libdir = {'linux': 'lib',
              'osx': 'lib',
              'windows': 'DLLs'}

    def __init__(self, version, ver_config, plt_config):
        """Initialise an instance:
        Arguments:
          version - CUDA version string
          ver_config - the configuration for this CUDA version
          plt_config - the configuration for this platform
        """
        self.cu_version = version
        self.md5_url = ver_config['md5_url']
        self.base_url = ver_config['base_url']
        self.patch_url_ext = ver_config['patch_url_ext']
        self.installers_url_ext = ver_config['installers_url_ext']
        self.cuda_libraries = ver_config['cuda_libraries']
        self.libdevice_versions = ver_config['libdevice_versions']
        self.cu_blob = plt_config['blob']
        self.cuda_lib_fmt = plt_config['cuda_lib_fmt']
        self.nvvm_lib_fmt = plt_config['nvvm_lib_fmt']
        self.libdevice_lib_fmt = plt_config['libdevice_lib_fmt']
        self.patches = plt_config['patches']
        self.config = {'version': version, **ver_config}
        self.prefix = os.environ['PREFIX']
        self.src_dir = os.environ['SRC_DIR']
        self.output_dir = os.path.join(self.prefix, self.libdir[getplatform()])
        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            pass

    def download_blobs(self):
        """Downloads the binary blobs to the $SRC_DIR
        """
        dl_url = urlparse.urljoin(self.base_url, self.installers_url_ext)
        dl_url = urlparse.urljoin(dl_url, self.cu_blob)
        dl_path = os.path.join(self.src_dir, self.cu_blob)
        print("downloading %s to %s" % (dl_url, dl_path))
        download(dl_url, dl_path)
        for p in self.patches:
            dl_url = urlparse.urljoin(self.base_url, self.patch_url_ext)
            dl_url = urlparse.urljoin(dl_url, p)
            dl_path = os.path.join(self.src_dir, p)
            print("downloading %s to %s" % (dl_url, dl_path))
            download(dl_url, dl_path)

    def check_md5(self):
        """Checks the md5sums of the downloaded binaries
        """
        md5file = self.md5_url.split('/')[-1]
        path = os.path.join(self.src_dir, md5file)
        download(self.md5_url, path)

        # compute hash of blob
        blob_path = os.path.join(self.src_dir, self.cu_blob)
        md5sum = hashsum_file(blob_path, 'md5')

        # get checksums
        with open(md5file, 'r') as f:
            checksums = [x.strip().split() for x in f.read().splitlines() if x]

        # check md5 and filename match up
        check_dict = {x[0]: x[1] for x in checksums}
        assert check_dict[md5sum].startswith(self.cu_blob[:-7])

    def copy(self, *args):
        """The method to copy extracted files into the conda package platform
        specific directory. Platform specific extractors must implement.
        """
        raise RuntimeError('Must implement')

    def extract(self, *args):
        """The method to extract files from the cuda binary blobs.
        Platform specific extractors must implement.
        """
        raise RuntimeError('Must implement')

    def get_paths(self, libraries, dirpath, template):
        """Gets the paths to the various cuda libraries and bc files
        """
        pathlist = []
        for libname in libraries:
            filename = template.format(libname)
            paths = fnmatch.filter(os.listdir(dirpath), filename)
            for path in paths:
                tmppath = os.path.join(dirpath, path)
                assert os.path.isfile(tmppath), 'missing {0}'.format(tmppath)
                pathlist.append(tmppath)
        return pathlist

    def copy_files(self, cuda_lib_dir, nvvm_lib_dir, libdevice_lib_dir):
        """Copies the various cuda libraries and bc files to the output_dir
        """
        filepaths = []
        filepaths += self.get_paths(self.cuda_libraries,
                                    cuda_lib_dir, self.cuda_lib_fmt)
        filepaths += self.get_paths(['nvvm'], nvvm_lib_dir, self.nvvm_lib_fmt)
        filepaths += self.get_paths(self.libdevice_versions, libdevice_lib_dir,
                                    self.libdevice_lib_fmt)

        for fn in filepaths:
            print('copying %s to %s' % (fn, self.output_dir))
            shutil.copy(fn, self.output_dir)

    def dump_config(self):
        """Dumps the config dictionary into the output directory
        """
        dumpfile = os.path.join(self.output_dir, 'cudatoolkit_config.yaml')
        with open(dumpfile, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


class WindowsExtractor(Extractor):
    """The windows extractor
    """

    def copy(self, *args):
        basepath, store = args
        self.copy_files(
            cuda_lib_dir=store,
            nvvm_lib_dir=store,
            libdevice_lib_dir=store)

    def extract(self):
        runfile = self.cu_blob
        patches = self.patches
        try:
            with tempdir() as tmpd:
                check_call(['7za', 'x', '-o%s' %
                            tmpd, os.path.join(self.src_dir, runfile)])
                for p in patches:
                    check_call(['7za', 'x', '-aoa', '-o%s' %
                                tmpd, os.path.join(self.src_dir, p)])
                # fetch all the dlls into DLLs
                store_name = 'DLLs'
                store = os.path.join(tmpd, store_name)
                os.mkdir(store)
                for path, dirs, files in os.walk(tmpd):
                    if 'jre' not in path:  # don't get jre dlls
                        for filename in fnmatch.filter(files, "*.dll"):
                            if not Path(os.path.join(
                                    store, filename)).is_file():
                                shutil.copy(
                                    os.path.join(path, filename),
                                    store)
                        for filename in fnmatch.filter(files, "*.bc"):
                            if not Path(os.path.join(
                                    store, filename)).is_file():
                                shutil.copy(
                                    os.path.join(path, filename),
                                    store)
                self.copy(tmpd, store)
        except PermissionError:
            # TODO: fix this
            # cuda 8 has files that refuse to delete, figure out perm changes
            # needed and apply them above, tempdir context exit fails to rmtree
            pass


class LinuxExtractor(Extractor):
    """The linux extractor
    """

    def copy(self, *args):
        basepath = args[0]
        self.copy_files(
            cuda_lib_dir=os.path.join(
                basepath, 'lib64'), nvvm_lib_dir=os.path.join(
                basepath, 'nvvm', 'lib64'), libdevice_lib_dir=os.path.join(
                basepath, 'nvvm', 'libdevice'))

    def extract(self):
        runfile = self.cu_blob
        patches = self.patches
        os.chmod(runfile, 0o777)
        with tempdir() as tmpd:
            check_call([os.path.join(self.src_dir, runfile),
                        '--toolkitpath', tmpd, '--toolkit', '--silent'])
            for p in patches:
                os.chmod(p, 0o777)
                check_call([os.path.join(self.src_dir, p),
                            '--installdir', tmpd, '--accept-eula', '--silent'])
            self.copy(tmpd)


@contextmanager
def _hdiutil_mount(mntpnt, image):
    """Context manager to mount osx dmg images and ensure they are
    unmounted on exit.
    """
    check_call(['hdiutil', 'attach', '-mountpoint', mntpnt, image])
    yield mntpnt
    check_call(['hdiutil', 'detach', mntpnt])


class OsxExtractor(Extractor):
    """The osx extractor
    """

    def copy(self, *args):
        basepath, store = args
        self.copy_files(cuda_lib_dir=store,
                        nvvm_lib_dir=store,
                        libdevice_lib_dir=store)

    def _extract_matcher(self, tarmembers):
        """matcher helper for tarfile.extractall()
        """
        for tarinfo in tarmembers:
            ext = os.path.splitext(tarinfo.name)[-1]
            if ext == '.dylib' or ext == '.bc':
                yield tarinfo

    def _mount_extract(self, image, store):
        """Mounts and extracts the files from an image into store
        """
        with tempdir() as tmpmnt:
            with _hdiutil_mount(tmpmnt, os.path.join(os.getcwd(), image)) as mntpnt:
                for tlpath, tldirs, tlfiles in os.walk(mntpnt):
                    for tzfile in fnmatch.filter(tlfiles, "*.tar.gz"):
                        with tarfile.open(os.path.join(tlpath, tzfile)) as tar:
                            tar.extractall(
                                store, members=self._extract_matcher(tar))

    def extract(self):
        runfile = self.cu_blob
        patches = self.patches
        with tempdir() as tmpd:
            # fetch all the dylibs into lib64, but first get them out of the
            # image and tar.gzs into tmpstore
            extract_store_name = 'tmpstore'
            extract_store = os.path.join(tmpd, extract_store_name)
            os.mkdir(extract_store)
            store_name = 'lib64'
            store = os.path.join(tmpd, store_name)
            os.mkdir(store)
            self._mount_extract(runfile, extract_store)
            for p in self.patches:
                self._mount_extract(p, extract_store)
            for path, dirs, files in os.walk(extract_store):
                for filename in fnmatch.filter(files, "*.dylib"):
                    if not Path(os.path.join(store, filename)).is_file():
                        shutil.copy(os.path.join(path, filename), store)
                for filename in fnmatch.filter(files, "*.bc"):
                    if not Path(os.path.join(store, filename)).is_file():
                        shutil.copy(os.path.join(path, filename), store)
            self.copy(tmpd, store)


def getplatform():
    plt = sys.platform
    if plt.startswith('linux'):
        return 'linux'
    elif plt.startswith('win'):
        return 'windows'
    elif plt.startswith('darwin'):
        return 'osx'
    else:
        raise RuntimeError('Unknown platform')

dispatcher = {'linux': LinuxExtractor,
              'windows': WindowsExtractor,
              'osx': OsxExtractor}


def _main():
    print("Running build")

    # package version decl must match cuda release version
    cu_version = os.environ['PKG_VERSION']

    # get an extractor
    plat = getplatform()
    extractor_impl = dispatcher[plat]
    version_cfg = config[cu_version]
    extractor = extractor_impl(cu_version, version_cfg, version_cfg[plat])

    # download binaries
    extractor.download_blobs()

    # check md5sum
    extractor.check_md5()

    # extract
    extractor.extract()

    # dump config
    extractor.dump_config()

if __name__ == "__main__":
    _main()
