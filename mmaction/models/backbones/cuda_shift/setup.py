import os
import subprocess
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import CUDAExtension


def _find_cuda_home():
    # set directory path to CUDA which supports `nvcc` command as CUDA_HOME
    # other than 3 Guess in PyTorch
    nvcc = subprocess.check_output(['which', 'nvcc']).decode().rstrip('\r\n')
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    print(f'find cuda home:{cuda_home}')
    return cuda_home


# overwrite PyTorch auto-detected CUDA_HOME which may not be our expected,
# because PyTorch determines the CUDA_HOME using this priority:
# os.environ['CUDA_HOME'] > path to CUDA supporting `nvcc` > '/usr/local/cuda',
# although The first two guess should be the same in most cases.
torch.utils.cpp_extension.CUDA_HOME = _find_cuda_home()
CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME

sources = []
headers = []
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/shift_cuda.cpp']
    sources += ['src/cuda/shift_kernel_cuda.cu']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
else:
    raise ValueError('CUDA is not available')

extra_compile_args = dict(cxx=[])
extra_compile_args['nvcc'] = []

this_file = os.path.dirname(os.path.realpath(__file__))
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ext_module = [
    CUDAExtension(
        'cudashift',
        sources=sources,
        include_dirs=['src/'],
        define_macros=defines,
        extra_compile_args=extra_compile_args)
]

setup(
    name='CUDASHIFT',
    version='0.2',
    author='Hao Shao',
    ext_modules=ext_module,
    packages=find_packages(exclude=(
        'configs',
        'tests',
    )),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
