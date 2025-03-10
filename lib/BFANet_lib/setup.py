from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
include_dirs = ['/data1/weiguang/public_lib/cuda-11.8/']
setup(
    name='BFANet_lib',
    ext_modules=[
        CUDAExtension('BFANet_lib', [
            'src/bfanet_lib_api.cpp',
            'src/cuda.cu'
        ],  include_dirs=include_dirs, extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

