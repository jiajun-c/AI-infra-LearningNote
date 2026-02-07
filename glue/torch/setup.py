from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_ops',  # 安装后的包名
    ext_modules=[
        CUDAExtension(
            name='my_cuda_ops', # import 时的模块名
            sources=['vector_add.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)