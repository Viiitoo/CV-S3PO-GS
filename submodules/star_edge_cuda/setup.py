from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="star_edge_cuda",
    packages=["star_edge_cuda"],
    ext_modules=[
        CUDAExtension(
            name="star_edge_cuda._C",
            sources=[
                "ext.cpp",
                "star_edge.cu",
                "knn_morton.cu",
                "direction_sampling.cu",
                "sphere_kde.cu",
                "sht_power.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "--std=c++17",
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
