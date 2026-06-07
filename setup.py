from __future__ import annotations

import os
import sys

from setuptools import Extension, find_packages, setup

BUILD_CYTHON = os.environ.get("FASTISO_BUILD_CYTHON") == "1"
BUILD_OPENMP = BUILD_CYTHON and os.environ.get("FASTISO_BUILD_OPENMP", "1") != "0"

if BUILD_CYTHON:
    try:
        import numpy as np
        from Cython.Build import cythonize
    except ImportError as exc:
        raise RuntimeError(
            "FASTISO_BUILD_CYTHON=1 requires installed numpy and Cython. "
            "Install dev dependencies or build with --no-build-isolation."
        ) from exc


extensions = []
if BUILD_CYTHON:
    if BUILD_OPENMP:
        if sys.platform == "win32":
            openmp_compile_args = ["/openmp"]
            openmp_link_args = []
        else:
            openmp_compile_args = ["-fopenmp"]
            openmp_link_args = ["-fopenmp"]
    else:
        openmp_compile_args = []
        openmp_link_args = []
    extensions = cythonize(
        [
            Extension(
                "fastiso._cython_log_table",
                ["src/fastiso/_cython_log_table.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
        ],
        compiler_directives={"language_level": "3"},
    )


setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=extensions,
)
