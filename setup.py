from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension("_motif_sampler", ["_motif_sampler.pyx"],
              include_dirs = [numpy.get_include()],
              extra_compile_args=['-O3', '-march=native', '-ffast-math', '-funroll-loops'],
          )
]

setup(
    name = "motif sampler",
    ext_modules = cythonize(extensions),
)
