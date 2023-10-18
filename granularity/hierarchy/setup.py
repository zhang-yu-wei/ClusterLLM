# python3 setup.py build_ext --inplace
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('_hierarchy', ['_hierarchy.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))