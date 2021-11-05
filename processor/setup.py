from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        r'processor',
        [r'processor.pyx']
    ),
]

setup(
    name='processor',
    ext_modules=cythonize(ext_modules),
)