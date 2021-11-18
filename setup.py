import io
import sys
import re
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

with open('funfact/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import tox  # import here, cause outside the eggs aren't loaded
        errcode = tox.cmdline(self.test_args)
        sys.exit(errcode)


setup(
    name='funfact',
    version=__version__,
    python_requires='>=3.7',
    url='https://github.com/yhtang/FunFact',
    license='BSD',
    author='Yu-Hang Tang',
    tests_require=['tox'],
    install_requires=[
        'numpy>=1.17',
        'deap>=1.3',
        'treelib>=1.6',
        'tqdm>=4.55',
        'dill>=0.3.3',
        'matplotlib>=3.4',
        'scipy>=1.7.1',
        'asciitree>=0.3.3',
        'jax[cpu]>=0.2.24',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
            'm2r2',
        ],
        'devel': [
            'expectexception>=0.1.1',
            'tox>=3.24.4',
            'pytest>=4.6.11',
            'pytest-cov>=3.0.0',
            'flake8>=4.0.1',
        ]
    },
    cmdclass={'test': Tox},
    author_email='Tang.Maxin@gmail.com',
    description='Functional factorization for matrices and tensors',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude='test'),
    package_data={'': ['README.md', 'funfact/cpp/*']},
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
