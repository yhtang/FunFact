import io
import sys
import re
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

with open('ntf/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')


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
    name='ntf',
    version=__version__,
    python_requires='>=3.8',
    url='https://gitlab.com/yhtang/ntf',
    license='BSD',
    author='Yu-Hang Tang',
    tests_require=['tox'],
    install_requires=[
        'numpy>=1.17',
        'deap>=1.3',
        'torch>=1.7',
    ],
    extras_require={
        'docs': ['sphinx', 'sphinx-rtd-theme'],
    },
    cmdclass={'test': Tox},
    author_email='Tang.Maxin@gmail.com',
    description='TBD',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude='test'),
    package_data={'': ['README.md']},
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
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
