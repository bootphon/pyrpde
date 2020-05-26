# Copyright CoML 2020, Licensed under the EUPL

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split("\n")

setup(
    name='pyrpde',
    version='0.1.2',
    description='A Python implementation of the Recurrence Period Density Entropy (RPDE)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bootphon/pyrpde',
    author='Hadrien Titeux & Rachid Riad',
    author_email='hadrien.titeux@ens.fr',
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=requirements,
    setup_requires=['pytest-runner', 'setuptools>=38.6.0'],  # >38.6.0 needed for markdown README.md
    tests_require=['pytest'],
)
