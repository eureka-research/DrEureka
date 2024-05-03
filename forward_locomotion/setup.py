from setuptools import find_packages
from distutils.core import setup

setup(
    name='forward_locomotion',
    version='1.0.0',
    author='William Liang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='willjhliang@gmail.com',
    install_requires=[
        'matplotlib',
        'gym',
        'ml-logger==0.8.117',
    ]
)
