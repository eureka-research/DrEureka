from setuptools import find_packages
from distutils.core import setup

setup(
    name='dr_eureka',
    version='1.0.0',
    author='Will Liang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='willjhliang@gmail.com',
    install_requires=[
        "openai==0.28.0",
        "gpustat",
        "wandb",
        "numpy<1.24"
        "hydra-core",
    ]
)
