from setuptools import find_packages
from distutils.core import setup

setup(
    name='globe_walking',
    version='1.0.0',
    author='William Liang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='willjhliang@gmail.com',
    install_requires=[
        'params-proto==2.10.0',
        'ml-logger==0.8.117',
        # 'gym==0.18.0',
        'gym',
        'tqdm',
        'matplotlib',
        'numpy==1.23.5',
        'wandb==0.15.0',
        'wandb_osh',
        'moviepy',
        'imageio'
    ]
)
