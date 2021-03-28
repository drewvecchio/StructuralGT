from setuptools import setup

descr = """StructuralGT: An automated python package for graph theory analysis of structural networks.
Designed for processing digital micrographs of complex network materials.
For example, analyzing SEM images of polymer network.
Copyright (C) 2021, The Regents of the University of Michigan.
"""

setup(
    name='StructuralGT',
    version='1.0.1a',
    packages=[''],
    url='https://github.com/drewvecchio/StructuralGT',
    license='GNU General Public License v3',
    author='drewvecchio',
    author_email='vecdrew@umich.edu',
    description='Automated graph theory analysis of digital structural networks images',
    long_description=descr,
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',
        'networkx',
        'opencv-python',
        'sknw',
        'Pillow',
        'pandas',
        'GraphRicciCurvature'
    ],
)
