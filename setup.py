from setuptools import setup, find_packages

descr = """StructuralGT: An automated python package for graph theory analysis of structural networks.\n
Designed for processing digital micrographs of complex network materials.\n
For example, analyzing SEM images of polymer network.\n
See the README for detail information.

Copyright (C) 2021, The Regents of the University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contributers: Drew Vecchio, Samuel Mahler, Mark D. Hammig, Nicholas A. Kotov\n
Contact email: vecdrew@umich.edu

![](Images/SGT_BC_ex.png?raw=true)
"""

setup(
    name='StructuralGT',
    version='1.0.1a0',
    packages=find_packages(),
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
