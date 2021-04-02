from setuptools import setup, find_packages

descr = """StructuralGT: An automated python package for graph theory analysis of structural networks.\n
Designed for processing digital micrographs of complex network materials.\n
For example, analyzing SEM images of polymer network.\n

StructuralGT is designed as an easy-to-use python-based application for applying graph theory (GT) analysis to 
structural networks of a wide variety of material systems. This application converts digital images of 
nano-/micro-/macro-scale structures into a graph theoretical (GT) representation of the structure in the image 
consisting of nodes and the edges that connect them. Fibers (or fiber-like structures) are taken to represent edges, 
and the location where a fiber branches, or 2 or more fibers intersect are taken to represent nodes. The program 
operates with a graphical user interface (GUI) so that selecting images and processing the graphs are intuitive and 
accessible to anyone, regardless of programming experience.  Detection of networks from input images, the extraction of 
the graph object, and the subsequent GT analysis of the graph is handled entirely from the GUI, and a PDF file with the
results of the analysis is saved.

\nSee the README for detail information.\n
https://github.com/drewvecchio/StructuralGT'

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

"""

setup(
    name='StructuralGT',
    version='1.0.1a1',
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
