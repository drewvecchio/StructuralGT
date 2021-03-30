"""skel_ID: A collection of methods and tools for analyzing
and altering a skeletal image.  Prepares the skeleton for
conversion into a graph object.

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

Contributers: Drew Vecchio, Samuel Mahler, Mark D. Hammig, Nicholas A. Kotov
Contact email: vecdrew@umich.edu
"""

from __main__ import *

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.morphology import disk, remove_small_objects
from skimage.morphology import binary_dilation as dilate




def branchedPoints(skel):

    # defining branch shapes to locate nodes
    # overexplained this section a bit
    xbranch0 = np.array([[1,0,1],
                         [0,1,0],
                         [1,0,1]])

    xbranch1 = np.array([[0,1,0],
                         [1,1,1],
                         [0,1,0]])

    tbranch0 = np.array([[0,0,0],
                         [1,1,1],
                         [0,1,0]])

    # flipud is flipping them up-down
    # tbranch2 is tbranch0 transposed, which permutes it in all directions (might not be using that word right)
    # tbranch3 is tbranch2 flipped left right
    # those 3 functions are used to create all possible branches with just a few starting arrays below

    tbranch1 = np.flipud(tbranch0)
    tbranch2 = tbranch0.T
    tbranch3 = np.fliplr(tbranch2)

    tbranch4 = np.array([[1,0,1],
                         [0,1,0],
                         [1,0,0]])
    tbranch5 = np.flipud(tbranch4)
    tbranch6 = np.fliplr(tbranch4)
    tbranch7 = np.fliplr(tbranch5)

    ybranch0 = np.array([[1,0,1],
                         [0,1,0],
                         [0,1,0]])

    ybranch1 = np.flipud(ybranch0)
    ybranch2 = ybranch0.T
    ybranch3 = np.fliplr(ybranch2)

    ybranch4 = np.array([[0,1,0],
                         [1,1,0],
                         [0,0,1]])

    ybranch5 = np.flipud(ybranch4)
    ybranch6 = np.fliplr(ybranch4)
    ybranch7 = np.fliplr(ybranch5)

    offbranch0 = np.array([[0,1,0],
                           [1,1,0],
                           [1,0,1]])

    offbranch1 = np.flipud(offbranch0)
    offbranch2 = np.fliplr(offbranch0)
    offbranch3 = np.fliplr(offbranch1)
    offbranch4 = offbranch0.T
    offbranch5 = np.flipud(offbranch4)
    offbranch6 = np.fliplr(offbranch4)
    offbranch7 = np.fliplr(offbranch5)

    clustbranch0 = np.array([[0,1,1],
                             [0,1,1],
                             [1,0,0]])

    clustbranch1 = np.flipud(clustbranch0)
    clustbranch2 = np.fliplr(clustbranch0)
    clustbranch3 = np.fliplr(clustbranch1)

    clustbranch4 = np.array([[1,1,1],
                             [0,1,1],
                             [1,0,0]])

    clustbranch5 = np.flipud(clustbranch4)
    clustbranch6 = np.fliplr(clustbranch4)
    clustbranch7 = np.fliplr(clustbranch5)

    clustbranch8 = np.array([[1,1,1],
                             [0,1,1],
                             [1,0,1]])

    clustbranch9 = np.flipud(clustbranch8)
    clustbranch10 = np.fliplr(clustbranch8)
    clustbranch11 = np.fliplr(clustbranch9)

    crossbranch0 = np.array([[1,0,0],
                             [1,1,1],
                             [0,1,0]])

    crossbranch1 = np.flipud(crossbranch0)
    crossbranch2 = np.fliplr(crossbranch0)
    crossbranch3 = np.fliplr(crossbranch1)
    crossbranch4 = crossbranch0.T
    crossbranch5 = np.flipud(crossbranch4)
    crossbranch6 = np.fliplr(crossbranch4)
    crossbranch7 = np.fliplr(crossbranch5)


    # finding the location of all the branch points based on the arrays above
    br1 = ndimage.binary_hit_or_miss(skel, xbranch0)
    br2 = ndimage.binary_hit_or_miss(skel, xbranch1)
    br3 = ndimage.binary_hit_or_miss(skel, tbranch0)
    br4 = ndimage.binary_hit_or_miss(skel, tbranch1)
    br5 = ndimage.binary_hit_or_miss(skel, tbranch2)
    br6 = ndimage.binary_hit_or_miss(skel, tbranch3)
    br7 = ndimage.binary_hit_or_miss(skel, tbranch4)
    br8 = ndimage.binary_hit_or_miss(skel, tbranch5)
    br9 = ndimage.binary_hit_or_miss(skel, tbranch6)
    br10 = ndimage.binary_hit_or_miss(skel, tbranch7)
    br11 = ndimage.binary_hit_or_miss(skel, ybranch0)
    br12 = ndimage.binary_hit_or_miss(skel, ybranch1)
    br13 = ndimage.binary_hit_or_miss(skel, ybranch2)
    br14 = ndimage.binary_hit_or_miss(skel, ybranch3)
    br15 = ndimage.binary_hit_or_miss(skel, ybranch4)
    br16 = ndimage.binary_hit_or_miss(skel, ybranch5)
    br17 = ndimage.binary_hit_or_miss(skel, ybranch6)
    br18 = ndimage.binary_hit_or_miss(skel, ybranch7)
    br19 = ndimage.binary_hit_or_miss(skel, offbranch0)
    br20 = ndimage.binary_hit_or_miss(skel, offbranch1)
    br21 = ndimage.binary_hit_or_miss(skel, offbranch2)
    br22 = ndimage.binary_hit_or_miss(skel, offbranch3)
    br23 = ndimage.binary_hit_or_miss(skel, offbranch4)
    br24 = ndimage.binary_hit_or_miss(skel, offbranch5)
    br25 = ndimage.binary_hit_or_miss(skel, offbranch6)
    br26 = ndimage.binary_hit_or_miss(skel, offbranch7)
    br27 = ndimage.binary_hit_or_miss(skel, clustbranch0)
    br28 = ndimage.binary_hit_or_miss(skel, clustbranch1)
    br29 = ndimage.binary_hit_or_miss(skel, clustbranch2)
    br30 = ndimage.binary_hit_or_miss(skel, clustbranch3)
    br31 = ndimage.binary_hit_or_miss(skel, clustbranch4)
    br32 = ndimage.binary_hit_or_miss(skel, clustbranch5)
    br33 = ndimage.binary_hit_or_miss(skel, clustbranch6)
    br34 = ndimage.binary_hit_or_miss(skel, clustbranch7)
    br35 = ndimage.binary_hit_or_miss(skel, clustbranch8)
    br36 = ndimage.binary_hit_or_miss(skel, clustbranch9)
    br37 = ndimage.binary_hit_or_miss(skel, clustbranch10)
    br38 = ndimage.binary_hit_or_miss(skel, clustbranch11)
    br39 = ndimage.binary_hit_or_miss(skel, crossbranch0)
    br40 = ndimage.binary_hit_or_miss(skel, crossbranch1)
    br41 = ndimage.binary_hit_or_miss(skel, crossbranch2)
    br42 = ndimage.binary_hit_or_miss(skel, crossbranch3)
    br43 = ndimage.binary_hit_or_miss(skel, crossbranch4)
    br44 = ndimage.binary_hit_or_miss(skel, crossbranch5)
    br45 = ndimage.binary_hit_or_miss(skel, crossbranch6)
    br46 = ndimage.binary_hit_or_miss(skel, crossbranch7)

    br = br1+br2+br3+br4+br5+br6+br7+br8+br9+br10+br11+br12+br13+br14+br15+br16+br17+br18+br19+br20+br21+br22+br23+br24\
         +br25+br26+br27+br28+br29+br30+br31+br32+br33+br34+br35+br36+br37+br38+br39+br40+br41+br42+br43+br44+br45+br46
    return br


def endPoints(skel):

    # defining different types of endpoints
    endpoint1 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

    endpoint2 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

    endpoint3 = np.array([[0, 0, 0],
                          [0, 1, 1],
                          [0, 0, 0]])

    endpoint4 = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [0, 0, 0]])

    endpoint5 = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 0, 0]])

    endpoint6 = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])

    endpoint7 = np.array([[0, 0, 0],
                          [1, 1, 0],
                          [0, 0, 0]])

    endpoint8 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0]])

    endpoint9 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])

    # finding all the locations of the endpoints
    ep1 = ndimage.binary_hit_or_miss(skel, endpoint1)
    ep2 = ndimage.binary_hit_or_miss(skel, endpoint2)
    ep3 = ndimage.binary_hit_or_miss(skel, endpoint3)
    ep4 = ndimage.binary_hit_or_miss(skel, endpoint4)
    ep5 = ndimage.binary_hit_or_miss(skel, endpoint5)
    ep6 = ndimage.binary_hit_or_miss(skel, endpoint6)
    ep7 = ndimage.binary_hit_or_miss(skel, endpoint7)
    ep8 = ndimage.binary_hit_or_miss(skel, endpoint8)
    ep9 = ndimage.binary_hit_or_miss(skel, endpoint9)
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8 + ep9
    return ep

def pruning(skeleton, size, Bps):
    branchpoints = Bps
    #remove iteratively end points "size" times from the skeleton
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        points = np.logical_and(endpoints, branchpoints)
        endpoints = np.logical_xor(endpoints, points)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton


def merge_nodes(skeleton):

    # overlay a disk over each branch point and find the overlaps to combine nodes
    skeleton_integer = 1 * skeleton
    radius = 2
    mask_elem = disk(radius)
    BpSk = branchedPoints(skeleton_integer)
    BpSk = 1*(dilate(BpSk, mask_elem))

    # widenodes is initially an empty image the same size as the skeleton image
    sh = skeleton_integer.shape
    widenodes = np.zeros(sh, dtype='int')

    # this overlays the two skeletons
    # skeleton_integer is the full map, BpSk is just the branch points blown up to a larger size
    for x in range(sh[0]):
        for y in range(sh[1]):
            if skeleton_integer[x, y] == 0 and BpSk[x, y] == 0:
                widenodes[x, y] = 0
            else:
                widenodes[x, y] = 1

    # reskeletonzing widenodes and returning it, nearby nodes in radius 2 of each other should have been merged
    newskel = skeletonize(widenodes)
    return newskel

def make_skel(img_bin, merge, prune, clean, r_size):

    # rebuilding the binary image as a boolean for skeletonizing
    img_bin = (img_bin*(1/255)).astype(np.bool)

    # making the initial skeleton image, then getting x and y coords of all branch points and endpoints
    skeleton = skeletonize(img_bin)
    skel_int = 1*skeleton
    Bp = branchedPoints(skel_int)
    Ep = endPoints(skel_int)

    Bp_coord_y, Bp_coord_x = np.where(Bp == 1)
    Ep_coord_y, Ep_coord_x = np.where(Ep == 1)

    # calling the three functions for merging nodes, pruning edges, and removing disconnected segments
    if(merge == 1):
        skeleton = merge_nodes(skeleton)

    if(clean == 1):
        skeleton = remove_small_objects(skeleton, r_size, connectivity=2)

    skel_int = 1*skeleton

    Bps = branchedPoints(skel_int)

    if(prune == 1):
        skeleton = pruning(skeleton, 500, Bps)


    clean_skel = skeleton

    return clean_skel, skel_int, Bp_coord_x, Bp_coord_y, Ep_coord_x, Ep_coord_y
