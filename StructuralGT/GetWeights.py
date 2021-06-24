"""GetWeights: Obtains the 'weight' of fibers in a binary
image by measuring the pixel width of the perpendicular bisector.

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

import numpy as np

def unitvector(u,v):
    # Inputs:
    # u, v: two coordinates (x, y) or (x, y, z)

    vec = u-v # find the vector between u and v

    # returns the unit vector in the direction from v to u
    return vec/np.linalg.norm(vec)

def halflength(u,v):
    # Inputs:
    # u, v: two coordinates (x, y) or (x, y, z)

    vec = u-v # find the vector between u and v

    # returns half of the length of the vector
    return np.linalg.norm(vec)/2

def findorthogonal(u,v):
    # Inputs:
    # u, v: two coordinates (x, y) or (x, y, z)

    n = unitvector(u,v)         # make n a unit vector along u,v
    if (np.isnan(n[0]) or np.isnan(n[1])):
        n[0] , n[1] = float(0) , float(0)
    hl = halflength(u,v)        # find the half-length of the vector u,v
    orth = np.random.randn(2)   # take a random vector
    orth -= orth.dot(n) * n     # make it orthogonal to vector u,v
    orth /= np.linalg.norm(orth)# make it a unit vector

    # Returns the coordinates of the midpoint of vector u,v; the orthogonal unit vector
    return (v + n*hl), orth

def boundarycheck(coord, w, h):
    # Inputs:
    # coord: the coordinate (x,y) to check; no (x,y,z) compatibility yet
    # w,h: the width and height of the image to set the boundaries

    oob = 0     # Generate a boolean check for out-of-boundary
    # Check if coordinate is within the boundary
    if(coord[0]<0 or coord[1]<0 or coord[0]>(w-1) or coord[1]>(h-1)):
        oob = 1
        coord[0], coord[1] = 1, 1

    # returns the boolean oob (1 if boundary error); coordinates (reset to (1,1) if boundary error)
    return oob, coord

def lengthtoedge(m,orth,img_bin):
    # Inputs:
    # m: the midpoint of a trace of an edge
    # orth: an orthogonal unit vector
    # img_bin: the binary image that the graph is derived from

    w,h = img_bin.shape             # finds dimensions of img_bin for boundary check
    check = 0                       # initializing boolean check
    i = 0                           # initializing iterative variable
    while(check==0):                # iteratively check along orthogonal vector to see if the coordinate is either...
        ptcheck = m + i*orth        # ... out of bounds, or no longer within the fiber in img_bin
        ptcheck[0], ptcheck[1] = int(ptcheck[0]), int(ptcheck[1])
        oob, ptcheck = boundarycheck(ptcheck, w, h)
        if(img_bin[int(ptcheck[0])][int(ptcheck[1])] == 0 or oob == 1):
            edge = m + (i-1)*orth
            edge[0], edge[1] = int(edge[0]), int(edge[1])
            l1 = edge               # When the check indicates oob or black space, assign width to l1
            check = 1
        else:
            i += 1
    check = 0
    i = 0
    while(check == 0):              # Repeat, but following the negative orthogonal vector
        ptcheck = m - i*orth
        ptcheck[0], ptcheck[1] = int(ptcheck[0]), int(ptcheck[1])
        oob, ptcheck = boundarycheck(ptcheck, w, h)
        if(img_bin[int(ptcheck[0])][int(ptcheck[1])] == 0 or oob == 1):
            edge = m - (i-1)*orth
            edge[0], edge[1] = int(edge[0]), int(edge[1])
            l2 = edge              # When the check indicates oob or black space, assign width to l1
            check = 1
        else:
            i += 1

    # returns the length between l1 and l2, which is the width of the fiber associated with an edge, at its midpoint
    return np.linalg.norm(l1-l2)

def assignweightsbywidth(ge, img_bin):
    # Inputs:
    # ge: a list of pts that trace along a graph edge
    # img_bin: the binary image that the graph is derived from

    # check to see if ge is an empty or unity list, if so, set wt to 1
    if(len(ge)<2):
        pix_width = 10
        wt = 1
    # if ge exists, find the midpoint of the trace, and orthogonal unit vector
    else:
        endindex = len(ge) - 1
        midindex = int(len(ge)/2)
        pt1 = ge[0]
        pt2 = ge[endindex]
        m = ge[midindex]
        midpt, orth = findorthogonal(pt1, pt2)
        m[0] = int(m[0])
        m[1] = int(m[1])
        pix_width = int(lengthtoedge(m, orth, img_bin))
        wt = lengthtoedge(m, orth, img_bin)/10

    # returns the width in pixels; the weight which is the width normalized by 10
    return pix_width, wt