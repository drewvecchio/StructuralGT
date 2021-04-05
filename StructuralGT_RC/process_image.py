"""process_image: A collection of methods and tools for image
processing, computer vision, and thresholding.  Aims to return
a binary image of a network from a input grayscale image.

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
import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import autolevel, median

def adjust_gamma(image, gamma):
    if(gamma != 1.00):
        invGamma = 1.00/gamma
        table = np.array([((i/255.0) ** invGamma) * 255 \
                          for i in np.arange(0,256)]).astype('uint8')
        return cv2.LUT(image,table)
    else:
        return image

def Hamming_window(image, windowsize):
    w, h = image.shape
    ham1x = np.hamming(w)[:, None]  # 1D hamming
    ham1y = np.hamming(h)[:, None]  # 1D hamming
    ham2d = np.sqrt(np.dot(ham1x, ham1y.T)) ** windowsize  # expand to 2D hamming
    f = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
    filtered_img = np.abs(inv_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img * 255 / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

def thresh_it(image, Threshtype, fg_color, asize, thresh):

    # only needed for OTSU threshold
    ret = 0

    # applying universal threshold, checking if it should be inverted (dark foreground)
    if(Threshtype ==0):
        if(fg_color ==1):
            img_bin = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            img_bin = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

    # adaptive threshold generation
    elif(Threshtype ==1 ):
        if (fg_color == 1):
            img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, asize, 2)
        else:
            img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, asize, 2)

    #OTSU threshold generation
    elif (Threshtype == 2):
        if (fg_color == 1):
            img_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ret = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
        else:
            img_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ret = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    return img_bin, ret


def binarize(source, Threshtype, gamma, md_filter, g_blur, autolvl, fg_color, \
             laplacian, scharr, sobel, lowpass, asize, bsize, wsize, thresh):
    global img
    global img_bin

    img = source

    img = adjust_gamma(img, gamma)

    # applies a low-pass filter
    if(lowpass ==1):
        img = Hamming_window(img, wsize)


    # making a 5x5 array of all 1's for median filter, and a disk for the autolevel filter
    darray = np.zeros((5, 5)) + 1
    selem = disk(bsize)

    # applying median filter
    if (md_filter == 1):
        img = median(img, darray)

    # applying gaussian blur
    if (g_blur == 1):
        img = cv2.GaussianBlur(img, (bsize, bsize), 0)

    # applying autolevel filter
    if (autolvl == 1):
        img = autolevel(img, selem=selem)

    # applying a scharr filter, and then taking that image and weighting it 25% with the original
    # this should bring out the edges without separating each "edge" into two separate parallel ones
    if (scharr == 1):
        ddepth = cv2.CV_16S
        grad_x = cv2.Scharr(img, ddepth, 1, 0)
        grad_y = cv2.Scharr(img, ddepth, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        dst = cv2.convertScaleAbs(dst)
        img = cv2.addWeighted(img, 0.75, dst, 0.25, 0)
        img = cv2.convertScaleAbs(img)

    # applying sobel filter
    if (sobel == 1):
        scale = 1;
        delta = 0;
        ddepth = cv2.CV_16S
        grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        dst = cv2.convertScaleAbs(dst)
        img = cv2.addWeighted(img, 0.75, dst, 0.25, 0)
        img = cv2.convertScaleAbs(img)

    # applying laplacian filter
    if (laplacian == 1):
        ddepth = cv2.CV_16S
        dst = cv2.Laplacian(img, ddepth, ksize=5)

        # dst = cv2.Canny(img, 100, 200); # canny edge detection test
        dst = cv2.convertScaleAbs(dst)
        img = cv2.addWeighted(img, 0.75, dst, 0.25, 0)
        img = cv2.convertScaleAbs(img)

    # this is my attempt at a fast fourier transformation with a band pass filter
    # I would highly reccomend taking a look at this and seeing if its working right
    # I had no idea what I was doing but I think it works, it could use some more testing because it just
    # kinda makes the image blurry, but that could be the band pass filter
    #if fourier == 1:

        #rows, cols = img.shape
        #m = cv2.getOptimalDFTSize(rows)
        #n = cv2.getOptimalDFTSize(cols)
        #padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        #planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
        #complexI = cv2.merge(planes)
        #dft = cv2.dft(np.float32(complexI), flags=cv2.DFT_COMPLEX_OUTPUT)
        #dft_shift = np.fft.fftshift(dft)

        # Band pass filter mask

        #rows, cols = img.shape
        #crow, ccol = int(rows / 2), int(cols / 2)

        #mask = np.zeros((rows, cols, 2), np.uint8)
        #r_out = 80
        #r_in = 10
        #center = [crow, ccol]
        #x, y = np.ogrid[:rows, :cols]
        #mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                                   #((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
        #mask[mask_area] = 1

        # apply mask and inverse DFT
        #fshift = dft_shift * mask

        #f_ishift = np.fft.ifftshift(fshift)
        #img_back = cv2.idft(f_ishift)
        #img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        #cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        #img = cv2.convertScaleAbs(img_back)

    img_bin, ret = thresh_it(img, Threshtype, fg_color, asize, thresh)
    return img, img_bin, ret
