"""settings: Generates the graphical user interface for
selecting the image detection, graph extraction, and
GT parameter calculation settings on a selected image.

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

from tkinter.ttk import Progressbar
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statistics import stdev
from PIL import Image, ImageTk
import networkx as nx
import os
import re
import csv
import cv2
import time
import skel_ID
from StructuralGT import GetWeights, GT_Params, process_image
import numpy as np
import sknw
import datetime
import single_image_looper



def progress(currentValue):

    # setting the progressbar
    progressbar["value"]=currentValue
    progressbar.update()  # Force an update of the GUI

def update_label(label_text):
    label_count.set(label_text)
    settings.update()


def norm_value(value, data_list):
    min_val = min(data_list)
    max_val = max(data_list)
    #min_val = -1
    #max_val = 1

    #Normalize a value in the range from 0 to 1 based on a given data range
    norm_value = (value - min_val) / (max_val - min_val)

    return norm_value

def save_data(src, Thresh_method, gamma, md_filter, g_blur, autolvl, fg_color, asize, bsize, wsize, thresh, laplacian, \
              scharr, sobel, lowpass, merge_nodes, prune, clean, Exp_EL, Do_gexf, r_size, weighted, display_nodeID, \
              no_self_loops, multigraph, Do_kdist, Do_dia, Do_BCdist, Do_CCdist, Do_ECdist, Do_GD, Do_Eff, Do_clust, \
              Do_ANC, Do_Ast, heatmap):

    # Updating the label alert
    update_label("Processing image...")

    # processing the image itself to create a binary image img_bin and updating progress
    img, img_bin, ret = process_image.binarize(src, Thresh_method, gamma, md_filter, g_blur, autolvl, fg_color, \
                                               laplacian, scharr, sobel, lowpass, asize, bsize, wsize, thresh)
    progress(10)
    update_label("Extracting graph...")

    # making the skeleton image and updating progress
    newskel, skel_int, Bp_coord_x, Bp_coord_y, Ep_coord_x, Ep_coord_y = \
        skel_ID.make_skel(img_bin, merge_nodes, prune, clean, r_size)
    progress(25)
    update_label("Formatting graph...")

    # skeleton analysis object with sknw
    if multigraph:
        G = sknw.build_sknw(newskel, multi=True)
        for (s, e) in G.edges():
            for k in range(int(len(G[s][e]))):
                G[s][e][k]['length'] = G[s][e][k]['weight']
                if (G[s][e][k]['weight'] == 0):
                    G[s][e][k]['length'] = 2
        # since the skeleton is already built by skel_ID.py the weight that sknw finds will be the length
        # if we want the actual weights we get it from GetWeights.py, otherwise we drop them
        for (s, e) in G.edges():
            if (weighted == 1):
                for k in range(int(len(G[s][e]))):
                    ge = G[s][e][k]['pts']
                    pix_width, wt = GetWeights.assignweightsbywidth(ge, img_bin)
                    G[s][e][k]['pixel width'] = pix_width
                    G[s][e][k]['weight'] = wt
            else:
                for k in range(int(len(G[s][e]))):
                    try:
                        del G[s][e][k]['weight']
                    except KeyError:
                        None

    else:
        G = sknw.build_sknw(newskel)

        # the actual length of the edges we want is stored as weight, so the two are set equal
        # if the weight is 0 the edge length is set to 2
        for (s, e) in G.edges():
            G[s][e]['length'] = G[s][e]['weight']
            if (G[s][e]['weight'] == 0):
                G[s][e]['length'] = 2
        # since the skeleton is already built by skel_ID.py the weight that sknw finds will be the length
        # if we want the actual weights we get it from GetWeights.py, otherwise we drop them
        for (s, e) in G.edges():
            if(weighted == 1):
                ge = G[s][e]['pts']
                pix_width, wt = GetWeights.assignweightsbywidth(ge, img_bin)
                G[s][e]['pixel width'] = pix_width
                G[s][e]['weight'] = wt
            else:
                del G[s][e]['weight']

    # Removing all instances of edges were the start and end are the same, or "self loops"
    if no_self_loops:
        if multigraph:
            g = G
            for (s, e)  in list(G.edges()):
                if s == e:
                   g.remove_edge(s, e)
            G = g
        else:
            for (s, e) in G.edges():
                if s == e:
                    G.remove_edge(s, e)

    progress(30)

    # running GT calcs
    data, klist, Tlist, BCdist, CCdist, ECdist= \
        GT_Params.run_GT_calcs(G, Do_kdist, Do_dia, Do_BCdist, Do_CCdist, Do_ECdist, Do_GD, Do_Eff, \
                               Do_clust, Do_ANC, Do_Ast, Do_WI, multigraph)
    progress(85)

    # running weighted calcs if requested
    if(weighted == 1):
        w_data, w_klist, w_BCdist, w_CCdist, w_ECdist= \
            GT_Params.run_weighted_GT_calcs(G, Do_kdist, Do_BCdist, Do_CCdist, Do_ECdist, Do_ANC, Do_Ast, Do_WI, \
                                            multigraph)
    progress(90)
    # original, filtered, and binary image, with histogram
    raw_img = src
    img_filt = img
    img_bin = img_bin
    histo = cv2.calcHist([img_filt], [0], None, [256], [0, 256])

    update_label("Generating PDF results...")
    # exporting to pdf
    with PdfPages(file) as pdf:
        font1 = {'fontsize': 12}
        font2 = {'fontsize': 9}
        # plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
        f1 = plt.figure(figsize=(8.5, 8.5), dpi=400)
        f1.add_subplot(2, 2, 1)
        plt.imshow(raw_img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("Original Image")
        f1.add_subplot(2, 2, 2)
        plt.imshow(img_filt, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("Processed Image")
        f1.add_subplot(2, 2, 3)
        plt.plot(histo)
        if (Thresh_method == 0):
            Th = np.array([[thresh, thresh], [0, max(histo)]], dtype='object')
            plt.plot(Th[0], Th[1], ls='--', color='black')
        elif (Thresh_method == 2):
            Th = np.array([[ret, ret], [0, max(histo)]], dtype='object')
            plt.plot(Th[0], Th[1], ls='--', color='black')
        plt.yticks([])
        plt.title("Histogram of Processed Image")
        plt.xlabel("Pixel values")
        plt.ylabel("Counts")
        f1.add_subplot(2, 2, 4)
        plt.imshow(img_bin, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("Binary Image")

        pdf.savefig()
        plt.close()



        # plotting skeletal images
        f2 = plt.figure(figsize=(8.5, 11), dpi=400)
        f2.add_subplot(2, 1, 1)
        #skel_int = -1*(skel_int-1)
        plt.imshow(skel_int, cmap='gray')
        plt.scatter(Bp_coord_x, Bp_coord_y, s=0.25, c='b')
        plt.scatter(Ep_coord_x, Ep_coord_y, s=0.25, c='r')
        plt.xticks([])
        plt.yticks([])
        plt.title("Skeletal Image")
        f2.add_subplot(2, 1, 2)
        plt.imshow(src, cmap='gray')
        if multigraph:
            for (s, e) in G.edges():
                for k in range(int(len(G[s][e]))):
                    ge = G[s][e][k]['pts']
                    plt.plot(ge[:, 1], ge[:, 0], 'red')
        else:
            for (s, e) in G.edges():
                ge = G[s][e]['pts']
                plt.plot(ge[:, 1], ge[:, 0], 'red')

        # plotting the final graph with the nodes
        nodes = G.nodes()
        gn = np.array([nodes[i]['o'] for i in nodes])
        if (display_nodeID == 1):
            i = 0;
            for x, y in zip(gn[:, 1], gn[:, 0]):
                plt.annotate(i, (x, y), fontsize=5)
                i += 1
            plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)

        else:
            plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
        plt.xticks([])
        plt.yticks([])
        plt.title("Final Graph")
        pdf.savefig()
        plt.close()

        progress(95)

        # displaying all of the GT calculations requested
        f3 = plt.figure(figsize=(8.5, 11), dpi=300)
        if weighted == 1:
            f3.add_subplot(2, 1, 1)
            f3.patch.set_visible(False)
            plt.axis('off')
            colw = [2 / 3, 1 / 3]
            table = plt.table(cellText=data.values[:, :], loc='upper center', colWidths=colw, cellLoc='left')
            table.scale(1, 1.5)
            plt.title("Unweighted GT parameters")
            try:
                f3.add_subplot(2, 1, 2)
                f3.patch.set_visible(False)
                plt.axis('off')
                table2 = plt.table(cellText=w_data.values[:, :], loc='upper center', colWidths=colw, cellLoc='left')
                table2.scale(1, 1.5)
                plt.title("Weighted GT Parameters")
            except:
                pass

        else:
            f3.add_subplot(2, 2, 1)
            f3.patch.set_visible(False)
            plt.axis('off')
            colw = [2 / 3, 1 / 3]
            # try:
            table = plt.table(cellText=data.values[:, :], loc='center', colWidths=colw, cellLoc='left')
            table.scale(1, 1.5)
            # except:
            #     table = plt.table(cellText=[["Number of Nodes","Number of Edges"],["Value",]])
            plt.title("Unweighted GT Parameters")
            if Do_kdist:
                f3.add_subplot(2, 2, 2)
                bins1 = np.arange(0.5, max(klist) + 1.5, 1)
                try:
                    k_sig = str(round(stdev(klist),3))
                except:
                    k_sig = "N/A"
                k_txt = "Degree Distribution: $\sigma$=" + k_sig
                plt.hist(klist, bins=bins1)
                plt.title(k_txt)
                plt.xlabel("Degree")
                plt.ylabel("Counts")
            if (Do_clust and multigraph == 0):
                f3.add_subplot(2, 2, 3)
                binsT = np.linspace(min(Tlist), max(Tlist), 50)
                try:
                    T_sig = str(round(stdev(Tlist), 3))
                except:
                    T_sig = "N/A"
                T_txt = "Clustering Coefficients: $\sigma$=" + T_sig
                plt.hist(Tlist, bins=binsT)
                plt.title(T_txt)
                plt.xlabel("Clust. Coeff.")
                plt.ylabel("Counts")
        try:
            pdf.savefig()
        except:
            None
        try:
            plt.close()
        except:
            None

        if(multigraph == 0 and weighted == 0):
            if(Do_BCdist or Do_CCdist or Do_ECdist):
                f4 = plt.figure(figsize=(8.5, 11), dpi=400)
                if Do_BCdist:
                    f4.add_subplot(2, 2, 1)
                    bins2 = np.linspace(min(BCdist), max(BCdist), 50)
                    try:
                        BC_sig = str(round(stdev(BCdist), 3))
                    except:
                        BC_sig = "N/A"
                    BC_txt = "Betweenness Centrality: $\sigma$=" + BC_sig
                    plt.hist(BCdist, bins=bins2)
                    plt.title(BC_txt)
                    plt.xlabel("Betweenness value")
                    plt.ylabel("Counts")
                if Do_CCdist:
                    f4.add_subplot(2, 2, 2)
                    bins3 = np.linspace(min(CCdist), max(CCdist), 50)
                    try:
                        CC_sig = str(round(stdev(CCdist), 3))
                    except:
                        CC_sig = "N/A"
                    CC_txt = "Closeness Centrality: $\sigma$=" + CC_sig
                    plt.hist(CCdist, bins=bins3)
                    plt.title(CC_txt)
                    plt.xlabel("Closeness value")
                    plt.ylabel("Counts")
                if Do_ECdist:
                    f4.add_subplot(2, 2, 3)
                    bins4 = np.linspace(min(ECdist), max(ECdist), 50)
                    try:
                        EC_sig = str(round(stdev(ECdist), 3))
                    except:
                        EC_sig = "N/A"
                    EC_txt = "Eigenvector Centrality: $\sigma$=" + EC_sig
                    plt.hist(ECdist, bins=bins4)
                    plt.title(EC_txt)
                    plt.xlabel("Eigenvector value")
                    plt.ylabel("Counts")



            try:
                pdf.savefig()
            except:
                None
            try:
                plt.close()
            except:
                None


        # displaying weighted GT parameters if requested
        if (weighted == 1):
            if multigraph:
                Do_BCdist = 0
                Do_ECdist = 0
                Do_clust = 0
            g_count = Do_kdist + Do_clust + Do_BCdist + Do_CCdist + Do_ECdist
            g_count2 = g_count - Do_clust + 1
            index = 1
            if(g_count > 2):
                sy1 = 2
                fnt = font2
            else:
                sy1 = 1
                fnt = font1
            f4 = plt.figure(figsize=(8.5, 11), dpi=400)
            if Do_kdist:
                f4.add_subplot(sy1, 2, index)
                bins1 = np.arange(0.5, max(klist) + 1.5, 1)
                try:
                    k_sig = str(round(stdev(klist), 3))
                except:
                    k_sig = "N/A"
                k_txt = "Degree Distribution: $\sigma$=" + k_sig
                plt.hist(klist, bins=bins1)
                plt.title(k_txt, fontdict=fnt)
                plt.xlabel("Degree", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)
                index += 1
            #if Do_clust:
                #f4.add_subplot(sy1, 2, index)
                #binsT = np.linspace(min(Tlist), max(Tlist), 50)
                #try:
                    #T_sig = str(round(stdev(Tlist), 3))
                #except:
                    #T_sig = "N/A"
                #T_txt = "Clustering Coefficients: $\sigma$=" + T_sig
                #plt.hist(Tlist, bins=binsT)
                #plt.title(T_txt, fontdict=fnt)
                #plt.xlabel("Clust. Coeff.", fontdict=fnt)
                #plt.ylabel("Counts", fontdict=fnt)
                #index += 1
            if Do_BCdist:
                f4.add_subplot(sy1, 2, index)
                bins2 = np.linspace(min(BCdist), max(BCdist), 50)
                try:
                    BC_sig = str(round(stdev(BCdist), 3))
                except:
                    BC_sig = "N/A"
                BC_txt = "Betweenness Centrality: $\sigma$=" + BC_sig
                plt.hist(BCdist, bins=bins2)
                plt.title(BC_txt, fontdict=fnt)
                plt.xlabel("Betweenness value", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)
                index += 1
            if Do_CCdist:
                f4.add_subplot(sy1, 2, index)
                bins3 = np.linspace(min(CCdist), max(CCdist), 50)
                try:
                    CC_sig = str(round(stdev(CCdist), 3))
                except:
                    CC_sig = "N/A"
                CC_txt = "Closeness Centrality: $\sigma$=" + CC_sig
                plt.hist(CCdist, bins=bins3)
                plt.title(CC_txt, fontdict=fnt)
                plt.xlabel("Closeness value", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)
                index += 1
            if Do_ECdist:
                f4.add_subplot(sy1, 2, index)
                bins4 = np.linspace(min(ECdist), max(ECdist), 50)
                try:
                    EC_sig = str(round(stdev(ECdist), 3))
                except:
                    EC_sig = "N/A"
                BC_txt = "Eigenvector Centrality: $\sigma$=" + EC_sig
                plt.hist(BCdist, bins=bins4)
                plt.title(BC_txt, fontdict=fnt)
                plt.xlabel("Eigenvector value", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)


            pdf.savefig()
            plt.close()


            f5 = plt.figure(figsize=(8.5, 11), dpi=400)
            if(g_count2 > 2):
                sy2 = 2
                fnt = font2
            else:
                sy2 = 1
                fnt = font1
            index = 1
            #f5.add_subplot(sy2, 2, index)
            #weight_data = []
            #if multigraph:
                #for (s, e) in G.edges():
                    #for k in range(int(len(G[s][e]))):
                        #weight_data.append(G[s][e][k]['weight'])
                    #try:
                        #bins7 = np.linspace(min(weight_data), max(weight_data), 50)
                    #except:
                        # this only happens if there are no edges I hope
                        #bins7 = np.linspace(0, 1, 50)
            #else:
                #for (s, e) in G.edges():
                    #weight_data.append(G[s][e]['weight'])
                #try:
                    #bins7 = np.linspace(min(weight_data), max(weight_data), 50)
                #except:
                    # this only happens if there are no edges I hope
                    #bins7 = np.linspace(0, 1, 50)
            #try:
                #wt_sig = str(round(stdev(weight_data),3))
            #except:
                #wt_sig = "N/A"
            #wt_txt = "Edge weight Dist.: $\sigma$=" + wt_sig
            #plt.hist(weight_data, bins=bins7)
            #plt.title(wt_txt, fontdict=fnt)
            #plt.xlabel("Line Weight", fontdict=fnt)
            #plt.xticks(fontsize=8)
            #plt.ylabel("Counts", fontdict=fnt)
            #index += 1
            if Do_kdist:
                f5.add_subplot(sy2, 2, index)
                bins4 = np.arange(0.5, max(w_klist) + 1.5, 1)
                try:
                    wk_sig = str(round(stdev(w_klist),3))
                except:
                    wk_sig = "N/A"
                wk_txt = "Weighted Degree: $\sigma$=" + wk_sig
                plt.hist(w_klist, bins=bins4)
                plt.title(wk_txt, fontdict=fnt)
                plt.xlabel("Degree", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)
                index += 1
            if Do_BCdist:
                f5.add_subplot(sy2, 2, index)
                bins5 = np.linspace(min(w_BCdist), max(w_BCdist), 50)
                plt.hist(w_BCdist, bins=bins5)
                try:
                    wBC_sig = str(round(stdev(w_BCdist),3))
                except:
                    wBC_sig = "N/A"
                wBC_txt = "Width-Weighted Betweeness: $\sigma$=" + wBC_sig
                plt.title(wBC_txt, fontdict=fnt)
                plt.xlabel("Betweenness value", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)
                index += 1
            if Do_CCdist:
                f5.add_subplot(sy2, 2, index)
                bins6 = np.linspace(min(w_CCdist), max(w_CCdist), 50)
                try:
                    wCC_sig = str(round(stdev(w_CCdist),3))
                except:
                    wCC_sig = "N/A"
                wCC_txt = "Length-Weighted Closeness: $\sigma$=" + wCC_sig
                plt.hist(w_CCdist, bins=bins6)
                plt.title(wCC_txt, fontdict=fnt)
                plt.xlabel("Closeness value", fontdict=fnt)
                plt.xticks(fontsize=8)
                plt.ylabel("Counts", fontdict=fnt)
                index += 1
            if Do_ECdist:
                f5.add_subplot(sy2, 2, index)
                bins7 = np.linspace(min(w_ECdist), max(w_ECdist), 50)
                plt.hist(w_ECdist, bins=bins7)
                try:
                    wEC_sig = str(round(stdev(w_ECdist),3))
                except:
                    wEC_sig = "N/A"
                wEC_txt = "Width-Weighted Eigenvector Cent.: $\sigma$=" + wEC_sig
                plt.title(wEC_txt, fontdict=fnt)
                plt.xlabel("Eigenvetor value", fontdict=fnt)
                plt.ylabel("Counts", fontdict=fnt)

            pdf.savefig()
            plt.close()

        if heatmap:
            sz = 30
            lw = 1.5
            update_label("Generating heat maps...")
            time.sleep(0.5)
            if(Do_kdist == 1):
                f6a = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6a.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=klist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Degree Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_kdist == 1 and weighted == 1):
                f6b = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6b.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_klist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Weighted Degree Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_clust == 1 and multigraph == 0):
                f6c = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6c.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=Tlist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Clustering Coefficient Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_BCdist == 1 and multigraph == 0):
                f6d = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6d.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=BCdist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Betweenness Centrality Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_BCdist == 1 and weighted == 1 and multigraph == 0):
                f6e = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6e.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_BCdist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Width-Weighted Betweenness Centrality Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_CCdist == 1):
                f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6f.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=CCdist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Closeness Centrality Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_CCdist == 1 and weighted == 1):
                f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6f.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_CCdist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Length-Weighted Closeness Centrality Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_ECdist == 1 and multigraph == 0):
                f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6h.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=ECdist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Eigenvector Centrality Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()
            if (Do_ECdist == 1 and weighted == 1 and multigraph == 0):
                f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                f6h.add_subplot(1, 1, 1)
                plt.imshow(src, cmap='gray')
                nodes = G.nodes()
                gn = np.array([nodes[i]['o'] for i in nodes])
                plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_ECdist, cmap='plasma')
                if multigraph:
                    for (s, e) in G.edges():
                        for k in range(int(len(G[s][e]))):
                            ge = G[s][e][k]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                else:
                    for (s, e) in G.edges():
                        ge = G[s][e]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                plt.xticks([])
                plt.yticks([])
                plt.title('Width-Weighted Eigenvector Centrality Heatmap', fontdict=font1)
                cbar = plt.colorbar()
                cbar.set_label('Value')
                pdf.savefig()
                plt.close()

        f8 = plt.figure(figsize=(8.5, 8.5), dpi=300)
        f8.add_subplot(1, 1, 1)
        plt.text(0.5, 0.5, run_info, horizontalalignment='center', verticalalignment='center')
        plt.xticks([])
        plt.yticks([])
        pdf.savefig()
        plt.close()

    if(Exp_EL == 1):
        if(weighted == 1):
            fields = ['Source', 'Target', 'Weight', 'Length']
            el = nx.generate_edgelist(G, delimiter=',', data=["weight", "length"])
            with open(file2, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(fields)
                for line in el:
                    line = str(line)
                    row = line.split(',')
                    try:
                        writer.writerow(row)
                    except:
                        None
            csvfile.close()
        else:
            fields = ['Source', 'Target']
            el = nx.generate_edgelist(G, delimiter=',', data=False)
            with open(file2, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(fields)
                for line in el:
                    line = str(line)
                    row=line.split(',')
                    try:
                        writer.writerow(row)
                    except:
                        None
            csvfile.close()

    # exporting as gephi file
    if(Do_gexf == 1):
        if multigraph:
            # deleting extraneous info and then exporting the final skeleton
            for (x) in G.nodes():
                del G.nodes[x]['pts']
                del G.nodes[x]['o']
            for (s, e) in G.edges():
                for k in range(int(len(G[s][e]))):
                    try:
                        del G[s][e][k]['pts']
                    except KeyError:
                        None

            nx.write_gexf(G, file1)
        else:
            # deleting extraneous info and then exporting the final skeleton
            for (x) in G.nodes():
                del G.nodes[x]['pts']
                del G.nodes[x]['o']
            for (s, e) in G.edges():
                del G[s][e]['pts']
            nx.write_gexf(G, file1)

    update_label("Done")


def get_checks():

    # global variables for all of the check boxes
    global Gamma, md_filter, g_blur, autolvl, fg_color, laplacian, scharr, sobel, lowpass, Thresh_method, asize, \
        bsize, wsize, thresh, merge_nodes, prune, clean, Exp_EL, Do_gexf, r_size, weighted, display_nodeID, \
        no_self_loops, multigraph, Do_clust, Do_ANC, Do_GD, Do_Eff, Do_kdist, Do_BCdist, Do_CCdist, Do_ECdist, Do_dia, \
        Do_Ast, Do_WI, heatmap

    # checkboxes for image detection settings
    thresh = var10.get()
    md_filter = var11.get()
    g_blur = var12.get()
    autolvl = var13.get()
    fg_color = var14.get()
    laplacian = var16.get()
    scharr = var17.get()
    sobel = var18.get()
    Gamma = var19.get()
    lowpass = var1f.get()

    # image threshold method
    Thresh_method = var15.get()

    # local kernel size for adaptive threshold
    asize = adaptsize.get()

    # error parsing to round off the value to the nearest odd integer
    # if a non-number was entered, the default value is used
    # max value is set at 255
    try:
        asize = int(asize)
    except TypeError:
        asize = 11
        adaptsize.delete(0, END)
        adaptsize.insert('end', asize)
    if (asize % 2 == 0):
        asize = int(asize) + 1
        adaptsize.delete(0, END)
        adaptsize.insert('end', asize)
    if (asize > 511):
        asize = 511
        adaptsize.delete(0, END)
        adaptsize.insert('end', asize)
    elif (asize < 0):
        asize = 3
        adaptsize.delete(0, END)
        adaptsize.insert('end', asize)

    # blurring kernel size if using adaptive gaussian blur
    bsize = blursize.get()

    # error parsing to round off the value to the nearest odd integer
    # if a non-number was entered, the default value is used
    # max value is set at 255
    try:
        bsize = int(bsize)
    except TypeError:
        bsize = 3
        blursize.delete(0, END)
        blursize.insert('end', bsize)
    if(bsize%2 == 0):
        bsize = int(bsize) + 1
        blursize.delete(0, END)
        blursize.insert('end', bsize)
    if(bsize > 255):
        bsize = 255
        blursize.delete(0, END)
        blursize.insert('end', bsize)
    elif(bsize<0):
        bsize = 3
        blursize.delete(0, END)
        blursize.insert('end', bsize)

    # checkboxes for graph extraction settings, which is mainly for skeleton image building
    merge_nodes = var21.get()
    prune = var22.get()
    clean = var23.get()
    Do_gexf = var24.get()
    weighted = var25.get()
    display_nodeID = var26.get()
    no_self_loops = var27.get()
    multigraph = var28.get()
    Exp_EL = var20.get()

    # maximum size of objects to remove if removing disconnected segments
    r_size = removesize.get()

    # error parsing to round off to the nearest integer
    # if a non-number is entered, the default value is used
    # maximum size is 1000, minimum size is 2
    try:
        r_size = int(r_size)
    except TypeError:
        r_size = 100
        removesize.delete(0, END)
        removesize.insert('end', r_size)
    if(r_size > 50000):
        r_size = 50000
        removesize.delete(0, END)
        removesize.insert('end', r_size)
    elif(r_size<0):
        r_size = 2
        removesize.delete(0, END)
        removesize.insert('end', r_size)

    wsize = windowsize.get()
    try:
        wsize = int(wsize)
    except TypeError:
        wsize = 10
        windowsize.delete(0, END)
        windowsize.insert('end', wsize)
    if(wsize > 500):
        wsize = 500
        windowsize.delete(0, END)
        windowsize.insert('end', wsize)
    elif(wsize<0):
        wsize = 2
        windowsize.delete(0, END)
        windowsize.insert('end', wsize)

    # NetworkX Calculation Settings checkboxes
    Do_ANC = var31.get()
    Do_GD = var32.get()
    Do_Eff = var33.get()
    Do_clust = var37.get()
    Do_kdist = var34.get()
    Do_BCdist = var35.get()
    Do_CCdist = var36.get()
    Do_ECdist = var30.get()
    Do_dia = var38.get()
    Do_Ast = var39.get()
    Do_WI = var40.get()
    heatmap = var29.get()


    # returning all the values
    return Gamma, md_filter, g_blur, autolvl, fg_color, Thresh_method, asize, bsize, wsize, thresh, laplacian, scharr, \
           sobel, lowpass, merge_nodes, prune, clean, Exp_EL, Do_gexf, r_size, weighted, display_nodeID, \
           no_self_loops, multigraph, Do_clust, Do_ANC, Do_GD, Do_Eff, Do_kdist, Do_BCdist, Do_CCdist, Do_ECdist, \
           Do_dia, Do_Ast, Do_WI, heatmap


def get_Settings(filename):

    # similar to the start of the csv file, this is just getting all the relevant settings to display in the pdf
    global run_info
    run_info = "Run Info\n"
    run_info = run_info + oldfilename
    now = datetime.datetime.now()
    run_info = run_info + " || " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n"
    if Thresh_method == 0:
        run_info = run_info + " || Global Threshold (" + str(thresh) + ")"
    elif Thresh_method == 1:
        run_info = run_info + " || Adaptive Threshold, " + str(asize) + " bit kernel"
    elif Thresh_method == 2:
        run_info = run_info + " || OTSU Threshold"
    if Gamma != 1:
        run_info = run_info + "|| Gamma = " + str(Gamma)
    if md_filter:
        run_info = run_info + " || Median Filter"
    if g_blur:
        run_info = run_info + " || Gaussian Blur, " + str(bsize) + " bit kernel"
    if autolvl:
        run_info = run_info + " || Autolevel"
    if fg_color:
        run_info = run_info + " || Dark Foreground"
    if laplacian:
        run_info = run_info + " || Laplacian Gradient"
    if scharr:
        run_info = run_info + " || Scharr Gradient"
    if sobel:
        run_info = run_info + " || Sobel Gradient"
    if lowpass:
        run_info = run_info + " || Low-pass filter" + str(wsize)
    run_info = run_info + "\n"
    if merge_nodes:
        run_info = run_info + " || Merge Nodes"
    if prune:
        run_info = run_info + " || Prune Dangling Edges"
    if clean:
        run_info = run_info + " || Remove Objects of Size " + str(r_size)
    if no_self_loops:
        run_info = run_info + " || Remove Self Loops"
    if multigraph:
        run_info = run_info + " || Multigraph allowed"

    return run_info


def Preview_button():

    # run get_checks() to get all the entered information
    # this function only actually needs image detection settings
    get_checks()

    # calling process_image.py to process the actual image
    img, img_bin, ret = process_image.binarize(src, Thresh_method, Gamma, md_filter, g_blur, autolvl, fg_color, \
                                               laplacian, scharr, sobel, lowpass, asize, bsize, wsize, thresh)

    # getting the raw image from src and the new filtered image and the binary image (img_bin)
    raw_img = src
    img_filt = img
    img_bin = img_bin

    # making a histogram of the grayscale values of all the pixels in the filtered image
    histo = cv2.calcHist([img_filt], [0], None, [256], [0,256])
    # making a plot with the matlab plotting module
    f = plt.figure(figsize=(4, 4), dpi=200)

    # 2x2 plot, top left section is original grayscale resized image
    f.add_subplot(2, 2, 1)
    plt.imshow(raw_img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Original Image", fontsize= 10)

    # top right plot is the processed image
    # the processed image is affected by the four checkboxes in the image detection settings
    # use the preview button to ensure the image has the starkest contrast possible
    f.add_subplot(2, 2, 2)
    plt.imshow(img_filt, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Processed Image", fontsize= 10)

    # bottom left is histogram of the processed image
    f.add_subplot(2, 2, 3)
    plt.plot(histo, color='blue')
    if(Thresh_method == 0):
        Th = np.array([[thresh, thresh],[0,max(histo)]], dtype='object')
        plt.plot(Th[0],Th[1], ls='--', color='black')
    elif(Thresh_method == 2):
        Th = np.array([[ret, ret], [0, max(histo)]], dtype='object')
        plt.plot(Th[0], Th[1], ls='--', color='black')
    plt.yticks([])
    plt.title("Histogram of Processed Image", fontsize= 10)
    plt.xlabel("Pixel values")
    plt.ylabel("Counts")

    # bottom right is the binary image
    f.add_subplot(2, 2, 4)
    plt.imshow(img_bin, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Binary Image", fontsize= 10)

    # showing the plots
    plt.show(block=True)


def Proceed_button():

    # starting the image analysis, first by finding the values of all the check boxes
    # also getting the start time for the actual time intensive part of the program
    print("Running...")
    button1["state"] = "disabled"
    button3["state"] = "disabled"
    button4["state"] = "disabled"
    start = time.time()
    progress(1)
    get_checks()
    get_Settings(oldfilename)
    progress(5)

    # save_data calls everything else and saves the results
    save_data(src, Thresh_method, Gamma, md_filter, g_blur, autolvl, fg_color, asize, bsize, wsize, thresh, \
              laplacian, scharr, sobel, lowpass, merge_nodes, prune, clean, Exp_EL, Do_gexf, r_size, weighted, \
              display_nodeID, no_self_loops, multigraph, Do_kdist, Do_dia, Do_BCdist, Do_CCdist, Do_ECdist, Do_GD, \
              Do_Eff, Do_clust, Do_ANC, Do_Ast, heatmap)

    progress(100)
    update_label("Done!")
    print("Done")

    # displaying the time it took to run the image analysis in minutes
    end = time.time()
    elapse1 = (end - start) / 60
    elapse = str("%.2f" % elapse1)
    print(elapse + " minutes")
    button1["state"] = "active"
    button3["state"] = "active"
    button4['state'] = "active"


def adjust_settings(root, source, saveloc, filename):

    # adjusting the settings for a new image
    root.destroy()
    global src, file, file1, file2, oldfilename
    src = source
    filename = re.sub('.png', '', filename)
    filename = re.sub('.tif', '', filename)
    filename = re.sub('.jpg', '', filename)
    filename = re.sub('.jpeg', '', filename)
    gfile = filename + "_graph.gexf"
    ELfile = filename + "_EL.csv"
    oldfilename = filename
    filename = filename + "_SGT_results.pdf"
    file = os.path.join(saveloc, filename)
    file1 = os.path.join(saveloc, gfile)
    file2 = os.path.join(saveloc, ELfile)
    update_label("Ready to Proceed")


def make_settings(root, source, saveloc, filename):

    # close previous window, open a new one
    root.destroy()
    global settings
    settings = Tk()
    settings.title("StructuralGT Settings")

    # file is the regular file, file1 is the gfile for gephi
    global src, file, file1, file2, oldfilename

    src = source

    # making the new filenames
    filename = re.sub('.png', '', filename)
    filename = re.sub('.tif', '', filename)
    filename = re.sub('.jpg', '', filename)
    filename = re.sub('.jpeg', '', filename)
    gfile = filename + "_graph.gexf"
    ELfile = filename + "_EL.csv"
    oldfilename = filename
    filename = filename + "_SGT_results.pdf"
    file = os.path.join(saveloc, filename)
    file1 = os.path.join(saveloc, gfile)
    file2 = os.path.join(saveloc, ELfile)

    # setting the frames for the window
    frame1 = Frame(settings)
    frame2 = Frame(settings)
    frame3 = Frame(settings)
    frame4 = Frame(settings)

    # separating the gui into four sections
    label1 = Label(settings, text = 'Image Detection Settings', bg="Gray", font="Bold", padx=200)
    label1.grid(row=0, column=0)
    frame1.grid(row=1, column=0)
    label2 = Label(settings, text = 'Binary Image Preview', bg="Gray", font="Bold", padx=211)
    label2.grid(row=2, column=0)
    frame2.grid(row=3, column=0)
    label3 = Label(settings, text = 'Graph Extraction Settings', bg="Gray", font="Bold", padx=220)
    label3.grid(row=0, column=1)
    frame3.grid(row=1, column=1)
    label4 = Label(settings, text = 'NetworkX Calculation Settings', bg="Gray", font="Bold", padx=205)
    label4.grid(row=2, column=1)
    frame4.grid(row=3, column=1)

    # this is for all the boolean check boxes
    # please note that its only some of the numbers from 11 to 39, not all of them
    # the the 10s place is the frame the variable is in
    global  var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var1f, var20, var21, var22, var23, \
        var24, var25, var26, var27, var28, var29, var30, var31, var32, var33, var34, var35, var36, var37, var38, \
        var39, var40, var41

    var10 = IntVar()
    var11 = IntVar()
    var12 = IntVar()
    var13 = IntVar()
    var14 = IntVar()
    var15 = IntVar()
    var16 = IntVar()
    var17 = IntVar()
    var18 = IntVar()
    var19 = DoubleVar()
    var1f = IntVar()
    var20 = IntVar()
    var21 = IntVar()
    var22 = IntVar()
    var23 = IntVar()
    var24 = IntVar()
    var25 = IntVar()
    var26 = IntVar()
    var27 = IntVar()
    var28 = IntVar()
    var29 = IntVar()
    var30 = IntVar()
    var31 = IntVar()
    var32 = IntVar()
    var33 = IntVar()
    var34 = IntVar()
    var35 = IntVar()
    var36 = IntVar()
    var37 = IntVar()
    var38 = IntVar()
    var39 = IntVar()
    var40 = IntVar()

    # all the checkboxes and their corresponding variables when clicked
    c11 = Checkbutton(frame1, text='Apply median filter', variable=var11, onvalue=1, offvalue=0)
    c12 = Checkbutton(frame1, text='Apply gaussian blur', variable=var12, onvalue=1, offvalue=0)
    c13 = Checkbutton(frame1, text='Use Autolevel', variable=var13, onvalue=1, offvalue=0)
    c14 = Checkbutton(frame1, text='Foreground is Dark', variable=var14, onvalue=1, offvalue=0)
    c16 = Checkbutton(frame1, text='Use Laplacian Gradient', variable=var16, onvalue=1, offvalue=0)
    c17 = Checkbutton(frame1, text='Use Scharr Gradient', variable=var17, onvalue=1, offvalue=0)
    c18 = Checkbutton(frame1, text='Use Sobel Gradient', variable=var18, onvalue=1, offvalue=0)
    c1f = Checkbutton(frame1, text='Apply Low-pass Filter', variable=var1f, onvalue=1, offvalue=0)
    c20 = Checkbutton(frame3, text='Export edge list', variable=var20, onvalue=1, offvalue=0)
    c21 = Checkbutton(frame3, text='Merge nearby nodes', variable=var21, onvalue=1, offvalue=0)
    c22 = Checkbutton(frame3, text='Prune dangling edges', variable=var22, onvalue=1, offvalue=0)
    c23 = Checkbutton(frame3, text='Remove disconnected segments', variable=var23, onvalue=1, offvalue=0)
    c24 = Checkbutton(frame3, text='Export as GEXF file', variable=var24, onvalue=1, offvalue=0)
    c25 = Checkbutton(frame3, text='Assign edge weights by diameters', variable=var25, onvalue=1, offvalue=0)
    c26 = Checkbutton(frame3, text='Display Node IDs in Final Graph', variable=var26, onvalue=1, offvalue=0)
    c27 = Checkbutton(frame3, text='Remove Self-Loops', variable=var27, onvalue=1, offvalue=0)
    c28 = Checkbutton(frame3, text='Disable Multigraph', variable=var28, onvalue=0, offvalue=1)
    c40 = Checkbutton(frame4, text='Calculate Wiener Index', variable=var40, onvalue=1, offvalue=0)
    c30 = Checkbutton(frame4, text='Create eigenvector centrality histogram', variable=var30, onvalue=1, offvalue=0)
    c31 = Checkbutton(frame4, text='Calculate average nodal connectivity', variable=var31, onvalue=1, offvalue=0)
    c32 = Checkbutton(frame4, text='Calculate graph density', variable=var32, onvalue=1, offvalue=0)
    c33 = Checkbutton(frame4, text='Calculate global efficiency', variable=var33, onvalue=1, offvalue=0)
    c34 = Checkbutton(frame4, text='Create degree histogram', variable=var34, onvalue=1, offvalue=0)
    c35 = Checkbutton(frame4, text='Create betweenness centrality histogram', variable=var35, onvalue=1, offvalue=0)
    c36 = Checkbutton(frame4, text='Create closeness centrality histogram', variable=var36, onvalue=1, offvalue=0)
    c37 = Checkbutton(frame4, text='Calculate average clustering coefficient', variable=var37, onvalue=1, offvalue=0)
    c38 = Checkbutton(frame4, text='Calculate network diameter', variable=var38, onvalue=1, offvalue=0)
    c39 = Checkbutton(frame4, text='Calculate assortativity coefficient', variable=var39, onvalue=1, offvalue=0)
    c29 = Checkbutton(frame4, text='Display Heat Maps', variable=var29, onvalue=1, offvalue=0)

    # radio button for the type of image detection since you can only run one at once
    R1 = Radiobutton(frame1, text='Global Threshold', variable = var15, value=0)
    R2 = Radiobutton(frame1, text='Adaptive Threshold', variable = var15, value=1)
    R3 = Radiobutton(frame1, text='OTSU Threshold', variable = var15, value=2)
    R1.grid(row=0, column=1)
    R2.grid(row=1, column=1)
    R3.grid(row=2, column=1)

    s1 = Scale(frame1, label='Gamma adjust', variable=var19, from_=0.01, to=5.00, \
               resolution=0.01, length=300, orient=HORIZONTAL)
    s1.set(1.00)
    s2 = Scale(frame1, label='Global threshold value', variable=var10, from_=1, to=255, length=300, orient=HORIZONTAL)
    s2.set(127)


    # global variables for the only integer values that the user can enter
    global adaptsize, blursize, windowsize, threshval, removesize

    # making the entry boxes and giving default values and information for the type of info to enter
    adaptlabel = Label(frame1, text='Local threshold kernel:\n(Adaptive only, must be odd integer)')
    adaptsize = Entry(frame1)
    adaptsize.insert('end', 11)
    blurlabel = Label(frame1, text='Blurring kernel size:\n(Must be odd integer)')
    blursize = Entry(frame1)
    blursize.insert('end', 3)
    windowlabel = Label(frame1, text='Filter window size:')
    windowsize = Entry(frame1)
    windowsize.insert('end', 10)
    removelabel = Label(frame3, text="Remove object size:")
    removesize = Entry(frame3)
    removesize.insert('end', 500)

    # organizing the window
    c11.grid(row=7, column=1)
    c12.grid(row=6, column=1)
    c13.grid(row=6, column=0)
    c14.grid(row=2, column=0)
    c16.grid(row=7, column=0)
    c17.grid(row=9, column=0)
    c18.grid(row=8, column=0)
    s1.grid(row=0, column=0)
    s2.grid(row=1, column=0)
    c1f.grid(row=8, column=1)

    c20.grid(row=0, column=1)
    c21.grid(row=0, column=0)
    c22.grid(row=1, column=0)
    c23.grid(row=4, column=0)
    c24.grid(row=1, column=1)
    c25.grid(row=3, column=0)
    c26.grid(row=3, column=1)
    c27.grid(row=2, column=0)
    c28.grid(row=2, column=1)

    c31.grid(row=2, column=0)
    c32.grid(row=3, column=0)
    c33.grid(row=4, column=0)
    c34.grid(row=2, column=2)
    c35.grid(row=3, column=2)
    c36.grid(row=4, column=2)
    c30.grid(row=5, column=2)
    c37.grid(row=5, column=0)
    c38.grid(row=6, column=0)
    c39.grid(row=6, column=2)
    c29.grid(row=1, column=0)
    c40.grid(row=7, column=0)

    adaptlabel.grid(row=3, column=0)
    adaptsize.grid(row=3, column=1)
    blurlabel.grid(row=4, column=0)
    blursize.grid(row=4, column=1)
    windowlabel.grid(row=5, column=0)
    windowsize.grid(row=5, column=1)
    removelabel.grid(row=5, column=0)
    removesize.grid(row=5, column=1)

    def select_all():
        cbox_list = [c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40]
        for i in cbox_list:
            i.select()

    # preview and proceed buttons call their respective functions
    # preview just runs image detection
    # proceed does full data analysis and graph extraction
    global button1, button2, button3, button4
    button1 = Button(frame1, text="Advanced Preview...", command=Preview_button)
    button2 = Button(frame4, text="Proceed", command=Proceed_button)
    button3 = Button(frame4, text="New Image", command=lambda: single_image_looper.make_gui())
    button4 = Button(frame4, text='Select All...', command=lambda:select_all())

    button1.grid(row=9, column=1)
    button2.grid(row=8, column=2)
    button3.grid(row=10, column=2)
    button4.grid(row=0,column=0)

    global label_count
    label_count = StringVar()
    label_count.set("Ready to Proceed")
    label_counter = Label(frame4, textvariable=label_count)
    label_counter.grid(row=9, column=0)

    get_checks()
    img, img_bin, ret = process_image.binarize(src, Thresh_method, Gamma, md_filter, g_blur, autolvl, fg_color, \
                                               laplacian, scharr, sobel, lowpass, asize, bsize, wsize, thresh)
    img1 = Image.fromarray(img_bin)
    img2 = ImageTk.PhotoImage(image=img1)
    panel = Label(frame2, image=img2)
    panel.pack()

    def update(e):
        get_checks()
        img, img_bin, ret = process_image.binarize(src, Thresh_method, Gamma, md_filter, g_blur, autolvl, fg_color, \
                                                   laplacian, scharr, sobel, lowpass, asize, bsize, wsize, thresh)
        img1 = Image.fromarray(img_bin)
        img2 = ImageTk.PhotoImage(image=img1)
        panel.configure(image=img2)
        panel.image = (img2)

    settings.bind("<ButtonRelease-1>", update)

    global progressbar
    # a convenient progressbar, not indicative of the actual relative time left for the program to finish
    progressbar= Progressbar(frame4,orient="horizontal",length=300,mode="determinate")
    progressbar.grid(row=10, column=0)
    maxValue=100
    currentValue=0
    progressbar["value"]=currentValue
    progressbar["maximum"]=maxValue

    # keeping the window alive
    settings.mainloop()

