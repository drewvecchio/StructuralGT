"""GT_Params: Calculates and collates graph theory indices from
an input graph. Utilizes the NetworkX and GraphRicciCurvature
libraries of algorithms.

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
import networkx as nx
import pandas as pd
import numpy as np
import settings
from statistics import mean
from time import sleep
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering, average_clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


def run_GT_calcs(G, Do_kdist, Do_dia, Do_BCdist, Do_CCdist, Do_ECdist, Do_GD, Do_Eff, Do_clust, \
                 Do_ANC, Do_Ast, Do_WI, multigraph, Do_Ricci):

    # getting nodes and edges and defining variables for later use
    klist = [0]
    Tlist = [0]
    BCdist = [0]
    CCdist = [0]
    ECdist = [0]
    data_dict = {"x":[], "y":[]}
    orc = {}
    frc = {}
    orc_list = []
    frc_list = []
    if multigraph:
        Do_BCdist = 0
        Do_ECdist = 0
        Do_clust = 0

    nnum = int(nx.number_of_nodes(G))
    enum = int(nx.number_of_edges(G))

    if Do_ANC | Do_dia:
        connected_graph = nx.is_connected(G)




    data_dict["x"].append("Number of nodes")
    data_dict["y"].append(nnum)

    data_dict["x"].append("Number of edges")
    data_dict["y"].append(enum)


    settings.progress(35)

    # calculating parameters as requested

    # creating degree histogram
    if(Do_kdist == 1):
        settings.update_label("Calculating degree...")
        klist1 = nx.degree(G)
        ksum = 0
        klist = np.zeros(len(klist1))
        for j in range(len(klist1)):
            ksum += klist1[j]
            klist[j] = klist1[j]
        k = ksum/len(klist1)
        k = round(k, 5)
        data_dict["x"].append("Average degree")
        data_dict["y"].append(k)


    settings.progress(40)

    # calculating network diameter
    if(Do_dia ==1):
        settings.update_label("Calculating diameter...")
        if connected_graph:
            dia = int(diameter(G))
        else:
            dia = 'NaN'
        data_dict["x"].append("Network diameter")
        data_dict["y"].append(dia)


    settings.progress(45)

    # calculating graph density
    if(Do_GD == 1):
        settings.update_label("Calculating density...")
        GD = nx.density(G)
        GD = round(GD, 5)
        data_dict["x"].append("Graph density")
        data_dict["y"].append(GD)


    settings.progress(50)

    # calculating global efficiency
    if (Do_Eff == 1):
        settings.update_label("Calculating efficiency...")
        Eff = global_efficiency(G)
        Eff = round(Eff, 5)
        data_dict["x"].append("Global efficiency")
        data_dict["y"].append(Eff)


    if (Do_WI == 1):
        settings.update_label("Calculating WI...")
        WI = wiener_index(G)
        WI = round(WI, 1)
        data_dict["x"].append("Wiener Index")
        data_dict["y"].append(WI)



    settings.progress(55)

    # calculating clustering coefficients
    if(Do_clust == 1):
        settings.update_label("Calculating clustering...")
        sleep(5)
        Tlist1 = clustering(G)
        Tlist = np.zeros(len(Tlist1))
        for j in range(len(Tlist1)):
            Tlist[j] = Tlist1[j]
        clust = average_clustering(G)
        clust = round(clust, 5)
        data_dict["x"].append("Average clustering coefficient")
        data_dict["y"].append(clust)


    settings.progress(60)

    # calculating average nodal connectivity

    if (Do_ANC == 1):
        settings.update_label("Calculating connectivity...")
        if connected_graph:
            ANC = average_node_connectivity(G)
            ANC = round(ANC, 5)
        else:
            ANC = 'NaN'
        data_dict["x"].append("Average nodal connectivity")
        data_dict["y"].append(ANC)


    settings.progress(65)

    # calculating assortativity coefficient
    if (Do_Ast == 1):
        settings.update_label("Calculating assortativity...")
        Ast = degree_assortativity_coefficient(G)
        Ast = round(Ast, 5)
        data_dict["x"].append("Assortativity coefficient")
        data_dict["y"].append(Ast)


    settings.progress(70)

    # calculating betweenness centrality histogram
    if (Do_BCdist == 1):
        settings.update_label("Calculating betweenness...")
        BCdist1 = betweenness_centrality(G)
        Bsum = 0
        BCdist = np.zeros(len(BCdist1))
        for j in range(len(BCdist1)):
            Bsum += BCdist1[j]
            BCdist[j] = BCdist1[j]
        Bcent = Bsum / len(BCdist1)
        Bcent = round(Bcent, 5)
        data_dict["x"].append("Average betweenness centrality")
        data_dict["y"].append(Bcent)


    settings.progress(75)

    # calculating closeness centrality
    if(Do_CCdist == 1):
        settings.update_label("Calculating closeness...")
        CCdist1 = closeness_centrality(G)
        Csum = 0
        CCdist = np.zeros(len(CCdist1))
        for j in range(len(CCdist1)):
            Csum += CCdist1[j]
            CCdist[j] = CCdist1[j]
        Ccent = Csum / len(CCdist1)
        Ccent = round(Ccent, 5)
        data_dict["x"].append("Average closeness centrality")
        data_dict["y"].append(Ccent)


    settings.progress(80)

    # calculating eigenvector centrality
    if(Do_ECdist == 1):
        settings.update_label("Calculating eigenvector...")
        try:
            ECdist1 = eigenvector_centrality(G, max_iter=100)
        except:
            ECdist1 = eigenvector_centrality(G, max_iter=10000)
        Esum = 0
        ECdist = np.zeros(len(ECdist1))
        for j in range(len(ECdist1)):
            Esum += ECdist1[j]
            ECdist[j] = ECdist1[j]
        Ecent = Esum / len(ECdist1)
        Ecent = round(Ecent, 5)
        data_dict["x"].append("Average eigenvector centrality")
        data_dict["y"].append(Ecent)

    if(Do_Ricci == 1):
        settings.update_label("Calculating ricci curvature...")

        orc = OllivierRicci(G, weight=None, method="OTD")
        orc.compute_ricci_curvature()

        frc = FormanRicci(G, weight=None)
        frc.compute_ricci_curvature()

        if multigraph:
            for (s, e) in G.edges():
                for k in range(int(len(G[s][e]))):
                    orc_list.append(orc.G[s][e][k]["ricciCurvature"])
                    frc_list.append(frc.G[s][e][k]["ricciCurvature"])
        else:
            for (s, e) in G.edges():
                orc_list.append(orc.G[s][e]["ricciCurvature"])
                frc_list.append(frc.G[s][e]["formanCurvature"])

        av_orc = round(mean(orc_list),5)
        av_frc = round(mean(frc_list),2)
        data_dict["x"].append("Average Ollivier-Ricci Curvature")
        data_dict["y"].append(av_orc)
        data_dict["x"].append("Average Forman-Ricci Curvature")
        data_dict["y"].append(av_frc)



    data = pd.DataFrame(data_dict)

    return data, klist, Tlist, BCdist, CCdist, ECdist, orc, orc_list, frc, frc_list

def run_weighted_GT_calcs(G, Do_kdist, Do_BCdist, Do_CCdist, Do_ECdist, Do_ANC, Do_Ast, Do_WI, Do_Ricci, multigraph):

    settings.update_label("Performing weighted analysis...")

    # includes weight in the calculations
    klist = [0]
    BCdist = [0]
    CCdist = [0]
    ECdist = [0]
    w_orc = {}
    w_frc = {}
    w_orc_list = []
    w_frc_list = []
    if multigraph:
        Do_BCdist = 0
        Do_ECdist = 0
        Do_ANC = 0

    wdata_dict = {"x": [], "y": []}

    if Do_ANC:
        connected_graph = nx.is_connected(G)




    if(Do_kdist == 1):
        klist1 = nx.degree(G, weight='weight')
        ksum = 0
        klist = np.zeros(len(klist1))
        for j in range(len(klist1)):
            ksum += klist1[j]
            klist[j] = klist1[j]
        k = ksum/len(klist1)
        k = round(k, 5)
        wdata_dict["x"].append("Weighted average degree")
        wdata_dict["y"].append(k)


    if (Do_WI == 1):
        WI = wiener_index(G, weight='length')
        WI = round(WI, 1)
        wdata_dict["x"].append("Length-weighted Wiener Index")
        wdata_dict["y"].append(WI)


    if (Do_ANC == 1):
        if connected_graph:
            max_flow = float(0)
            p = periphery(G)
            q = len(p) - 1
            for s in range(0, q - 1):
                for t in range(s + 1, q):
                    flow_value = maximum_flow(G, p[s], p[t], capacity='weight')[0]
                    if (flow_value > max_flow):
                        max_flow = flow_value
            max_flow = round(max_flow, 5)
        else:
            max_flow = 'NaN'
        wdata_dict["x"].append("Max flow between periphery")
        wdata_dict["y"].append(max_flow)


    if (Do_Ast == 1):
        Ast = degree_assortativity_coefficient(G, weight = 'pixel width')
        Ast = round(Ast, 5)
        wdata_dict["x"].append("Weighted assortativity coefficient")
        wdata_dict["y"].append(Ast)


    if(Do_BCdist == 1):
        BCdist1 = betweenness_centrality(G, weight='weight')
        Bsum = 0
        BCdist = np.zeros(len(BCdist1))
        for j in range(len(BCdist1)):
            Bsum += BCdist1[j]
            BCdist[j] = BCdist1[j]
        Bcent = Bsum / len(BCdist1)
        Bcent = round(Bcent, 5)
        wdata_dict["x"].append("Width-weighted average betweenness centrality")
        wdata_dict["y"].append(Bcent)


    if(Do_CCdist == 1):
        CCdist1 = closeness_centrality(G, distance='length')
        Csum = 0
        CCdist = np.zeros(len(CCdist1))
        for j in range(len(CCdist1)):
            Csum += CCdist1[j]
            CCdist[j] = CCdist1[j]
        Ccent = Csum / len(CCdist1)
        Ccent = round(Ccent, 5)
        wdata_dict["x"].append("Length-weighted average closeness centrality")
        wdata_dict["y"].append(Ccent)


    if (Do_ECdist == 1):
        try:
            ECdist1 = eigenvector_centrality(G, max_iter=100, weight='weight')
        except:
            ECdist1 = eigenvector_centrality(G, max_iter=10000, weight='weight')
        Esum = 0
        ECdist = np.zeros(len(ECdist1))
        for j in range(len(ECdist1)):
            Esum += ECdist1[j]
            ECdist[j] = ECdist1[j]
        Ecent = Esum / len(ECdist1)
        Ecent = round(Ecent, 5)
        wdata_dict["x"].append("Width-weighted average eigenvector centrality")
        wdata_dict["y"].append(Ecent)

    if (Do_Ricci == 1):
        try:
            w_orc = OllivierRicci(G, weight='length', method="OTD")
            w_orc.compute_ricci_curvature()
        except:
            w_orc = OllivierRicci(G, weight='length', method="ATD")
            w_orc.compute_ricci_curvature()

        w_frc = FormanRicci(G, weight='length')
        w_frc.compute_ricci_curvature()

        if multigraph:
            for (s, e) in G.edges():
                for k in range(int(len(G[s][e]))):
                    w_orc_list.append(w_orc.G[s][e][k]["ricciCurvature"])
                    w_frc_list.append(w_frc.G[s][e][k]["ricciCurvature"])
        else:
            for (s, e) in G.edges():
                w_orc_list.append(w_orc.G[s][e]["ricciCurvature"])
                w_frc_list.append(w_frc.G[s][e]["formanCurvature"])

        av_w_orc = round(mean(w_orc_list), 5)
        av_w_frc = round(mean(w_frc_list), 2)
        wdata_dict["x"].append("Average Weighted Ollivier-Ricci Curvature")
        wdata_dict["y"].append(av_w_orc)
        wdata_dict["x"].append("Average Weighted Forman-Ricci Curvature")
        wdata_dict["y"].append(av_w_frc)


    wdata = pd.DataFrame(wdata_dict)

    return wdata, klist, BCdist, CCdist, ECdist, w_orc, w_orc_list, w_frc, w_frc_list