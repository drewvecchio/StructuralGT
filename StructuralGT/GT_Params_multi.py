"""GT_Params_multi: Calculates and collates graph theory indices from
an input graph. Utilizes the NetworkX and GraphRicciCurvature
libraries of algorithms.  Operates with the multi-image analysis
feature of StructuralGT.

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
from statistics import mean
import multi_image_settings
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering, average_clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index


def run_GT_calcs(G, just_data, Do_kdist, Do_dia, Do_BCdist, Do_CCdist, Do_ECdist, Do_GD, Do_Eff, \
                               Do_clust, Do_ANC, Do_Ast, Do_WI, multigraph):

    # getting nodes and edges and defining variables for later use
    klist = [0]
    Tlist = [0]
    BCdist = [0]
    CCdist = [0]
    ECdist = [0]
    if multigraph:
        Do_BCdist = 0
        Do_ECdist = 0
        Do_clust = 0

    data_dict = {"x":[], "y":[]}

    nnum = int(nx.number_of_nodes(G))
    enum = int(nx.number_of_edges(G))

    if Do_ANC | Do_dia:
        connected_graph = nx.is_connected(G)

    # making a dictionary for the parameters and results
    just_data.append(nnum)
    data_dict["x"].append("Number of nodes")
    data_dict["y"].append(nnum)
    just_data.append(enum)
    data_dict["x"].append("Number of edges")
    data_dict["y"].append(enum)
    multi_image_settings.progress(35)

    # calculating parameters as requested

    # creating degree histogram
    if(Do_kdist == 1):
        klist1 = nx.degree(G)
        ksum = 0
        klist = np.zeros(len(klist1))
        for j in range(len(klist1)):
            ksum = ksum + klist1[j]
            klist[j] = klist1[j]
        k = ksum/len(klist1)
        k = round(k, 5)
        just_data.append(k)
        data_dict["x"].append("Average degree")
        data_dict["y"].append(k)

    multi_image_settings.progress(40)

    # calculating network diameter
    if(Do_dia ==1):
        if connected_graph:
            dia = int(diameter(G))
        else:
            dia = 'NaN'
        just_data.append(dia)
        data_dict["x"].append("Network Diameter")
        data_dict["y"].append(dia)

    multi_image_settings.progress(45)

    # calculating graph density
    if(Do_GD == 1):
        GD = nx.density(G)
        GD = round(GD, 5)
        just_data.append(GD)
        data_dict["x"].append("Graph density")
        data_dict["y"].append(GD)

    multi_image_settings.progress(50)

    # calculating global efficiency
    if (Do_Eff == 1):
        Eff = global_efficiency(G)
        Eff = round(Eff, 5)
        just_data.append(Eff)
        data_dict["x"].append("Global Efficiency")
        data_dict["y"].append(Eff)

    multi_image_settings.progress(55)

    if (Do_WI == 1):
        WI = wiener_index(G)
        WI = round(WI, 1)
        just_data.append(WI)
        data_dict["x"].append("Wiener Index")
        data_dict["y"].append(WI)

    multi_image_settings.progress(60)

    # calculating clustering coefficients
    if(Do_clust == 1):
        Tlist1 = clustering(G)
        Tlist = np.zeros(len(Tlist1))
        for j in range(len(Tlist1)):
            Tlist[j] = Tlist1[j]
        clust = average_clustering(G)
        clust = round(clust, 5)
        just_data.append(clust)
        data_dict["x"].append("Average clustering coefficient")
        data_dict["y"].append(clust)

    # calculating average nodal connectivity
    if (Do_ANC == 1):
        if connected_graph:
            ANC = average_node_connectivity(G)
            ANC = round(ANC, 5)
        else:
            ANC = 'NaN'
        just_data.append(ANC)
        data_dict["x"].append("Average nodal connectivity")
        data_dict["y"].append(ANC)

    multi_image_settings.progress(65)

    # calculating assortativity coefficient
    if (Do_Ast == 1):
        Ast = degree_assortativity_coefficient(G)
        Ast = round(Ast, 5)
        just_data.append(Ast)
        data_dict["x"].append("Assortativity Coefficient")
        data_dict["y"].append(Ast)

    multi_image_settings.progress(70)

    # calculating betweenness centrality histogram
    if(Do_BCdist == 1):
        BCdist1 = betweenness_centrality(G)
        Bsum = 0
        BCdist = np.zeros(len(BCdist1))
        for j in range(len(BCdist1)):
            Bsum += BCdist1[j]
            BCdist[j] = BCdist1[j]
        Bcent = Bsum / len(BCdist1)
        Bcent = round(Bcent, 5)
        just_data.append(Bcent)
        data_dict["x"].append("Average betweenness centrality")
        data_dict["y"].append(Bcent)
    multi_image_settings.progress(75)

    # calculating closeness centrality
    if(Do_CCdist == 1):
        CCdist1 = closeness_centrality(G)
        Csum = 0
        CCdist = np.zeros(len(CCdist1))
        for j in range(len(CCdist1)):
            Csum += CCdist1[j]
            CCdist[j] = CCdist1[j]
        Ccent = Csum / len(CCdist1)
        Ccent = round(Ccent, 5)
        just_data.append(Ccent)
        data_dict["x"].append("Average closeness centrality")
        data_dict["y"].append(Ccent)

        multi_image_settings.progress(80)

        # calculating eigenvector centrality
        if (Do_ECdist == 1):
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
            Ecent = round(Ccent, 5)
            just_data.append(Ecent)
            data_dict["x"].append("Average eigenvector centrality")
            data_dict["y"].append(Ecent)


    data = pd.DataFrame(data_dict)

    return data, just_data, klist, Tlist, BCdist, CCdist, ECdist

def run_weighted_GT_calcs(G, just_data, Do_kdist, Do_BCdist, Do_CCdist, Do_ECdist, Do_ANC, Do_Ast, Do_WI, multigraph):

    # includes weight in the calculations
    klist = [0]
    BCdist = [0]
    CCdist = [0]
    ECdist = [0]
    if multigraph:
        Do_BCdist = 0
        Do_ECdist = 0
        Do_ANC = 0


    if Do_ANC:
        connected_graph = nx.is_connected(G)

    wdata_dict = {"x": [], "y": []}

    if(Do_kdist == 1):
        klist1 = nx.degree(G, weight='weight')
        ksum = 0
        klist = np.zeros(len(klist1))
        for j in range(len(klist1)):
            ksum = ksum + klist1[j]
            klist[j] = klist1[j]
        k = ksum/len(klist1)
        k = round(k, 5)
        just_data.append(k)
        wdata_dict["x"].append("Weighted average degree")
        wdata_dict["y"].append(k)

    if (Do_WI == 1):
        WI = wiener_index(G, weight='length')
        WI = round(WI, 1)
        just_data.append(WI)
        wdata_dict["x"].append("Length-weighted Wiener Index")
        wdata_dict["y"].append(WI)

    if (Do_ANC == 1):
        if connected_graph:
            max_flow = float(0)
            p = periphery(G)
            q = len(p) - 1
            for s in range(0,q-1):
                for t in range(s+1,q):
                    flow_value = maximum_flow(G, p[s], p[t], capacity='weight')[0]
                    if(flow_value > max_flow):
                        max_flow = flow_value
            max_flow = round(max_flow, 5)
        else:
            max_flow = 'NaN'
        just_data.append(max_flow)
        wdata_dict["x"].append("Max flow between periphery")
        wdata_dict["y"].append(max_flow)

    if (Do_Ast == 1):
        Ast = degree_assortativity_coefficient(G, weight='pixel width')
        Ast = round(Ast, 5)
        just_data.append(Ast)
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
        just_data.append(Bcent)
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
        just_data.append(Ccent)
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
        just_data.append(Ecent)
        wdata_dict["x"].append("Width-weighted average eigenvector centrality")
        wdata_dict["y"].append(Ecent)


    wdata = pd.DataFrame(wdata_dict)

    return wdata, just_data, klist, BCdist, CCdist, ECdist