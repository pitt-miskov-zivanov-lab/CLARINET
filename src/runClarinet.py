"""
@author: Yasmine Ahmed
"""

import pandas as pd
import re
import numpy as np
import networkx as nx
import math
import pickle
from community import community_louvain
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import seaborn as sns
import argparse
import os
import time
import logging
from datetime import datetime


def create_eclg(interaction_filename,model_dict):
    """
    This function creates the ECLG where a node is an event (e.g., biochemical interaction) and there
    is an edge between two nodes (two events) if they happen to occur in the same paper. The reading output
    file is in INDRA format.

    Parameters
    ----------
    interaction_filename : str
               The path of the reading output file (extracted events)
    model_dict : dict
               Dictionary that holds critical information of each baseline model element

    Returns
    -------
    G : Graph
               Event CoLlaboration Graph
    """
    papers=list() #list of Paper IDs within the RO file
    Mean_interactions_per_paper = 0

    with open(interaction_filename) as interaction_file:
        for line in interaction_file:
            if 'regulator_name' in line:
                continue
            line = line.strip()
            interaction = re.split(',',line)
            #FIXME: hard coded here, the 26th column of ReadingOutput file has to be 'PaperID'
            papers.append(interaction[25])

    for p in np.unique(papers):
        Mean_interactions_per_paper+=papers.count(p)

    Paper_ID_unique=np.unique(papers)
    logging.info('Number of unique paper IDs: {}'.format(str(len(Paper_ID_unique))))
    logging.info('Average interactions per paper: {}'.format(str(Mean_interactions_per_paper/len(Paper_ID_unique))))


    G = nx.Graph() #this will contain ECLG nodes and edges
    IDs = len(Paper_ID_unique)
    curr_map=dict()
    #FIXME: hard coded here, the 9th, 17th columns of ReadingOutput file have to be 'regulated_name' and 'interaction'
    regulated_col = 8
    interaction_col = 16

    #Create ECLG from RO file

    for jj in range(0, IDs):
        my_list = []
        with open(interaction_filename) as interaction_file:
            for line in interaction_file:
                if 'regulator_name' in line:
                    continue
                line = line.strip()
                interaction = re.split(',',line)
                if Paper_ID_unique[jj] in interaction[25]:
                    inter1P=''
                    if (interaction[0] != '' and interaction[8] != ''):
                        elem11 = getVariableName(model_dict,curr_map,interaction[0:regulated_col])
                        elem22 = getVariableName(model_dict,curr_map,interaction[regulated_col:interaction_col])
                        if interaction[16] == 'increases':
                            inter1P = elem11+'->'+elem22+'->+'#+'->'+we
                        else:
                            inter1P = elem11+'->'+elem22+'->-'#+'->'+we
                        my_list.append(inter1P)

        for n in range(0, len(my_list)-1):
            for nn in range(n+1, len(my_list)):
                if G.has_edge(my_list[n], my_list[nn]) or G.has_edge(my_list[nn], my_list[n]):
                    G[my_list[n]][my_list[nn]]['weight'] += 1
                    if my_list[n] == my_list[nn]:
                        G.remove_edge(my_list[n],my_list[nn])
                else:
                    G.add_edge(my_list[n], my_list[nn], weight=1)
                    if my_list[n] == my_list[nn]:#self loops
                        G.remove_edge(my_list[n],my_list[nn])

    return G


def create_eclg_el(interaction_filename,model_dict):

    """
    This function creates the ECLG where a node is an event (e.g., biochemical interaction) and there
    is an edge between two nodes (two events) if they happen to occur in the same paper. The reading output
    file header must contain the following fields: Element Name, Element Type, Element Identifier,
    PosReg Name/Type/ID, NegReg Name/Type/ID and Paper ID.

    Parameters
    ----------
    interaction_filename : str
               The path of the reading output file (extracted events)
    model_dict : dict
               Dictionary that holds critical information of each baseline model element

    Returns
    -------
    G : Graph
               Event CoLlaboration Graph
    """
    df = pd.read_excel(interaction_filename)
    papers=list()

    #remove date entries
    date_indices = list()
    for idx,ele_name in enumerate(df['Element Name']):
        if isinstance(ele_name, datetime) or isinstance(df['PosReg Name'].iloc[idx], datetime) or isinstance(df['NegReg Name'].iloc[idx], datetime):
            date_indices.append(int(idx))
    for iii in date_indices:
        df = df.drop([iii])

    Element_name=df['Element Name']
    Element_type=df['Element Type']
    Element_ID=df['Element Identifier']
    PosReg_Name=df['PosReg Name']
    PosReg_type=df['PosReg Type']
    PosReg_ID=df['PosReg ID']
    NegReg_Name=df['NegReg Name']
    NegReg_type=df['NegReg Type']
    NegReg_ID=df['NegReg ID']
    Paper_ID=df['Paper ID']

    for p in Paper_ID:
        papers.append(p)

    Paper_ID_unique=np.unique(papers)
    print(len(Paper_ID_unique))
    Mean_interactions_per_paper =0
    for p in np.unique(papers):
        Mean_interactions_per_paper+=papers.count(p)

    G = nx.Graph()
    IDs=len(Paper_ID_unique)

    curr_map=dict()
    for jj in range(0,IDs):
        my_list= []
        for idx,ele_name in enumerate(Element_name):
            if Paper_ID_unique[jj] in Paper_ID.iloc[idx]:
                inter1P=''
                elem=getVariableName(model_dict,curr_map,[(ele_name),'','',Element_ID.iloc[idx],'',Element_type.iloc[idx],'',''])#ele_name
                Pos=getVariableName(model_dict,curr_map,[str(PosReg_Name.iloc[idx]),'','',str(PosReg_ID.iloc[idx]),'',str(PosReg_type.iloc[idx]),'',''])#PosReg_Name[idx]
                Neg=getVariableName(model_dict,curr_map,[str(NegReg_Name.iloc[idx]),'','',str(NegReg_ID.iloc[idx]),'',str(NegReg_type.iloc[idx]),'',''])#NegReg_Name[idx]

                if str(Pos) != "nan_ext" and str(Pos) != "":
                    for posi in re.split(',',Pos):
                        if elem in model_dict and posi in model_dict:
                            if posi in model_dict[elem]['regulators']:
                                we=str(3)
                            elif posi not in model_dict[elem]['regulators'] and elem not in model_dict[posi]['regulators'] :
                                we=str(2)
                            elif elem in model_dict[posi]['regulators']:
                                we=str(0)
                        elif posi in model_dict and elem not in model_dict:
                            we=str(1)
                        elif elem in model_dict and posi not in model_dict:
                            we=str(1)
                        else:
                            we=str(0)
                        inter1P=posi+'->'+elem+'->+'#+'->'+we
                        my_list.append(inter1P)

                if str(Neg) != "nan_ext" and str(Neg) != "":#type(Neg) is not float:
                    for nega in re.split(',',Neg):
                        if elem in model_dict and nega in model_dict:
                            if nega in model_dict[elem]['regulators']:
                                we=str(3)
                            elif nega not in model_dict[elem]['regulators'] and elem not in model_dict[nega]['regulators'] :
                                we=str(2)
                            elif elem in model_dict[nega]['regulators']:
                                we=str(0)
                        elif nega in model_dict and elem not in model_dict:
                            we=str(1)
                        elif elem in model_dict and nega not in model_dict:
                            we=str(1)
                        else:
                            we=str(0)
                        inter1P=nega+'->'+elem+'->-'#+'->'+we
                        my_list.append(inter1P)


        my_list = list(set(my_list))
        for n in range(0, len(my_list)-1):
            for nn in range(n+1, len(my_list)):

                if G.has_edge(my_list[n], my_list[nn]) or G.has_edge(my_list[nn], my_list[n]):
                    G[my_list[n]][my_list[nn]]['weight']+= 1#*(weight1+weight2) #+= 1/IDs
                    if my_list[n]==my_list[nn]:
                        G.remove_edge(my_list[n],my_list[nn])
                else:
                    G.add_edge(my_list[n], my_list[nn], weight=1)#*(weight1+weight2))#=1/IDs)
                    if my_list[n]==my_list[nn]:
                        G.remove_edge(my_list[n],my_list[nn])
    return G

# Individual assessment (IA)
def node_weighting(G, freqTh, path):
    """
    This function assigns weights to graph nodes using frequency class, and returns a new ECLG after removing less frequent nodes.
    In the meantime, ECLG nodes and their freqClass level, ECLG edges before and after the removal will be saved to specified directory.

    Parameters
    ----------
    G : undirected graph
        Event CoLlaboration Graph
    freqTh : int
        Frequency class threshold value, events (nodes) having FC greater than this value will be removed
    path : str
        The output directory where the genereted files will be saved

    Returns
    -------
    G : undirected graph
        a new ECLG after the removal of less frequent nodes

    """
    # Assign weights to nodes using frequency class concept

    ebunch = list() #less frequent nodes that will be removed
    nodesDegree = list()
    fill = os.path.join(path, "freqClass")
    output_stream = open(fill, 'w')

    for g in G.nodes:
        nodesDegree.append(G.degree[g])
        if G.degree[g] == 0:
            continue

    freqMostCommonNode = max(nodesDegree)
    #print("Frequency of most frequent node before: "+str(freqMostCommonNode))

    #Mapping node names to frequency class
    for g in G.nodes:
        if G.degree[g] == 0: continue
        freqCLASS = math.floor(0.5-np.log2(G.degree[g]/freqMostCommonNode))
        newName = g + '->'+str(freqCLASS)
        mapping = {g: newName}
        G = nx.relabel_nodes(G, mapping)

    nx.write_edgelist(G, os.path.join(path,"ECLGbefore.txt")) #ECLG nodes and edges before the removal of less frequent nodes

    for g in G.nodes:
        if G.degree[g] == 0:
            continue
        freqCLASS = math.floor(0.5-np.log2(G.degree[g]/freqMostCommonNode))
        if freqCLASS > int(freqTh): # Frequency class threshold which may vary from case study to another (set it =2 for T cell case study)
            ebunch.append(g)
        output_stream.write(g+' '+str(G.degree[g])+' '+str(freqCLASS)+'\n')
        nodesDegree.append(G.degree[g])
    G.remove_nodes_from(ebunch)

    nx.write_edgelist(G, os.path.join(path,"ECLGafter.txt")) #ECLG nodes and edges after the removal of less frequent nodes

    return G

# Pair assessment (IA)
def edge_weighting(G, path, weightMethod):
    """
    This function assigns weights to graph edges using frequency class (FC) or inverse frequency formula (IF), and returns a weighted ECLG.
    In the meantime, ECLG edges and their weights will be saved to specified directory.

    Parameters
    ----------
    G : undirected graph
        Event CoLlaboration Graph
    path : str
        The output directory where the genereted files will be saved
    weightMethod : str
        'FC' or 'IF'

    Returns
    -------
    G : undirected graph
        ECLG after assigning weights to edges
    """
    # Assign weights to edges using frequency class concept (fc) or inverse frequency concept (iif)

    fill = os.path.join(path, "LSS") #This file will be input to get_cluster_info()
    output_stream = open(fill, 'w')
    N = G.number_of_nodes()
    weights = list()

    for n1, n2, w in G.edges.data():
        weights.append(G[n1][n2]['weight'])

    maxWeight = int(max(weights))

    for n1, n2, w in G.edges.data():
        w1 = G[n1][n2]['weight']
        if weightMethod == 'FC':
            dd = math.floor(0.5-np.log2(w1/(maxWeight))) #fc
            G[n1][n2]['weight']= dd
            output_stream.write(n1+' '+n2+' '+str(dd)+'\n')
        elif weightMethod == 'IF':
            p1 = np.log(N/G.degree[n1])
            p2 = np.log(N/G.degree[n2])
            dd = w1*(p1 + p2)
            if dd < 0:
                dd = 0
            G[n1][n2]['weight']= dd
            output_stream.write(n1+' '+n2+' '+str(dd)+'\n')
    return G

def clustering(G, path):
    """
    This function implements three things:
    (1) clusters the ECLG using the community detection algorithm by Blondel et al., and returns a pickle file containing grouped
    (clustered) extensions, specified as nested lists. Each group starts with an integer, followed by interactions specified as
    [regulator element, regulated element, Interaction type: Activation (+) or Inhibition (-)];
    (2) displays the cluster result;
    (3) saves each cluster in a separate file, in both uninterpreted (under GeneratedClusters/) and interpreted manners (under InterpretedClusters/).

    Parameters
    ----------
    G : undirected graph
        Event CoLlaboration Graph
    path : str
        The output directory where the genereted files will be saved

    """
    #Clustering

    partition = community_louvain.best_partition(G)
    centers = {}
    communities = {}
    G_main_com = G.copy()
    min_nb = 2
    com_edges = list()
    group_num = 1
    for com in set(partition.values()) :
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        if len(list_nodes) < min_nb:
            G_main_com.remove_nodes_from(list_nodes)
        else:
            # Get center
            H = G_main_com.subgraph(list_nodes)
            d_c = nx.degree_centrality(H)
            center = max(d_c, key=d_c.get)
            centers[center] = com
            communities[com] = center
            # Print community
            logging.info('Community centered at "{}"(community label is {}) has {} interactions:\n{}\n'.format(center, com, len(list_nodes),list_nodes))
            NODESS=list()
            for ii in range(0, len(list_nodes)-1):
                for jj in range(ii+1, len(list_nodes)):
                    if G.has_edge(list_nodes[ii],list_nodes[jj]):
                        wi=G[list_nodes[ii]][list_nodes[jj]]['weight']
                        temp=list()
                        temp.append(list_nodes[ii])
                        temp.append(list_nodes[jj])
                        temp.append(wi)
                        NODESS.append(temp)
            com_edges.append([group_num]+NODESS)
            group_num=group_num+1
    pickle.dump(com_edges, open(os.path.join(path, "grouped_ext"),'wb')) #all clusters in one pickle file

    # Display graph
    plt.figure(figsize=(13, 5))
    node_size = 30
    count = 0
    pos = nx.spring_layout(G_main_com)
    colors = dict(zip(communities.keys(), sns.color_palette('hls', len(communities.keys()))))
    for com in communities.keys():
        count = count + 1
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com and nodes not in communities.values()]
        nx.draw_networkx_nodes(G_main_com, pos, list_nodes, node_size = node_size, node_color = colors[com])
        nx.draw_networkx_nodes(G_main_com, pos, list([communities[com]]), node_size = node_size*5, node_color = colors[com])
    nx.draw_networkx_edges(G_main_com, pos, alpha=0.5)
    labels = {k: k for k,v in centers.items()}
    nx.draw_networkx_labels(G_main_com, pos, labels)
    plt.axis('off')
    plt.show()

    #Save each cluster in a separate file

    thePath = os.path.join(path, "GeneratedClusters") # Directory containing generated clusters
    if not os.path.exists(thePath):
        os.makedirs(thePath)
    clusterFile = os.path.join(thePath, 'GeneratedCluster')  # generated (uninterpreted) clusters

    for e in com_edges:
        cl = []
        fill=clusterFile+str(e[0])
        output_stream = open(fill, 'w')
        for ee in e[1:]:
            s0 = ee[0]
            s1 = ee[1]
            if ee[1] == ee[0]:continue
            if ee[0]+' '+ee[1]+' '+str(ee[2])+'\n' not in cl:
                output_stream.write(ee[0]+' '+ee[1]+' '+str(ee[2])+'\n')
                cl.append(ee[0]+' '+ee[1]+' '+str(ee[2])+'\n')
        output_stream.close()

    thePath = os.path.join(path, "InterpretedClusters")
    if not os.path.exists(thePath):
        os.makedirs(thePath)
    clusterFile = os.path.join(thePath, 'InterpretedCluster') #interpreted clusters

    for e in com_edges:
        cl = []
        fill = clusterFile+str(e[0])
        output_stream = open(fill, 'w')
        for ee in e[1:]:
            s0 = ee[0].split('->')
            s1 = ee[1].split('->')
            if ee[1] == ee[0]:continue
            if s0[0]+' '+s0[1]+' '+s0[2]+' '+s0[3]+'\n' not in cl and s1[0]+' '+s1[1]+' '+s1[2]+' '+s1[3]+'\n' not in cl:
                output_stream.write(s0[0]+' '+s0[1]+' '+s0[2]+' '+s0[3]+'\n')
                output_stream.write(s1[0]+' '+s1[1]+' '+s1[2]+' '+s1[3]+'\n')
                cl.append(s0[0]+' '+s0[1]+' '+s0[2]+' '+s0[3]+'\n')
                cl.append(s1[0]+' '+s1[1]+' '+s1[2]+' '+s1[3]+'\n')
        output_stream.close()

def get_cluster_info(generated_clu_path, LSS_file, output_path):
    """
    This function returns some basic information about each of these generated clusters as a DataFrame object, as well as saves it as .csv file
    Information includes Cluster_index, Nodes, Edges, Density, AvgPathLength, Coeff, LSS, NodesX, EdgesX, DensityX, AvgPathLength, CoeffX, FreqClass, node_perc.

    Parameters
    ----------
    generated_clu_path : str
        The directory that contains the genereted clusters
    LSS_file : str
        The path of LSS_file, containing ECLG edges and their weights, generated in edge_weighting()
    output_path : str
        The output directory where ClusterInfoFile.csv will be saved

    Returns
    -------
    cluster_df : pandas.DataFrame()
        DataFrame that contains information for each generated cluster
    """

    entries = os.listdir(generated_clu_path) #directory of generated clusters
    NewFile = os.path.join(output_path, "ClusterInfoFile.csv") #output file that contains clusters'info
    output_stream = open(NewFile, "w")
    output_stream.write("Cluster_index, Nodes, Edges, Density, AvgPathLength, Coeff, LSS, NodesX, EdgesX, DensityX, AvgPathLength, CoeffX, FreqClass, node_perc"+'\n')
    cluster_df = pd.DataFrame(columns=['Cluster_index', 'Nodes', 'Edges', 'Density', 'AvgPathLength', 'Coeff', 'LSS', 'NodesX', 'EdgesX', 'DensityX', 'AvgPathLength', 'CoeffX', 'FreqClass', 'node_perc'])
    frequencyCLasses = dict()

    with open(LSS_file) as f: #This file that has been generated from runClarinet, it contains weighted edges of the ECLG, its name is LSSfc
        content = f.readlines()
        for c in content:
            x = c.strip()
            xx = x.split(" ")
            theKey = xx[0]+" "+xx[1]
            frequencyCLasses[theKey]=xx[2] #edge weights either FC or IF

    for entry in entries:
        Cl = ''

        for e in entry:
            if e.isdigit(): Cl += str(e)
        if not entry.startswith("GeneratedCluster"): continue

        G = nx.Graph() #generated cluster
        G1 = nx.Graph() #interpreted cluster

        weights = []

        with open(os.path.join(generated_clu_path, entry)) as f:
            content = f.readlines()
            freqCount = 0

            for c in content:
                x = c.strip()
                xx = x.split(" ")
                G.add_edge(xx[0],xx[1])
                G[xx[0]][xx[1]]['weight']=xx[2]
                weights.append(float(xx[2]))
                Int1 = xx[0].split('->')
                Int2 = xx[1].split('->')
                G1.add_edge(Int1[0],Int1[1])
                G1.add_edge(Int2[0],Int2[1])
                G1[Int1[0]][Int1[1]]['weight']=Int1[3] #Int1[2] will be +/- based on type of interaction
                G1[Int2[0]][Int2[1]]['weight']=Int2[3]

            num_of_nodes = G1.number_of_nodes()
            nod = 0 # increment this variable whenever you find an elemnt that is in the baseline model

            for no in G1.nodes():
                nn1 = no.split('_')
                if len(nn1)>=2:
                    if nn1[-1]!="ext":
                        nod += 1
                elif len(nn1)==1:
                    nod += 1

            # compute % node overlap
            node_perc = 100*(nod/num_of_nodes)
            for n in G.nodes():
                xX = n.split("->")
                if str(xX[3]) == '0':
                    freqCount += 1

            weightsss=list()

            for n1, n2, w in G.edges.data():
                for key in frequencyCLasses:
                    Inter = n1+" "+n2
                    if Inter ==key:
                        weightsss.append(float(frequencyCLasses[key]))

                pathlengths = []
                pathlengths1 = []

                for n in G.nodes():
                    spl = dict(nx.single_source_shortest_path_length(G, n))
                    for p in spl:
                        pathlengths.append(spl[p])
                Paths = sum(pathlengths) / len(pathlengths)

                for n in G1.nodes():
                    spl = dict(nx.single_source_shortest_path_length(G1, n))
                    for p in spl:
                        pathlengths1.append(spl[p])
                Paths1 = sum(pathlengths1) / len(pathlengths1)

            cluster_df.loc[int(Cl)]= [int(Cl), G.number_of_nodes(), G.number_of_edges(), nx.density(G), Paths, nx.average_clustering(G), sum(weights)/len(weights), G1.number_of_nodes(), G1.number_of_edges(), nx.density(G1), Paths1, nx.average_clustering(G1), freqCount, node_perc]
            output_stream.write(', '.join(str(x) for x in cluster_df.loc[int(Cl)]) + '\n')
    output_stream.close()
    return cluster_df

def merge_clusters(regulators, path, ReturnTh):
    """
    This function records indices of clusters to be merged based on the existence of return paths.
    It generates the grouped_ext_Merged pickle file that contains the merged clusters.

    Parameters
    ----------
    regulators : dict
		Contains baseline model elements and corresponding regulator elements
    path : str
		The path of the directory that contains the grouped_ext file
    ReturnTh : int
		A user-defined integer threshold for the number of return paths, beyond which clusters will be merged
    """
    # Merge clusters if there is one or more return paths

    G = nx.DiGraph()
    G = make_diGraph(regulators)
    com_edges = list()
    group_num = 1
    extensions = pickle.load(open(os.path.join(path,"grouped_ext"),'rb'))

    for ii in range(0,len(extensions)):
        for jj in range(ii+1,len(extensions)):
            count = 0
            cluster1 = extensions[ii]
            cluster2 = extensions[jj]
            G1 = nx.DiGraph()
            G2 = nx.DiGraph()

            for e in cluster1[1:]:
                ee=e[1].split('->')
                if ee[2] == '+':
                    G1.add_edge(ee[0], ee[1],weight=1)
                elif ee[2] == '-':
                    G1.add_edge(ee[0], ee[1],weight=0)

            for e in cluster2[1:]:
                ee=e[1].split('->')
                if ee[2] == '+':
                    G2.add_edge(ee[0], ee[1],weight=1)
                elif e[2] == '-':
                    G2.add_edge(ee[0], ee[1],weight=0)
            Gall = nx.compose(G1,G2)
            for g in G.edges:
                if g[0] in G1:
                    for ne in G.successors(g[1]):
                        if ne in G2:
                            for ne1 in G2.successors(ne):
                                if ne1 in G:
                                    count = count+1
                if g[0] in G1:
                    for ne in G.successors(g[0]):
                        if ne in G2:
                            for ne1 in G2.successors(ne):
                                if ne1 in G:
                                    count = count+1

            if count > int(ReturnTh): #set threshold for the number of return paths
                logging.info('Merge clusters NO.{} and NO.{}'.format(str(ii+1),str(jj+1)))
                Gall = nx.compose(G1,G2)
                NODESS = list()
                for (node1,node2,data) in Gall.edges(data=True):
                    temp=list()
                    temp.append(node1)
                    temp.append(node2)
                    if data['weight'] == 0:
                        temp.append('-')
                    elif data['weight'] == 1:
                        temp.append('+')
                    NODESS.append(temp)
                com_edges.append([group_num] + NODESS)
                group_num = group_num+1

    pickle.dump(com_edges, open(os.path.join(path, "grouped_ext_Merged"),'wb')) #Merged clusters

    return

#This and the following function are inherited from DySE framework
# define regex for valid characters in variable names
_VALID_CHARS = r'a-zA-Z0-9\@\_\/'
def get_model(model_file: str):
	"""
	This function reads the baseline model of BioRECIPES format and returns two useful dictionaries

	Parameters
	----------
	model_file : str
		The path of the baseline model file

	Returns
	-------
	model_dict : dict
		Dictionary that holds critical information of each baseline model element
	regulators : dict
		 Contains baseline model elements and corresponding regulator elements
	"""

	global _VALID_CHARS

	regulators = dict()
	model_dict = dict()

	# Load the input file containing elements and regulators
	df_model = pd.read_excel(model_file, na_values='NaN', keep_default_na = False)
	# check model format
	if df_model.columns[0].lower() == 'element attributes':
		df_model = df_model.reset_index()
		df_model = df_model.rename(columns=df_model.iloc[1]).drop([0,1]).set_index('#')

	input_col_name = [x.strip() for x in df_model.columns if ('element name' in x.lower())]
	input_col_ids = [x.strip() for x in df_model.columns if ('ids' in x.lower())]
	input_col_type = [x.strip() for x in df_model.columns if ('element type' in x.lower())]
	input_col_X = [x.strip() for x in df_model.columns if ('variable' in x.lower())]
	input_col_A = [x.strip() for x in df_model.columns if ('positive' in x.lower())]
	input_col_I = [x.strip() for x in df_model.columns if ('negative' in x.lower())]

	# set index to variable name column
	# remove empty variable names
	# append cols with the sets of regulators using .apply

	for curr_row in df_model.index:
		element_name = df_model.loc[curr_row,input_col_name[-1]].strip()
		ids = df_model.loc[curr_row,input_col_ids[0]].strip().upper().split(',')
		element_type = df_model.loc[curr_row,input_col_type[0]].strip()
		var_name = df_model.loc[curr_row,input_col_X[0]].strip()
		pos_regulators = df_model.loc[curr_row,input_col_A[0]].strip()
		neg_regulators = df_model.loc[curr_row,input_col_I[0]].strip()

		if var_name == '':
			continue

		curr = []

		if pos_regulators != '':
			curr += re.findall('['+_VALID_CHARS+']+',pos_regulators)

		if neg_regulators != '':
			curr += re.findall('['+_VALID_CHARS+']+',neg_regulators)

		# returning regulators separately for compatibility with runMarkovCluster
		regulators[var_name] = set(curr)
		model_dict[var_name] = {
			'name' : element_name,
			'ids' : ids,
			'type' : element_type,
			'regulators' : set(curr)}

	return model_dict, regulators

def getVariableName(model_dict, curr_map, ext_element_info):
	"""
	A utility function for create_eclg() and create_eclg_el(), which matches the element name from the extracted event to an element in the baseline model

	Parameters
	----------
	model_dict : dict
		Dictionary that holds critical information of each baseline model element
	curr_map: dict
		Temporary dictionary that contains already matched pairs
	ext_element_info: list
		List of information for certain element in the extracted event, starting with element name

	Returns
	-------
	match : str
		The most likely matched element name in model_dict, to the element represented by ext_element_info; Otherwise, return the extended element name suffix by "_ext"
	"""

	global _VALID_CHARS

	ext_element_name = ext_element_info[0]
	#print(ext_element_name)
	#print(ext_element_info)

	# Check for valid element name
	if ext_element_name=='':
		#logging.warn('Missing element name in extensions')
		return ''
	elif re.search('[^'+_VALID_CHARS+']+',ext_element_name):
		#logging.warn(('Skipping due to invalid characters in variable name: %s') % str(ext_element_name))
		return ''

	ext_element_id = ext_element_info[3]
	#print(ext_element_id)
	ext_element_type = ext_element_info[5]
	#print(ext_element_type)

	if ext_element_name in curr_map:
		return curr_map[ext_element_name]

	# from the location and type
	match = ext_element_name + '_ext'
	confidence = 0.0
	# Iterate all names in the dictionary and find the most likely match
	for key,value in model_dict.items():
		#print(ext_element_id)

		curr_conf = 0.0
		if str(ext_element_id).upper() in value['ids']:
			curr_conf = 1
		elif ext_element_name.upper().startswith(value['name'].upper()) \
			or value['name'].upper().startswith(ext_element_name.upper()):
			curr_conf = 0.8

		if curr_conf>0 and value['type'].lower().startswith(ext_element_type):
			curr_conf += 1

		if curr_conf > confidence:
			match = key
			confidence = curr_conf
			if curr_conf==2: break

	curr_map[ext_element_name] = match
	return match


def make_diGraph(mdldict):

    """
    A utility function for merge_clusters(), this function converts the baseline model into a directed graph.

    Parameters
    ----------
    regulators : dict
		Contains baseline model elements and corresponding regulator elements

    Returns
    -------
    G : DiGraph()
        Directed graph of the baseline model
    """

    G = nx.DiGraph()
    G.clear()
    for key, values in mdldict.items():
        G.add_node(key)
        for value in values:
            G.add_edge(value, key)

    return G

def get_args():

    parser = argparse.ArgumentParser(description="Network model extension using CLARINET")
    parser.add_argument('ReadingOutput', type=str,help="Reading output spreadsheet")
    parser.add_argument('Baseline', type=str,help="Baseline model in BioRECIPES format")
    parser.add_argument('out', type=str,help="Output directory")
    parser.add_argument('ReturnTh', type=str,help="Return path threshold")
    parser.add_argument('FCTh', type=str,help="Frequency class threshold")
    args = parser.parse_args()

    return(args)


def main():

    t0 = time.time()

    args = get_args()
    #Reading output (RO) .csv file. File format(RegulatedName,RegulatedID,RegulatedType,RegulatorName,RegulatorID,RegulatorType,PaperID)
    #for the other RO format use create_eclg_el
    interaction_filename = args.ReadingOutput
    #Baseline model in BioRECIPES tabular format
    model_dict, regulators = get_model(args.Baseline)
    weightMethod = 'FC' # or 'IF'
    G = create_eclg(interaction_filename, model_dict) #or use create_eclg
    G = node_weighting(G, args.FCTh, args.out)
    G = edge_weighting(G, args.out, weightMethod)
    clustering(G, args.out)
    # Get cluster information
    get_cluster_info(os.path.join(args.out,"GeneratedClusters"), os.path.join(args.out,"LSS"), args.out)
    merge_clusters(regulators, args.out, args.ReturnTh)

    t1 = time.time()
    total = t1-t0
    print("time to run CLARINET in seconds: " + str(total))


if __name__ == '__main__':
    main()
