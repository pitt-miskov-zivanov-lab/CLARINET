"""
@author: Yasmine Ahmed
"""

import pandas as pd
import re
import numpy as np
import networkx as nx
import math
import pickle
import community
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import time
from datetime import datetime, date


def create_eclg(interaction_filename,model_dict):   
    """ 
    This function creates the ECLG where a node is an event (e.g., biochemical interaction) and there
    is an edge between two nodes (events) if they happen to occur in the same paper.
     
    Parameters
    ----------
    model_dict : dict
         Dictionary that holds baseline model regulator and regulated elements
    interaction_filename : str 
         The path of the reading output file
         
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
            print((interaction))
            papers.append(interaction[25])
            
    for p in np.unique(papers):
        #print(p+" "+str(papers.count(p)))
        Mean_interactions_per_paper+=papers.count(p)    
        
    Paper_ID_unique=np.unique(papers)
    print("Paper_ID_unique "+str(len(Paper_ID_unique)))
    print("Mean_interactions_per_paper "+str(Mean_interactions_per_paper/len(Paper_ID_unique)))
            

    G = nx.Graph() #this will contain ECLG nodes and edges
    IDs = len(Paper_ID_unique)
    curr_map=dict()
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
                    #print('repeated')
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
    is an edge between two nodes (events) if they happen to occur in the same paper. The reading output 
    file header must contain the following fields: Element Name, Element Type, Element Identifier, 
    PosReg Name/Type/ID, NegReg Name/Type/ID and Paper ID.
     
    Parameters
    ----------
    model_dict : dict
         Dictionary that holds baseline model regulator and regulated elements
    interaction_filename : str 
         The path of the reading output file
         
     Returns
     -------
     G : Graph
         Event CoLlaboration Graph
    """ 
    df = pd.read_excel(interaction_filename)
    papers=list()
    #print(type(df['Element Name'].iloc[7587]))
    
    #remove date entries
    date_indices = list()
    for idx,ele_name in enumerate(df['Element Name']):
        #print(idx)
        if isinstance(ele_name, datetime) or isinstance(df['PosReg Name'].iloc[idx], datetime) or isinstance(df['NegReg Name'].iloc[idx], datetime):
            date_indices.append(int(idx))            
    #print((date_indices))         
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
        #papers.count(p)
        #print(p+" "+str(papers.count(p)))
        Mean_interactions_per_paper+=papers.count(p)
        
    print("Mean_interactions_per_paper "+str(Mean_interactions_per_paper/len(Paper_ID_unique)))
    
    G = nx.Graph()
    IDs=len(Paper_ID_unique)
    
    curr_map=dict()
    #print(str(Paper_ID.iloc[4407]))
    model_dict, regulators = get_model("/Users/jasmine/Desktop/PCCstudy/All/PCC_ALL.xlsx")
    for jj in range(0,IDs):
        my_list= []
        for idx,ele_name in enumerate(Element_name):
            #print(Paper_ID_unique[jj])
            #print(idx)
            #print(len(IDs))
            if Paper_ID_unique[jj] in Paper_ID.iloc[idx]:            
                inter1P='' 
                #print(idx)
                elem=getVariableName(model_dict,curr_map,[(ele_name),'','',Element_ID.iloc[idx],'',Element_type.iloc[idx],'',''])#ele_name
                Pos=getVariableName(model_dict,curr_map,[str(PosReg_Name.iloc[idx]),'','',str(PosReg_ID.iloc[idx]),'',str(PosReg_type.iloc[idx]),'',''])#PosReg_Name[idx]
                Neg=getVariableName(model_dict,curr_map,[str(NegReg_Name.iloc[idx]),'','',str(NegReg_ID.iloc[idx]),'',str(NegReg_type.iloc[idx]),'',''])#NegReg_Name[idx]
                #print(Neg)
    
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
                
                if str(Neg) != "nan_ext"and str(Neg) != "":#type(Neg) is not float:
                    #print(("habal"))
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
    This function assigns weights to nodes using frequency class.
     
    Parameters
    ----------
    G : undirected graph
        ECLG 
    freqTh : int
        Frequency class threshold value, events (nodes) having FC > this value will be removed
    path : str
        The output directory where the genereted files will be saved
    
     Returns
     -------
     G : undirected graph
        ECLG after the removal of the less frequent nodes

    """ 
    # Assign weights to nodes using frequency class concept
    
    ebunch = list() #less frequent nodes that will be removed
    nodesDegree = list()
    fill = path + "freqClass"
    output_stream = open(fill, 'w')
    
    for g in G.nodes:
        nodesDegree.append(G.degree[g])
        if G.degree[g] == 0: 
            continue
        #print(G.degree[g])
        
    freqMostCommonNode = max(nodesDegree) 
    print("Frequency of most frequent node before: "+str(freqMostCommonNode))
    
    
    #Mapping node names to frequency class
    
    for g in G.nodes:
        if G.degree[g] == 0: continue
        freqCLASS = math.floor(0.5-np.log2(G.degree[g]/freqMostCommonNode))
        newName = g + '->'+str(freqCLASS)
        mapping = {g: newName}
        G = nx.relabel_nodes(G, mapping)
        
    print("Total number of nodes before removing less frequent nodes:")
    print(len(G.nodes())) 
    print("Total number of edges before removing less frequent nodes:")
    print(len(G.edges()))
    nx.write_edgelist(G, path+"ECLGbefore.txt") #ECLG nodes and edges before the removal of less frequent nodes
    print("hist before removing less frequent nodes")
    print(nx.degree_histogram(G))  
    
    for g in G.nodes:
        if G.degree[g] == 0: 
            continue
        freqCLASS = math.floor(0.5-np.log2(G.degree[g]/freqMostCommonNode))
        if freqCLASS > int(freqTh): # Frequency class threshold which may vary from case study to another (set it =2 for T cell case study)
            ebunch.append(g) 
        output_stream.write(g+' '+str(G.degree[g])+' '+str(freqCLASS)+'\n')
        nodesDegree.append(G.degree[g])    
    G.remove_nodes_from(ebunch)   
    
    nx.write_edgelist(G, path+"ECLGafter.txt") #ECLG nodes and edges after the removal of less frequent nodes
    print("node length after removing less frequent nodes")
    print(len(G.nodes()))  
    print("edge length after removing less frequent nodes")
    print(len(G.edges()))    
    print("Frequency of most frequent node after: "+str(freqMostCommonNode))
    print("hist after removing less frequent nodes")
    print(nx.degree_histogram(G))
    return G

# Pair assessment (IA)
def edge_weighting(G, path, weightMethod):
    """ 
    This function assigns weights to nodes using frequency class (FC) or inverse frequency formula (IF).
     
    Parameters
    ----------
    G : undirected graph
        ECLG 
    path : str
        The output directory where the genereted files will be saved
    weightMethod : str
        FC or IF
    
     Returns
     -------
     G : undirected graph
        ECLG after assigning weights to edges
    """
    # Assign weights to edges using frequency class concept (fc) or inverse frequency concept (iif)
    
    fill = path + "LSSfc" #This file will be input to runClusterInfo.py
    output_stream = open(fill, 'w')
    N = G.number_of_nodes()
    weights = list()
    
    for n1, n2, w in G.edges.data():
        weights.append(G[n1][n2]['weight'])
        
    maxWeight = int(max(weights))
    
    for n1, n2, w in G.edges.data():
        w1 = G[n1][n2]['weight']
        p1 = np.log(N/G.degree[n1])#iif
        p2 = np.log(N/G.degree[n2])#iif
        dd = math.floor(0.5-np.log2(w1/(maxWeight))) #fc
        #dd=w1*(p1 + p2) #iif (uncomment if you wanna use iif to assign edge weights instead of fc)
        #if dd < 0: dd = 0 #iif (uncomment if you wanna use iif to assign edge weights instead of fc)
        G[n1][n2]['weight']= dd #np.log(p1_2/((p1)*(p2)))
        output_stream.write(n1+' '+n2+' '+str(dd)+'\n')
    return G
    
def clustering(G, path):
    """ 
    This function clusters the ECLG using the community detection algorithm by Blondel et al..
     
    Parameters
    ----------
    G : undirected graph
        ECLG 
    path : str
        The output directory where the genereted files will be saved      
        
    Returns
    -------

    """ 
    #Clustering
    
    partition = community.best_partition(G)
    centers = {}
    communities = {}
    G_main_com = G.copy()
    min_nb = 2
    com_edges = list()    
    group_num=1
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
            print('Community of ', center , '(ID ', com, ') - ', len(list_nodes), ' Interactions:')
            print(list_nodes, '\n')
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
            com_edges.append([group_num]+NODESS)#
            group_num=group_num+1
    pickle.dump(com_edges, open(path + "grouped_ext",'wb')) #all clusters in one pickle file
    
#    # Display graph
#    plt.figure(figsize=(13, 5))
#    node_size = 30
#    count = 0
#    pos = nx.spring_layout(G_main_com)
#    colors = dict(zip(communities.keys(), sns.color_palette('hls', len(communities.keys()))))
#    print("hihi")
#    for com in communities.keys():
#        count = count + 1
#        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com and nodes not in communities.values()]
#        nx.draw_networkx_nodes(G_main_com, pos, list_nodes, node_size = node_size, node_color = colors[com])
#        nx.draw_networkx_nodes(G_main_com, pos, list([communities[com]]), node_size = node_size*5, node_color = colors[com])
#    nx.draw_networkx_edges(G_main_com, pos, alpha=0.5)
#    labels = {k: k for k,v in centers.items()}    
#    nx.draw_networkx_labels(G_main_com, pos, labels)
#    plt.axis('off')
#    plt.show()    
    

    #Save each cluster in a separate file
    
    thePath = os.path.join(path, "GeneratedClusters/") # Directory containing generated clusters 
    os.mkdir(thePath)
    clusterFile = thePath + 'GeneratedCluster'  # generated (uninterpreted) clusters 
    
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
        
    thePath = os.path.join(path, "InterpretedClusters/")
    os.mkdir(thePath)
    clusterFile = thePath + 'InterpretedCluster' #interpreted clusters  
    
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
        
    
def merge_clusters(regulators, path, ReturnTh):
    """ 
    This function prints indices of clusters  to be merged based on the existence of one or more return paths.
    It generates the grouped_ext_Merged pickle file that contains the merged clusters.
     
    Parameters
    ----------
    regulators : dict
        contains baseline model elements and corresponding regulator elements
    path : str
        The path of the directory that contains the grouped_ext file 
        
     Returns
     -------
    """
    # Merge clusters if there is one or more return paths
    
    G = nx.DiGraph()
    G = make_diGraph(regulators)
    com_edges = list()
    group_num = 1
    extensions = pickle.load(open(path+"grouped_ext",'rb'))    
    
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
                print('merge:')
                print(str(ii) + " and " + str(jj))
                print(count)
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
                
    pickle.dump(com_edges, open(path + "grouped_ext_Merged",'wb')) #Merged clusters 
    
    return

#This and the following function are inherited from DySE framework
# define regex for valid characters in variable names
_VALID_CHARS = r'a-zA-Z0-9\@\_\/'
def get_model(model_file: str):
	""" Return a dictionary of the model regulators, ids, etc.
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
	""" Match the element name from the extension to an element in the model 
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
    This function creates a directed graph for a model in the BioRECIPES format.
     
    Parameters
    ----------
    mdldict : dict
        contains baseline model elements and corresponding regulator elements
        
     Returns
     -------
     G : DiGraph
         contains the baseline model in the form of directed graph where nodes 
         are model elements and edges are interaction between model elements.
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
    merge_clusters(regulators, args.out, args.ReturnTh)
    
    t1 = time.time()
    total = t1-t0
    print("time to run CLARINET in seconds: " + str(total))


if __name__ == '__main__':
    main()