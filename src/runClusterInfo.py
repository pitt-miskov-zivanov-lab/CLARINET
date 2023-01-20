"""
@author: Yasmine
"""

import os
import networkx as nx
import argparse



def get_args():
    
    parser = argparse.ArgumentParser(description="Computing parameters for each generated cluster")
    parser.add_argument('path', type=str,help="directory of generated clusters")
    parser.add_argument('out', type=str,help="Output directory")
    parser.add_argument('ClarinetOutFile', type=str,help="The file that contains the weighted edges of the ECLG")
    args = parser.parse_args()
    
    return(args)

def main():
    
    args = get_args()
    path = args.path#directory of generated clusters
    entries = os.listdir(path)
    NewFile = args.out + "ClusterInfoFile" #output file that contains clusters'info
    output_stream = open(NewFile,"w")
    output_stream.write("Cluster index, Nodes, Edges, Density, AvgPathLength, Coeff, LSS, NodesX, EdgesX, DensityX, AvgPathLength, CoeffX, FreqClass, node_perc, edg_perc"+'\n') 
    frequencyCLasses = dict()
    
    
    with open(args.ClarinetOutFile) as f:#This file that has been generated from runClarinet, it contains weighted edges of the ECLG, its name is LSSfc
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
        if entry == ".DS_Store": continue
    
        G = nx.Graph() #generated cluster 
        G1 = nx.Graph() #interpreted cluster
        #count = 0
        weights = []
        
        
        with open(path+"/"+entry) as f:
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
            nod_perc = 100*(nod/num_of_nodes)                    
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
                
            output_stream.write(str(Cl)+', '+str(G.number_of_nodes())+', '+str(G.number_of_edges())+', '+str(nx.density(G))+', '+str(Paths)+', '+str(nx.average_clustering(G))+', '+str(sum(weights)/len(weights))+', '+str(G1.number_of_nodes())+', '+str(G1.number_of_edges())+', '+str(nx.density(G1))+', '+str(Paths1)+', '+str(nx.average_clustering(G1))+', '+str(freqCount)+', '+str(nod_perc)+'\n')  #average for LSS
    output_stream.close() 
    

if __name__ == '__main__':
    main()
    
    