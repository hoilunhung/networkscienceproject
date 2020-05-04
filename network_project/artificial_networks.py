from networkx.generators.community import LFR_benchmark_graph
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from igraph import *
import math
import numpy as np
from sklearn import metrics
parameter = 0

G1 = LFR_benchmark_graph(1000,2.5,1.5,0.2,average_degree=20,max_degree= 50,min_community= 20,max_community= 50)  #LFR
G2 = LFR_benchmark_graph(1000,2.5,1.5,0.4,average_degree=20,max_degree= 50,min_community= 20,max_community= 50)  #LFR
G3 = LFR_benchmark_graph(1000,2.5,1.5,0.6,average_degree=20,max_degree= 50,min_community= 20,max_community= 50)  #LFR
G4 = LFR_benchmark_graph(1000,2.5,1.5,0.8,average_degree=20,max_degree= 50,min_community= 20,max_community= 50)  #LFR
G5 = LFR_benchmark_graph(1000,2.5,1.5,1.0,average_degree=20,max_degree= 50,min_community= 20,max_community= 50)  #LFR

def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def compute_NMI(G):
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    true_dict = {}
    print(len(communities))
    print(communities)
    i = 0
    for c in communities:
        label = [i] *len(c)
        true_dict.update(dict(zip(list(c), label)))
        i += 1
    ig = nx.Graph()
    for e in G.edges():
        ig.add_edge(*e)
    nx.write_gml(ig, 'lfr_graph.gml')
    lfr_graph = Graph.Read_GML('lfr_graph.gml')
    #p1 = lfr_graph.community_multilevel()
    #p1= lfr_graph.community_fastgreedy()
    #p1 = lfr_graph.community_infomap()
    p1 = lfr_graph.community_label_propagation()
    #p1 = lfr_graph.community_edge_betweenness()
    #p1 = lfr_graph.community_leading_eigenvector()
    i = 0
    pre_dict = {}
    for p in p1:
        label = [i ] * len(p)
        pre_dict.update(dict(zip(p, label)))
        i += 1
    d = sorted(true_dict.items(), key = lambda k: k[0])
    d1 = sorted(pre_dict.items(), key = lambda k: k[0])
    pred_label = []
    true_label = []
    for i in range(len(d)):
        pred_label.append(d1[i][1])
        true_label.append(d[i][1])
    print(pred_label)
    print(true_label)
    print(normalized_mutual_info_score(true_label,pred_label ))

compute_NMI(G1)
compute_NMI(G2)
compute_NMI(G3)
compute_NMI(G4)
compute_NMI(G5)
