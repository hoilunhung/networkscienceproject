from networkx.generators.community import LFR_benchmark_graph
import networkx as nx
import networkx
from sklearn.metrics.cluster import normalized_mutual_info_score
from igraph import *


# ------  fast greedy------
k = Graph.Read_GML('karate.gml')
f = Graph.Read_GML('football.gml')
k_clusters = k.community_fastgreedy()
f_clusters = f.community_fastgreedy()
k_modularity = k.modularity(k_clusters.as_clustering())
f_modularity = f.modularity(f_clusters.as_clustering())
print("data set: karate - modularity of fast greedy", k_modularity)
print("data set: football - modularity of fast greedy", f_modularity)


# ------  #louvain  ------

k_clusters = k.community_multilevel()
f_clusters = f.community_multilevel()
k_modularity = k.modularity(k_clusters)
f_modularity = f.modularity(f_clusters)
print("data set: karate - modularity of louvain", k_modularity)
print("data set: football - modularity of louvain", f_modularity)

# ------  infomap  ------

k_clusters = k.community_infomap()
f_clusters = f.community_infomap()
k_modularity = k.modularity(k_clusters)
f_modularity = f.modularity(f_clusters)
print("data set: karate - modularity of infomap", k_modularity)
print("data set: football - modularity of infomap", f_modularity)


# ------  label propagation  ------
k_clusters = k.community_label_propagation()
f_clusters = f.community_label_propagation()
k_modularity = k.modularity(k_clusters)
f_modularity = f.modularity(f_clusters)
print("data set: karate - modularity of label propagation", k_modularity)
print("data set: football - modularity of label propagation", f_modularity)


# ------  Girvan newman  ------
k_clusters = k.community_edge_betweenness()
f_clusters = f.community_edge_betweenness()
k_modularity = k.modularity(k_clusters.as_clustering())
f_modularity = f.modularity(f_clusters.as_clustering())
print("data set: karate - modularity of Girvan newman", k_modularity)
print("data set: football - modularity of Girvan newman", f_modularity)


# ----newman eigen vector community detection---

k_clusters = k.community_leading_eigenvector()
f_clusters = f.community_leading_eigenvector()
k_modularity = k.modularity(k_clusters)
f_modularity = f.modularity(f_clusters)
print("data set: karate - modularity of newman eigen vector", k_modularity)
print("data set: football - modularity of newman eigen vector", f_modularity)

