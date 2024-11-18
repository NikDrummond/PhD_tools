import graph_tool.all as gt
import Neurosetta as nr
import numpy as np
import fastcluster as fc
from scipy.cluster.hierarchy import fcluster



# Function to return cluster labels from subset of nodes
def vertex_hierarchical_clusters(N: nr.Tree_graph | gt.Graph,subset: np.ndarray | None = None ,distance: str = 'Path Length',method: str = 'ward',k: int = 3) -> (np.ndarray,np.ndarray):
    """_summary_

    Parameters
    ----------
    N : Tree_graph | gr.Graph
        neurosetta.Tree_graph or graph_tool.Graph representation of a neuron
    subset : np.ndarray | None
        Subset of nodes to calculate cluster allocation to. If None (default) calculate for all nodes 
    distance : str
        Distance to use for hierarchical clustering, can be 'Path Length' or 'Euclidean'.
        'Path Length' uses the distance along the graph between nodes
        'Euclidean' used the Euclidean distance between nodes
    method : str
        Passed to hierarchical clustering, can be 'ward','average' or any other option presented in
        scipy.clustering.hierarchical.linkage
    k : int
        Number of clusters to return.

    Returns
    -------
    np.ndarray
        Resulting linkage matrix
    np.ndarray
        cluster identites, up to k, for each node in subset, or all nodes if no subset given.

    Raises
    ------
    TypeError
        if N is neither a neurosetta Tree_graph or graph_tool Graph
    """
    
    if isinstance(N, nr.Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    if subset is None:
        subset = g.get_vertices()    
    # generate pairwise distance matrix based on path length
    mat = nr.dist_mat(g, subset, method = distance, flatten=True)
    # generate linkage matrix
    Z = fc.linkage(mat, method = method)
    # cluster inds
    c_ids = fcluster(Z,k, criterion = 'maxclust')
    return Z, c_ids        

def linkage_cluster_permutation(clusters:np.ndarray,Z:np.ndarray,inplace = True,root_cluster:int | None = None,perms:int = 1000,a:float = 0.05, multicomp: str = 'Bonferroni'):
    """_summary_

    Parameters
    ----------
    clusters : np.ndarray
        _description_
    Z : np.ndarray
        _description_
    inplace : bool, optional
        _description_, by default True
    root_cluster : int | None, optional
        _description_, by default None
    perms : int, optional
        _description_, by default 1000
    a : float, optional
        _description_, by default 0.001

    Returns
    -------
    _type_
        _description_
    """
    cluster_ids = np.unique(clusters)

    if root_cluster is not None:
        np.delete(cluster_ids,np.where(cluster_ids == root_cluster))
    if not inplace:
        data = clusters.copy()

    for cluster_id in cluster_ids:
        cluster = np.where(clusters == cluster_id)[0]
        link_dist = np.array([Z[np.where( (Z[:,0] == c) | (Z[:,1] == c))][0][2] for c in cluster])
        # perform permutation
        perm_sample = np.zeros(perms)
        for i in range(perms):
            perm_sample[i] = np.mean(np.random.choice(link_dist,2))
        # calculate exact p    
        ps = np.array([len(perm_sample[ perm_sample >= t]) / len(perm_sample) for t in link_dist])  

        if isinstance(a,str):
            if a == 'Bonferroni':
                a = 0.05 / len(cluster_id)  
            elif a is None:    
        if inplace:
            clusters[cluster[np.where(ps <= a)]] = -1
        else:
            data[cluster[np.where(ps <= a)]] = -1

    if inplace:            
        return clusters
    else:
        return data     
