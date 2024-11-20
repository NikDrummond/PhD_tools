

def CH(
    N: nr.Tree_graph | gt.Graph,
    dist_mat: None | np.ndarray,
    N: None | int = None,
    l_inds: None | np.ndarray = None,
) -> gt.Graph | nr.Tree_graph:
    """_summary_

    Parameters
    ----------
    N : nr.Tree_graph | gt.Graph
        _description_
    dist_mat : None | np.ndarray
        _description_
    N : None | int, optional
        _description_, by default None
    l_inds : None | np.ndarray, optional
        _description_, by default None

    Returns
    -------
    gt.Graph | nr.Tree_graph
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    # make sure we have a graph type we can use
    if isinstance(N, nr.Tree_graph):
        g = N.graph
    elif isinstance(N, gt.Graph):
        g = N
    else:
        raise TypeError("N must be Tree_graph or gt.Graph")

    # get current roots
    cluster_roots = np.where(g.degree_property_map("in").a == 0)[0]
    # first check that we have only one cluster
    if len(cluster_roots) == 1:
        print("Single cluster is undefined, returning 0")
        return 0
    else:
        ### Make sure we have the optional bits we want
        if dist_mat is None:
            verts = g.get_vertices()
            dist_mat = nr.dist_mat(g, verts, method="Path Length")
        if N is None:
            N = len(gt.get_vertices())
        if l_inds is None:
            l_inds = nr.g_leaf_inds(g)
        ### Get current K
        K = len(cluster_roots)
        ### Work out B

        # get the sub array of path lengths between roots from distance matrix
        dists = dist_mat[np.ix_(cluster_roots, cluster_roots)]
        # square and sum values in upper triangle of sub array
        B = sum(dists[np.triu_indices(len(cluster_roots), k=1)] ** 2)
        # get the sub array of path lengths between roots from distance matrix
        dists = dist_mat[np.ix_(cluster_roots, cluster_roots)]
        # square and sum values in upper triangle of sub array
        B = sum(dists[np.triu_indices(len(cluster_roots), k=1)] ** 2)

        ### Work out W

        # root inds we have in cluster roots already
        # for each cluster get the squared distance
        sq_dist = np.array(
            [
                gt.shortest_distance(
                    g, cr, l_inds, weights=g.ep["Path_length"], dag=True
                )
                for cr in cluster_roots
            ]
        )
        # remove infinities and squ
        sq_dist = sq_dist[sq_dist != np.inf] ** 2
        W = np.sum(sq_dist)

        return (B / W) * ((N - K) / K - 1)