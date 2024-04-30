import itertools

import numpy as np
from collections import defaultdict

from graph_tool.search import BFSVisitor, bfs_search
from graph_tool import Graph, GraphView
from graph_tool.topology import min_spanning_tree

import trimesh as tri
import Neurosetta as nr

def extract_edges_from_pred(source, target, pred):
    """edges from `target` to `source` using predecessor map, `pred`"""
    edges = []
    c = target
    while c != source and pred[c] != -1:
        edges.append((pred[c], c))
        c = pred[c]
    return edges


class DistPredVisitor(BFSVisitor):
    """visitor to track distance and predecessor"""

    def __init__(self, pred, dist):
        """np.ndarray"""
        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        s, t = int(e.source()), int(e.target())
        self.pred[t] = s
        self.dist[t] = self.dist[s] + 1


def init_visitor(g, root):
    dist = defaultdict(lambda: -1)
    dist[root] = 0
    pred = defaultdict(lambda: -1)
    vis = DistPredVisitor(pred, dist)
    return vis


def is_tree(t):
    # to undirected
    t = GraphView(t, directed=False)

    # num nodes = num edges+1
    if t.num_vertices() != (t.num_edges() + 1):
        return False

    # all nodes have degree > 0
    vs = list(map(int, t.vertices()))
    degs = t.degree_property_map("out").a[vs]
    if np.all(degs > 0) == 0:
        return False

    return True


def is_steiner_tree(t, X):
    if not is_tree(t):
        return False
    for x in X:
        if not has_vertex(t, x):
            return False
    return True


def has_vertex(g, i):
    # to avoid calling g.vertex
    return g._Graph__filter_state["vertex_filter"][0].a[i] > 0


def build_closure(g, terminals, debug=False, verbose=False):
    """build the transitive closure on terminals"""

    def get_edges(dist, root, terminals):
        """get adjacent edges to root with weight"""
        return ((root, t, dist[t]) for t in terminals if dist[t] != -1 and t != root)

    terminals = list(terminals)
    gc = Graph(directed=False)

    gc.add_vertex(g.num_vertices())

    edges_with_weight = set()
    r2pred = {}  # root to predecessor map (from bfs)

    # bfs to all other nodes
    for r in terminals:
        if debug:
            print("root {}".format(r))
        vis = init_visitor(g, r)
        bfs_search(g, source=r, visitor=vis)
        new_edges = set(get_edges(vis.dist, r, terminals))
        if debug:
            print("new edges {}".format(new_edges))
        edges_with_weight |= new_edges
        r2pred[r] = vis.pred

    for u, v, c in edges_with_weight:
        gc.add_edge(u, v)

    # edge weights
    eweight = gc.new_edge_property("int")
    weights = np.array([c for _, _, c in edges_with_weight])
    eweight.set_2d_array(weights)

    # # again speed up by predefining array and not iterating
    # vfilt = gc.new_vertex_property("bool")
    # vfilt.a = False
    # for v in terminals:
    #     vfilt[v] = True
    vfilt = np.zeros_like(gc.get_vertices())
    vfilt[terminals] = 1
    vfilt = gc.new_vp('bool',vfilt)

    gc.set_vertex_filter(vfilt)
    return gc, eweight, r2pred


def min_steiner_tree(g, obs_nodes, root = None, debug=False, verbose=False):

    if g.num_vertices() == len(obs_nodes):
        print("it's a minimum spanning tree problem")

    gc, eweight, r2pred = build_closure(g, obs_nodes, debug=debug, verbose=verbose)
    print('gc', gc)

    tree_map = min_spanning_tree(gc, eweight, root=root)
    tree = GraphView(gc, directed=True, efilt=tree_map)

    tree_edges = set()
    for e in tree.edges():
        u, v = map(int, e)
        recovered_edges = extract_edges_from_pred(u, v, r2pred[u])
        assert recovered_edges, "empty!"
        for i, j in recovered_edges:
            tree_edges.add(((i, j)))

    tree_nodes = list(set(itertools.chain(*tree_edges)))

    # this can be sped up by initialiseing the array
    # vfilt = g.new_vertex_property("bool")
    # vfilt.set_value(False)
    # for n in tree_nodes:
    #     vfilt[n] = True
    vfilt = np.zeros_like(g.get_vertices())
    vfilt[tree_nodes] = 1
    vfilt = g.new_vp('bool',vfilt)

    efilt = g.new_edge_property("bool")
    for i, j in tree_edges:
        efilt[g.edge(i, j)] = 1
    
    sg = GraphView(g, efilt=efilt, vfilt=vfilt)
    # purge the latice from the graph
    # sg.purge_vertices() 
    return sg



def latice_transform(N,step = 250,padding = -5):
    """_summary_

    Parameters
    ----------
    N : _type_
        _description_
    step : int, optional
        _description_, by default 250
    padding : int, optional
        _description_, by default -5

    Returns
    -------
    _type_
        _description_
    """
    ### Convert graph coordinates to latice space
    # Get bounds of bounding box, and lengths along each dimension
    bb, difs = latices.bounding_box(nr.g_vert_coords(N),'all')
    # get shape of bounding box (bins lengths into bins of width == step)
    shape = latices.latice_shape(difs,scaling = step)
    # minimum and maximum of ax
    ax_min = bb[0]
    ax_max = ax_min + shape * step
    # Copy the graph
    g2 = N.graph.copy()
    # get all coordinates
    coords = nr.g_vert_coords(N).T
    # transform them to discrete space
    x = latices.map_to_latice_axis(ax_min[0], ax_max[0],coords[0],step)
    y = latices.map_to_latice_axis(ax_min[1], ax_max[1],coords[1],step)
    z = latices.map_to_latice_axis(ax_min[2], ax_max[2],coords[2],step)
    latice_coords = np.array((x,y,z))
    # replace the coordinates array in the copy of the graph
    g2.vp['coordinates'].set_2d_array(latice_coords)
    # turn into nr.Tree_graph
    latice_N = nr.Tree_graph(name = N.name + '_latice_N', graph = g2)

    ### Get convex hull we will use to limit the size of the latice graph
    # Given this, get convex hull of latice points
    hull = vd.ConvexHull(latice_coords).alpha(0.2)
    
    ### Generate all points in the latice
    
    lat_coords = coords = np.indices(shape).reshape(len(shape), -1).T


    ### subset to those within the convex hull
    # initialise mask we will use to subset coordinates
    mask = np.zeros(lat_coords.shape[0])
    # turn convex hull into trimesh (to get signed distance)
    mesh = tri.Trimesh(hull.points(),hull.faces())
    # iterate through coords, mask == True if we are keeping
    for i in tqdm(range(len(lat_coords)), desc = 'limiting to hull: '):

        # get signed distance - negative is outside
        dist = tri.proximity.signed_distance(mesh, [lat_coords[i]])[0]
        # if the distance is greater than the padding value, we include it
        if dist > padding:
            mask[i] = 1
        else:
            pass      
    # apply mask    
    lat_coords = lat_coords[mask.astype(bool)]   
    ### Make latice points into a graph using KDTree      
    kd_tree = KDTree(lat_coords)
    edges = kd_tree.query_pairs(r = 1)
    # make graph
    g = gt.Graph(edges, hashed = True, hash_type = 'int')
    vp_coords = g.new_vp('vector<double>')
    coord_lookup = dict()
    for v in tqdm(g.iter_vertices(),desc = 'building latice: '):
        c = lat_coords[g.vp['ids'][v]]
        coord_lookup[tuple(c)] = v
        vp_coords[v] = c
    g.vp['coordinates'] = vp_coords 

    g.set_directed(False)

    return latice_N, g, coord_lookup

def get_terminals(N,lookup,method = 'leaves'):
    """_summary_

    Parameters
    ----------
    N : _type_
        _description_
    lookup : _type_
        _description_
    method : str, optional
        _description_, by default 'leaves'
    """
    if method == 'leaves':
        t_coords = nr.g_vert_coords(N,nr.g_lb_inds(N))
    elif method == 'branches':    
        t_coords = nr.g_vert_coords(N,nr.g_branch_inds(N))
    elif method == 'both':
        t_coords = nr.g_vert_coords(N,nr.g_lb_inds(N))
    
    root_coord = nr.g_vert_coords(N,nr.g_root_ind(N))[0]

    root = lookup[tuple(root_coord)]

    terminals = [lookup[tuple(i.astype(int))] for i in t_coords]
    terminals.append(root)

    return terminals, root

