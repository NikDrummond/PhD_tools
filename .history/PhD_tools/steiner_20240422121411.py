import itertools

import numpy as np
from collections import defaultdict

from graph_tool.search import BFSVisitor, bfs_search
from graph_tool import Graph, GraphView
from graph_tool.topology import min_spanning_tree

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

    # again speed up by predefining array and not iterating
    vfilt = gc.new_vertex_property("bool")
    vfilt.a = False
    for v in terminals:
        vfilt[v] = True
    gc.set_vertex_filter(vfilt)
    return gc, eweight, r2pred


def min_steiner_tree(g, obs_nodes, debug=False, verbose=False, root = None):
    if g.num_vertices() == len(obs_nodes):
        print("it's a minimum spanning tree problem")

    gc, eweight, r2pred = build_closure(g, obs_nodes, debug=debug, verbose=verbose)
    # print('gc', gc)

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

    efilt = g.new_edge_property("bool")
    for i, j in tree_edges:
        efilt[g.edge(i, j)] = 1
    return GraphView(g, efilt=efilt, vfilt=vfilt)
