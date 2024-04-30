import graph_tool.all as gt
import Neurosetta as nr
from scipy.spatial import cKDTree as KDTree
import pickle
import numpy as np

### Bounding box function
def bounding_box(points, output = 'all'):
    x_coordinates, y_coordinates, z_coordinates= zip(*points)
    bb = np.array([[min(x_coordinates), min(y_coordinates),min(z_coordinates)], [max(x_coordinates), max(y_coordinates),max(z_coordinates)]])

    if output == 'box':
        return bb
    else:
        # get lengths along each axis
        difs = bb[1] - bb[0]
        if output == 'magnitude':
            return difs
        elif output == 'all':
            return bb,difs
        else:
            raise ValueError("output must be all, magnitude, or box")
        
def latice_shape(difs:np.ndarray, scaling:int = 40, padding:int = None ) -> np.ndarray:
    
    if padding == None:
        padding = scaling
    shape = np.round(difs,-1) + padding
    shape /= scaling
    shape = shape.astype(int)
    return shape     
    
def latice_graph(shape, lookup = True):
    """ Create a 2/3 dimensional spatially embedded latice graph.

    Parameters
    ----------
    shape : np.ndarray
        x/y/z dimensions/size of the latice to be generated - equates to number of nodes along each axis

    lookup: bool
        return a dictionary of coordinates : node inds for fast lookup    
    Returns
    -------
    gt.Graph
        Latice graph with adjacent nodes connected by edge of length = 1, with coordinates property for each node.        
    """
    coords = np.indices(shape).reshape(len(shape), -1).T
    kd_tree = KDTree(coords)
    edges= kd_tree.query_pairs(r = 1)
    g = gt.Graph(edges, hashed = True, hash_type = 'int')
    vp_coords = g.new_vp('vector<double>')
    coord_lookup = dict()
    for v in g.iter_vertices():
        c = coords[g.vp['ids'][v]]
        coord_lookup[tuple(c)] = v
        vp_coords[v] = c
    g.vp['coordinates'] = vp_coords 

    if lookup:
        return g, coord_lookup
    else:
        return g

def map_to_latice_axis(ax_min, ax_max, n_coord, interval):
    # Calculate the number of intervals
    num_intervals = (ax_max - ax_min) // interval

    if isinstance(n_coord,(int,float)):
        n_coord = [n_coord]
        
    # Calculate the indices of the bins where each element of c falls
    bin_indices = [(x - ax_min) // interval for x in n_coord]

    # Ensure bin_indices are within bounds
    bin_indices = np.array([max(0, min(bin_index, num_intervals - 1)) for bin_index in bin_indices])

    bin_indices = bin_indices.astype(int)
    if len(bin_indices) == 1:
        return bin_indices[0]
    else:
        return bin_indices  
    
def map_to_latice_3d(coords,ax_min,shape,step):
    ax_max = ax_min + shape * step
    # map coords
    xs = map_to_latice_axis(ax_min[0],ax_max[0],coords[0],40)
    ys = map_to_latice_axis(ax_min[1],ax_max[1],coords[1],40)
    zs = map_to_latice_axis(ax_min[2],ax_max[2],coords[2],40)

    return np.vstack((xs,ys,zs)).T    

def load_pickle(f_path:str):
    """ Load pickled variable"""
    with open(f_path, 'rb') as handle:
        b = pickle.load(handle)
    return b

def save_pickle(variable, s_path:str):
    """ Pickle variable """
    with open(s_path, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)                    

def latice_transform(N, step = 40):
    """ Given a neuron embedded in 3D space, convert it's coordinates to those of an embedding latice...

    Parameters
    ----------
    N : _type_
        _description_

    step: int
        default = 40, distance (in units of spatial embedding of original neuron) between each point in latice
    Returns
    -------    
    """
    bb,difs = bounding_box(nr.g_vert_coords(N),'all')
    shape = latice_shape(difs, scaling = step)
    ax_min = bb[0]
    ax_max = ax_min + shape * step
    # Actually I should just be able to copy the graph
    g2 = N.graph.copy()
    # get all coordinates
    coords = nr.g_vert_coords(N).T
    # transform them
    x = map_to_latice_axis(ax_min[0], ax_max[0],coords[0],step)
    y = map_to_latice_axis(ax_min[1], ax_max[1],coords[1],step)
    z = latices.map_to_latice_axis(ax_min[2], ax_max[2],coords[2],step)
    latice_coords = np.array((x,y,z))
    # replace the coordinates array
    g2.vp['coordinates'].set_2d_array(latice_coords)
    latice_N = nr.Tree_graph(name = N.name + '_latice_N', graph = g2)

    return latice_N        