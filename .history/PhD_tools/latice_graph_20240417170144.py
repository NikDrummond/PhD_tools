import graph_tool.all as gt
import Neurosetta as nr
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