import numpy as np
from scipy.stats import vonmises_fisher as vmf



def project_to_sphere(coords,sphere):
    r = sphere.radius
    center = sphere.center_of_mass()
    length = np.linalg.norm(coords, axis = 1)
    s_coords = (r / length)[:,np.newaxis] * (coords - center) 
    return s_coords

### Cluster class
class cluster:

    def __init__(self,ID,n_ids,coords,types, columns):
        self.ID = ID
        self.n_ids = n_ids
        self.coords = coords
        self.types= types
        self.columns = columns
        
    def centroid(self):
        return self.coords.mean(axis = 0)
    def point_probability(self, points, likelihood = False):
        
        # normalise points
        if (points.ndim == 1) | (points.shape[0] == 1):
            points = points / np.linalg.norm(points)
        else:
            points = points / np.linalg.norm(points, axis = 1)[:,None] 
        

        prob = vmf.pdf(points, self.mu,self.kappa)
        
        if ~likelihood:
            prob = prob / vmf.pdf(self.mu,self.mu, self.kappa)
        
        return prob 

### basic assign column from prob
def assign_column(point, columns):
    curr_p = 0
    max_col = -1
    for i in columns:
        try:
            p = columns[i].point_probability(point)
            if p > curr_p:
                curr_p = p
                max_col = i
        except:
            pass

    return curr_p, max_col


### rayleigh test


def _components(
    data,
    p = 1.0,
    phi  = 0.0,
    axis = None,
    weights = None,
):
    # Utility function for computing the generalized rectangular components
    # of the circular data.
    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_to(weights, data.shape)
    except ValueError:
        raise ValueError("Weights and data have inconsistent shape.")

    C = np.sum(weights * np.cos(p * (data - phi)), axis) / np.sum(weights, axis)
    S = np.sum(weights * np.sin(p * (data - phi)), axis) / np.sum(weights, axis)

    return C, S
def _length(
    data,
    p = 1.0,
    phi = 0.0,
    axis = None,
    weights = None,
):
    # Utility function for computing the generalized sample length
    C, S = _components(data, p, phi, axis, weights)
    return np.hypot(S, C)

def rayleightest(
    data,
    axis = None,
    weights = None,
) :    
    n = np.size(data, axis=axis)
    Rbar = _length(data, 1.0, 0.0, axis, weights)
    z = n * Rbar * Rbar

    # see [3] and [4] for the formulae below
    tmp = 1.0
    if n < 50:
        tmp = (
            1.0
            + (2.0 * z - z * z) / (4.0 * n)
            - (24.0 * z - 132.0 * z**2.0 + 76.0 * z**3.0 - 9.0 * z**4.0)
            / (288.0 * n * n)
        )

    p_value = np.exp(-z) * tmp
    return z, p_value

