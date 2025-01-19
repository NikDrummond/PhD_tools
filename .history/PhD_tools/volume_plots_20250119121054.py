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


