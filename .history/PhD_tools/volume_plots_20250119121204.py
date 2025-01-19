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

### Unique column assignemnt/ linear assignment problem

def unique_col(columns, df, Type, Subtype, sphere_cols, method = 'prob'):
    tmp_df = df.loc[(df.Type == Type) & (df.Subtype == Subtype),['root_x','root_y','root_z']]

    if method == 'prob':
        # initialise array
        prob_array = np.zeros((len(tmp_df),len(columns)))

        for i in tqdm(columns):
            try:
                prob_array[:,i] = columns[i].point_probability(tmp_df.values, likelihood = True)
            except:
                pass

    elif method == 'euclidean':
       prob_array = 1 / cdist(tmp_df[['root_x','root_y','root_z']].values, sphere_cols, metric='euclidean')
    else:
        raise ValueError('method not recognised')

    try:
        assignments = find_unique_max_assignments_with_advanced_tie_handling(prob_array)
        # assignments = find_unique_max_assignments(prob_array)
        # print("Optimal assignments (row, column):", assignments)
        # total_value = sum(prob_array[row, col] for row, col in assignments)
        # print("Total value of assignments:", total_value)
        # max_prob = np.array([prob_array[a] for a in assignments])

        max_prob = np.zeros(len(assignments))
        for i in range(len(max_prob)):
            a = assignments[i]
            if ~np.isnan(a[1]):
                max_prob[i] = prob_array[a]
            else:
                max_prob[i] = np.nan

        unique_col = np.array([a[1] for a in assignments])
        
    except ValueError as e:
        print(Type + Subtype + ' Failed')
        print("Error:", e)

    # unpack column coordinates
    coords = []
    for i in unique_col:
        if np.isnan(i):
            c = np.array([np.nan,np.nan,np.nan])
        else:
            try:
                c = sphere_cols[int(i)]
            except:
                print(i)
        coords.append(c)
    coords = np.array(coords)

    df.loc[(df.Type == Type) & (df.Subtype == Subtype),'r_unique_c'] = unique_col
    df.loc[(df.Type == Type) & (df.Subtype == Subtype),'r_unique_p'] = max_prob
    df.loc[(df.Type == Type) & (df.Subtype == Subtype),['r_unique_x','r_unique_y','r_unique_z']] = coords

    return df

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

