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
def find_unique_max_assignments_with_improved_tie_handling(prob_matrix):
    """
    Finds the optimal set of [row, column] indices that maximize the sum of assigned values
    while handling ties systematically:
    - Rows with all zeros are ignored (assigned np.nan).
    - Non-zero ties are resolved by testing all possibilities for global optimization.
    - Ensures that no column is assigned to more than one row.

    Parameters:
        prob_matrix (numpy.ndarray): An n x m array of probabilities (values between 0 and 1).

    Returns:
        list: Optimal assignments (row, column) with np.nan for rows that are ignored.
    """
    n, m = prob_matrix.shape
    assignments = [None] * n  # Initialize assignments
    used_columns = set()  # Track assigned columns

    for i, row in enumerate(prob_matrix):
        max_val = np.max(row)
        tied_indices = np.where(row == max_val)[0]

        if max_val == 0:
            # Case 1: All values in the row are zero
            assignments[i] = np.nan
            continue

        # Track all valid columns for this row
        valid_columns = [col for col in tied_indices if col not in used_columns]
        if not valid_columns:
            # No valid column to assign
            assignments[i] = np.nan
            continue

        # Test all valid columns and pick the best option
        best_score = -np.inf
        best_assignment = None

        for col in valid_columns:
            # Create a temporary matrix where this row assigns to `col`
            temp_matrix = prob_matrix.copy()
            temp_matrix[i, :] = 0  # Zero out the row
            temp_matrix[i, col] = max_val  # Assign the tied value

            # Solve the assignment problem for the rest
            cost_matrix = -temp_matrix  # Negate for maximization
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            current_score = -cost_matrix[row_indices, col_indices].sum()

            if current_score > best_score:
                best_score = current_score
                best_assignment = col

        # Assign the best column, if found
        if best_assignment is not None:
            assignments[i] = (i, best_assignment)
            used_columns.add(best_assignment)
        else:
            # Fallback: assign the first available valid column
            fallback_column = valid_columns[0]
            assignments[i] = (i, fallback_column)
            used_columns.add(fallback_column)

    return assignments
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

