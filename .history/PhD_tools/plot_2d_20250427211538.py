import numpy as np

def Semi_axis_length(N_all, confidence = 0.9973, return_evecs = True, return_2D = True):
    # initialise empty array to put data in 
    axis_len_data = np.zeros((N_all.graph.num_vertices(), 3))
    evec_data = np.zeros_like(axis_len_data)
    # iterate over neurons
    for v in tqdm(N_all.graph.iter_vertices()):
        coords = nr.g_vert_coords(N_all.graph.vp['Neurons'][v])
        coords = coords * 1e-3
        evals, evecs = GeoJax.coord_eig_decomp(coords, robust = True, center = True, PCA = False, transpose = True)
        axis_len_data[v] = evals
        # keep only the first eigenvector
        evec_data[v] = evecs[0]

    # Chi_square value for 3 Dof (99.73% confidence/3-sigma)
    chi2_val = chi2.ppf(confidence, df = 3) 
    # scaled eigenvalues
    scaled_data = np.sqrt(axis_len_data * chi2_val)

    if return_2D:
        scaled_data = scaled_data[:,[0,1]]
    if return_evecs:
        return scaled_data, evec_data
    else:
        return scaled_data