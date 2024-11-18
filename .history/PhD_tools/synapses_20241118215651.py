import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster



def identify_polyads(df, t):
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame with same structure as neurosetta tree graph inputs.
    t : float
        threshold distance in same units as coordinates to collapse polyadic synapses together.

    Returns
    -------
    pandas.DataFrame
        Pandas Data Frame of 'input' synapse type, with added 'pre_id' column, which assigns a unique id to each pre-synapse, collapsing based on t
    """
    df['pre_id'] = None
    count = 0
    for in_N in np.unique(df.pre):
        # if only a single pre synapse
        if len(df.loc[df.pre == in_N]) == 1:
            count += 1
            df.loc[df.pre == in_N, 'pre_id'] = count
        else:
            # get coordinates
            coords = df.loc[df.pre == in_N].loc[:,["post_x", "post_y", "post_z"]].values
            Z = linkage(coords, method = 'single', metric = 'euclidean')
            labels = fcluster(Z,t = t, criterion = 'distance')
            labels = labels + count
            df.loc[df.pre == in_N,'pre_id'] = labels
            count = labels.max()
    return df



def get_polyads(df):
    """given dataframe returned from identify_polyads, returns centroid coordinates of each identified unique synapse, and mapping of synapse to post-synaptic partners.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe as returned by identify polyads

    Returns
    -------
    pd.DataFrame
        dataframe of collapsed synapsed, with synapse ID, pre-synapstic neuron id, and center of mass of points
    dict
        mapping from synapse ID to post-synaptic neuron ids    
    """
    data =df[['pre','post_x','post_y','post_z','post','pre_id']].values.astype(int)
    ids = np.unique(data[:,5])

    pre_id = []
    post_id = []
    coords = []

    for i in tqdm(ids):
        pre_id.append(data[np.where(data[:,5] == i)][:,0][0])
        coords.append(data[np.where(data[:,5] == i)][:,1:4].mean(axis = 0))
        post_id.append(np.unique(data[np.where(data[:,5] == i)][:,4]))

    pre_id = np.array(pre_id)
    coords = np.array(coords)
    data = np.column_stack([ids,pre_id,coords]).astype(int)

    collapsed_syns = pd.DataFrame(data,columns = ['syn_id','N_id','x','y','z'])
    map_dict = dict(zip(ids,post_id))
    return collapsed_syns,map_dict