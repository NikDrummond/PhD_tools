

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
            df.loc[df.pre == in_N] = count
        else:
            # get coordinates
            coords = df.loc[df.pre == in_N].loc[:,["post_x", "post_y", "post_z"]].values
            Z = linkage(coords, method = 'single', metric = 'euclidean')
            labels = fcluster(Z,t = t, criterion = 'distance')
            labels = labels + count
            df.loc[dfs.pre == in_N,'pre_id'] = labels
            count = labels.max()
    return df