import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Neurosetta as nr
from scipy.stats import bootstrap

from tqdm import tqdm
import graph_tool.all as gt
from scipy.stats import beta


def add_subtype(fs):
    subtype = []
    for n in fs['N_id'].values:
        if 'T4a' in n.as_py():
            subtype.append('a')
        elif 'T4b' in n.as_py():
            subtype.append('b')
        elif 'T4c' in n.as_py():
            subtype.append('c')
        elif 'T4d' in n.as_py():
            subtype.append('d')
    fs['subtype'] = np.array(subtype)

# comparison plot function
def comparison_plot(fs, metric = 'branches', plot_test = False, y_axis = None):
    """"""
    # b mean
    b_mean = np.mean(fs[fs.subtype == 'b'][metric].values)
    # b conf.interval
    data = (fs[fs.subtype == 'b'][metric].values,)
    res = bootstrap(data,np.std,confidence_level=0.95, n_resamples = 10000, method = 'BCa')
    ci_b = res.confidence_interval
    # d mean
    d_mean = np.mean(fs[fs.subtype == 'd'][metric].values)
    # d conf.interval
    data = (fs[fs.subtype == 'd'][metric].values,)
    res = bootstrap(data,np.std,confidence_level=0.95, n_resamples = 10000, method = 'BCa')
    ci_d = res.confidence_interval

    df = fs[fs['subtype'].isin(['b','d'])][metric,'subtype'].to_pandas_df()

    fig, ax = plt.subplots(figsize = (3,10))

    sns.swarmplot(data = df,ax = ax, x = 'subtype',
                y = metric, hue = 'subtype',
                order=['b','d'], hue_order = ['b','d'],
                alpha = 0.5)


    # add mean points
    ax.scatter(x = 0, y = b_mean, c = 'k', s = 40)
    ax.scatter(x = 1, y = d_mean, c = 'k', s = 40)

    # add confidence intervals
    ax.plot([0,0],[b_mean - ci_b[0], b_mean + ci_b[1]],"k")
    ax.plot([1,1],[d_mean - ci_d[0], d_mean + ci_d[1]],"k")

    if plot_test == True:
        b = fs[fs.subtype == 'b'][metric].values
        d = fs[fs.subtype == 'd'][metric].values

        t, p = nr.boot_welchTtest(b,d,bootstraps = 20000)

        if p >= 0.975:
            p = 1 - p

        ax.set_title('t = ' + str(round(t,2)) + ', p = ' + str(round(p,2)))

    if y_axis is    not None:
        ax.set_ylabel(y_axis)

    plt.tight_layout()
    
    return ax

def hist_plot(fs,metric,n_bins):

    all_data = list(fs[fs.subtype.isin(['b','d'])][metric].values)
    bins = np.linspace(min(all_data),max(all_data),n_bins)

    fig, ax = plt.subplots(figsize = (10,8))
    sns.histplot(fs[fs.subtype == 'b'][metric].values, bins = bins, kde = True, stat = 'density',ax = ax, label = 'b')
    sns.histplot(fs[fs.subtype == 'd'][metric].values, bins = bins, kde = True, stat = 'density',ax = ax, label = 'd')
    ax.set_xlabel(metric)
    plt.legend()

    return ax

# Get probabilities

def get_betas(N_simp):

    # initialise data 
    # branch to root dist
    br_b = []
    br_d = []
    # branch to parent dist
    bp_b = []
    bp_d = []
    # end to root dist
    er_b = []
    er_d = []

    # for each neuron
    for N in tqdm(N_simp):

        # only look at b and d
        if (('T4d' in N.name) | ('T4b' in N.name)):
            # get an undirected copy of the graph
            g = N.graph.copy()
            g.set_directed(False)
            # get root
            root = np.where(N.graph.get_in_degrees(N.graph.get_vertices()) == 0)
            root = N.graph.vertex(root[0])
            # get branch inds
            out_deg = N.graph.get_out_degrees(N.graph.get_vertices())
            b_inds = np.where(out_deg >= 2)[0]
            # get leaf inds
            out_deg = N.graph.get_out_degrees(N.graph.get_vertices())
            l_inds = np.where(out_deg == 0)[0]
            # for each leaf
            for l in l_inds:
                # get leaf ind
                l = g.vertex(l)
                # get path
                vertex, edges = gt.shortest_path(g,l,root)
                #append distance
                if 'T4b' in N.name:
                    er_b.append(np.sum([g.ep['path_length'][g.edge(e.source(),e.target())] for e in edges]))
                elif 'T4d' in N.name:
                    er_d.append(np.sum([g.ep['path_length'][g.edge(e.source(),e.target())] for e in edges]))

            # for each branch ind
            for b in b_inds:
                # if we are not currently looking at the root
                if root != b:        
                # convert b to vertex
                    # convert to vertex
                    b = g.vertex(b)
                    # distant to parent
                    s,t  = N.graph.get_in_edges(b)[0]
                    pd = N.graph.ep['path_length'][N.graph.edge(s,t)]
                    # distance to root
                    vertex, edges = gt.shortest_path(g, b, root)
                    rd = np.sum([g.ep['path_length'][g.edge(e.source(),e.target())] for e in edges])
                    
                    if pd != rd:

                        if 'T4b' in N.name:
                            br_b.append(rd)
                            bp_b.append(pd)
                        elif 'T4d' in N.name:
                            br_d.append(rd)
                            bp_d.append(pd)

    # get fit parameters

    # ending
    er_b_params = beta.fit(er_b)
    er_d_params = beta.fit(er_d)

    # branch from root
    br_b_params = beta.fit(br_b)
    br_d_params = beta.fit(br_d)


    # branch from branch
    bp_b_params = beta.fit(bp_b)
    bp_d_params = beta.fit(bp_d)

    return er_b_params,er_d_params,br_b_params,br_d_params,bp_b_params,bp_d_params