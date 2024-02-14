import fafbseg.flywire as fwy
import flybrains
import numpy as np
import trimesh as tri
import pandas as pd
import datetime as dt

from tqdm import tqdm

# get soma anotation table
somas = fwy.get_annotations('nuclei_v1')

# list neurotransmitters
neurotransmitters = ['gaba','acetylcholine','glutamate','octopamine','serotonin','dopamine']

# get proofread status table
client = fwy.get_cave_client('production')

mat_versions = client.materialize.get_versions()
materialization = max(mat_versions)


proof_table = client.materialize.query_table(table='proofreading_status_public_v1',
                                                materialization_version=materialization)

# whole brain surface mesh
brain = flybrains.FAFB14.mesh

def get_id_table(query:str)-> pd.DataFrame:
    """
    Get basic neuron ID table from flywire given query.

    Checks if proofread, and tries to get the soma possition as it is annotated in flywire

    Further, gives left/right possition, and checks if soma possition is within the whole brain mesh
    
    Returns pandas.DataFrame
    """

    ## Get all Mi1 neuron ids - for colums
    # get everything annotated as Mi1
    Ns = fwy.find_celltypes(query, exact = True)
    # the exact matching doesn't work, so  make sure we have only Mi1
    exact = [query+';' in i for i in Ns['tag'].values]
    # get the ids
    ids = Ns.root_id.values
    # subset to the ones we know are Mi1
    ids = ids[exact]
    # get the ones of these which are labeled as 'proofread'
    proof = fwy.is_proofread(ids)

    # somas is a table with soma coordinates in

    # pt_root_id is the root id of neurons

    soma_pos = []
    for i in ids:
        if i in somas.pt_root_id.values:
            soma_pos.append(somas.loc[somas.pt_root_id == i]['pt_position'].values[0])
        else:
            soma_pos.append(np.array([np.nan,np.nan,np.nan]))

    soma_pos = np.array(soma_pos)

    # get possition of soma
    lr_pos = fwy.get_lr_position(soma_pos)
    # annotate as left or right
    lr = []
    for i in lr_pos:
        if np.isnan(i):
            lr.append(np.nan)
        elif i > 0:
            lr.append('R')
        elif i < 0:
            lr.append('L')

    ## Test if soma coordinates are outside the brain mesh
    # add bool if soma is inside/outside the brain
    coords = soma_pos
    mesh_dist = []
    in_mesh = []
    for i in tqdm(coords, desc = 'Checking if soma in brain mesh: '):
        if np.isnan(i.sum()):
            mesh_dist.append(np.nan)
            in_mesh.append(np.nan)
        else:
            # get distance
            dist = tri.proximity.signed_distance(brain,[i])
            mesh_dist.append(dist[0])
            # if negative it is outside
            if np.sign(dist) < 0:
                in_mesh.append(False)
            # if positive is inside
            else:
                in_mesh.append(True)

    df = pd.DataFrame({'N_ids':ids,
                'is_proof':proof,
                'Query':query,
                'Soma_x':soma_pos[:,0],
                'Soma_y':soma_pos[:,1],
                'Soma_z':soma_pos[:,2],
                'lr_pos':lr_pos,
                'lr': lr,
                'in_mesh': in_mesh,
                'mesh_dist':mesh_dist
                })

    return df



def connectivity_summary(ids, return_conn = False):
    """
    Summary table for neuron connectivity
    """


    n_in = np.zeros(len(ids)).astype(int)
    n_out = np.zeros(len(ids)).astype(int)
    pre_ids = np.zeros(len(ids)).astype(int)
    post_ids = np.zeros(len(ids)).astype(int)
    in_proof = np.zeros(len(ids)).astype(int)
    out_proof = np.zeros(len(ids)).astype(int)
    in_proof_frac = np.zeros(len(ids)).astype(float)
    out_proof_frac = np.zeros(len(ids)).astype(float)
    main_nt = np.zeros(len(ids)).astype(str)
    nt_prop = np.zeros(len(ids)).astype(float)
    in_np = np.zeros(len(ids)).astype(str)
    out_np = np.zeros(len(ids)).astype(str)

    # get connectivity for all neruons

    conn = fwy.fetch_synapses(ids, filtered = True, 
                        transmitters = True,
                        clean = True,
                        neuropils = True, mat = 'latest')

    for i in range(len(ids)):
        N = ids[i]
        # number of input synapses - ones where the post synaptic site is N
        n_in[i] = len(np.unique(conn.loc[conn.post == N]['id'].values))
        # outputs, where presynaptic site is N
        n_out[i] = len(np.unique(conn.loc[conn.pre == N]['id'].values))
        # presynaptic neuron ids
        pre_id = np.unique(conn.loc[conn.post == N]['pre'].values)
        # postsynaptic ids
        post_id = np.unique(conn.loc[conn.pre == N]['post'].values)

        # add
        pre_ids[i] = len(pre_id)
        post_ids[i] = len(post_id)
        # bool ind of is proofread
        pre_proof = np.isin(pre_id,proof_table.pt_root_id.values)
        post_proof = np.isin(post_id,proof_table.pt_root_id.values)
        # number of proofread inputs / outputs
        in_proof[i] = pre_proof.sum()
        out_proof[i] = post_proof.sum()
        # fraction
        with np.errstate(divide='ignore'):
            in_proof_frac[i] = pre_proof.sum() / len(pre_id)
            out_proof_frac[i] = post_proof.sum() / len(post_id)

        # neurotransmitter raw data
        raw = conn.loc[conn.pre == N][neurotransmitters].values

        # lets create a binary version which only has the maximum?
        binary = np.zeros_like(raw)
        for j in range(len(raw)):
            
            binary[j,np.where(raw[j] == raw[j].max())] = 1

        # get fraction of each
        with np.errstate(divide='ignore'):
            nt_fracs = binary.sum(axis = 0) / binary.sum(axis = 0).sum()
        # if there is  nt info
        if not np.isnan(nt_fracs.sum()):
            # main type of nt
            main_nt[i] =  neurotransmitters[np.where(nt_fracs == nt_fracs.max())[0][0]]
            # probability of this
            nt_prop[i] = nt_fracs[np.where(nt_fracs == nt_fracs.max())][0]
        else:
            main_nt[i] = np.nan
            nt_prop[i] = np.nan

        # subset to inputs of N
        neuropils = conn.loc[conn.post == N].neuropil.values
        # set of neuropils
        neuropil_set = np.unique(neuropils)

        # if np data
        if not len(neuropil_set) == 0:
            # calculate the fraction of inputs in neuropil
            # initilise
            fractions = np.zeros_like(neuropil_set)
            for j in range(len(fractions)):
                current = neuropil_set[j]
                fractions[j] = len(neuropils[neuropils == current]) / len(neuropils)

            in_np[i] = neuropil_set[np.where(fractions == fractions.max())][0]

            ### As above but for outputs
            neuropils = conn.loc[conn.pre == N].neuropil.values
            # set of neuropils
            neuropil_set = np.unique(neuropils)
            # calculate the fraction of inputs in neuropil
            # initilise
            fractions = np.zeros_like(neuropil_set)
            for j in range(len(fractions)):
                current = neuropil_set[j]
                fractions[j] = len(neuropils[neuropils == current]) / len(neuropils)

            out_np[i] = neuropil_set[np.where(fractions == fractions.max())][0]
        else:
            in_np[i] = np.nan
            out_np[i] = np.nan

    proof = np.isin(ids,proof_table.pt_root_id)
    df = pd.DataFrame({'N_id':ids,
                'is_proofread':proof,
                'syn_in': n_in,
                'syn_out': n_out,
                'N_in': pre_ids,
                'N_out': post_ids,
                'N_in_proof': in_proof,
                'N_out_proof': out_proof,
                'N_in_proof_frac': in_proof_frac,
                'N_out_proof_frac': out_proof_frac,
                'nt': main_nt,
                'nt_frac':nt_prop,
                'In_np':in_np,
                'Out_np':out_np
                }
    )

    if return_conn:
        return df, conn
    else:
        return df
