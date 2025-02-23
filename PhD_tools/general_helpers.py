import Neurosetta as nr
from typing import List

def subset_neuron_list(N_all:List, by:str = 'subtype'):
    """ Get a dictionary of lists of neurons splitting the given list into T4 and T5 or T4/T5 subtypes"""

    subtypes = ['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']
    types = ['T4','T5']

    if by == 'subtype':
        sub_dict = {s:[] for s in subtypes}
        for N in N_all:
            sub_dict[N.graph.gp['subtype']].append(N)
        return sub_dict
    elif by == 'type':
        sub_dict = {s:[] for s in types}
        for N in N_all:
            sub_dict[N.graph.gp['type']].append(N)
        return sub_dict