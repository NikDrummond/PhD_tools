# simple functions for set operations on families of sets

import numpy as np
from functools import reduce

def Sfamily_intersect(fam: list) -> np.array:
    """
    Returns the intersection of a family of sets.
    """
    # Input validation
    if not fam or not all(isinstance(s, (set, np.ndarray)) for s in fam):
        raise ValueError("Input must be a non-empty list of sets or arrays.")

    # Early exit if there's only one set in the family
    if len(fam) == 1:
        return np.array(list(fam[0]))

    # Convert arrays to sets
    fam_sets = [set(arr) for arr in fam]

    # Use reduce to iteratively find intersection
    intersect_set = reduce(lambda x, y: x.intersection(y), fam_sets)

    return np.array(list(intersect_set))

def Sfamily_union(fam: list) -> np.array:
    """
    Returns the union of a family of sets.
    """
    # Input validation
    if not fam or not all(isinstance(s, (set, np.ndarray)) for s in fam):
        raise ValueError("Input must be a non-empty list of sets or arrays.")

    # Early exit if there's only one set in the family
    if len(fam) == 1:
        return np.array(list(fam[0]))

    # Convert arrays to sets
    fam_sets = [set(arr) for arr in fam]

    # Use reduce to iteratively find union
    union_set = reduce(lambda x, y: x.union(y), fam_sets)

    return np.array(list(union_set))

def Sfamily_XOR(fam:list) -> np.array:
    """
    returns the symetrical difference of a family of sets - union with the intersection removed
    """
    intersection = fam[0]
    union = fam[0]
    for i in range(1,len(fam)):
        intersection = np.intersect1d(intersection,fam[i])
        union = np.union1d(union,fam[i])

    x_or = union[~np.isin(union,intersection)]

    return x_or