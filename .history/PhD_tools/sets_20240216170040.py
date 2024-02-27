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

def Sfamily_XOR(fam: list) -> np.array:
    """
    Returns the symmetric difference of a family of sets - union with the intersection removed.
    """
    # Input validation
    if not fam or not all(isinstance(s, (set, np.ndarray)) for s in fam):
        raise ValueError("Input must be a non-empty list of sets or arrays.")

    # Early exit if there's only one set in the family
    if len(fam) == 1:
        return np.array(list(fam[0]))

    # Convert arrays to sets
    fam_sets = [set(arr) for arr in fam]

    # Calculate intersection and union
    intersection = fam_sets[0]
    union = fam_sets[0]
    for s in fam_sets[1:]:
        intersection &= s
        union |= s

    # Calculate symmetric difference
    x_or = np.array(list(union - intersection))

    return x_or