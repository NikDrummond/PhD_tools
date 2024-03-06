import numpy as np

def scale(x, r_min=None, r_max=None, t_min=0, t_max=1):
    """Scales the values in x to be within [t_min, t_max] based on the range [r_min, r_max].

    If r_min and r_max are not provided, they are set to the min and max of x.
    If t_min and t_max are not provided, they default to 0 and 1. 

    Parameters
    ----------
    x : int | float | np.array
        object to scale
    r_min : None | float | int, optional
        minimum observed value, by default None (take minimum of input)
    r_max : None | float | int, optional
        maximum observed value, by default None (take minimum of input)
    t_min : float | int, optional
        target minimum value of scaled output, by default 0
    t_max : None | float | int, optional
        target maximum value of scaled output, by default 1

    Returns
    -------
    _type_
        Type returned is same as input. values scaled between t_min and t_max
    """
    # set r_min/max if not given
    if r_min is None:
        r_min = np.min(x)
    if r_max is None:
        r_max = np.max(x)

    # scale input
    x = ((x - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min

    return x