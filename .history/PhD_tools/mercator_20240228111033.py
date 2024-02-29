import vedo as vd
import numpy as np


def fit_sphere(coords, sphere=False):

    s = vd.fit_sphere(coords)
    cent = s.center_of_mass()
    r = s.bounds()
    r = (r[1] - r[0]) / 2

    if sphere:
        return cent, r, s
    else:
        return cent, r


def project_to_sphere(point, center, radius):

    # shift origin of point to origin of sphere
    p = point - center
    # distance of point from origin
    length = np.linalg.norm(p)
    q = (radius / length) * p
    r = q + center

    return r


def long_lat(pnt, origin, r):
    """Convert point to longitude and latitude

    Parameters
    ----------
    pnt : _type_
        _description_
    origin : _type_
        _description_
    r : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    pnt = pnt - origin
    lat = np.arcsin((pnt[-1] / r))
    long = np.arctan2(pnt[0], pnt[1])
    return np.array([long, lat])


def mercator_proj(lnglat, truncate=False):
    """Convert longitude and latitude to web mercator x, y
    Parameters
    ----------
    lnglat : np.array
        Longitude and latitude array in decimal degrees, shape: (-1, 2)
    truncate : bool, optional
        Whether to truncate or clip inputs to web mercator limits.
    Returns
    -------
    np.array with x, y in webmercator
    >>> a = np.array([(0.0, 0.0), (-75.15963, -14.704620000000013)])
    >>> b = np.array(((0.0, 0.0), (-8366731.739810849, -1655181.9927159143)))
    >>> np.isclose(xy(a), b)
    array([[ True,  True],
           [ True,  True]], dtype=bool)
    """

    lng, lat = lnglat[0], lnglat[1]
    if truncate:
        lng = np.clip(lng, -180.0, 180.0)
        lat = np.clip(lng, -90.0, 90.0)
    x = 6378137.0 * np.radians(lng)
    y = 6378137.0 * np.log(np.tan((np.pi * 0.25) + (0.5 * np.radians(lat))))
    return np.array((x, y)).T
