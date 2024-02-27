import vedo as vd
import numpy as np

def fit_sphere(coords, sphere = False):
    
    s = vd.fit_sphere(coords)
    cent = s.center_of_mass()
    r = s.bounds()
    r = (r[1] - r[0]) / 2

    if sphere:
        return cent, r, sphere
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