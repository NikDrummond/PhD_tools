import numpy as np


def project_to_sphere(coords,sphere):
    r = sphere.radius
    center = sphere.center_of_mass()
    length = np.linalg.norm(coords, axis = 1)
    s_coords = (r / length)[:,np.newaxis] * (coords - center) 
    return s_coords