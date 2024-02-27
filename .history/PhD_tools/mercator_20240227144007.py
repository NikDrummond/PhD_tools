import vedo as vd
import numpy as np

def fit_sphere(coords, sphere = False):
    
    s = vd.fit_sphere(coords)
    cent = 