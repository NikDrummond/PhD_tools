import Neurosetta as nr
import matplotlib.pyplot as plt

def TMD_barcode_plot(g, **kwargs):
    x,y = nr.TMD_barcode(g)
    fig, ax = plt.subplots(**kwargs)
    ax.plot(x,y,c = 'k')
    return fig, ax

def TMD_persistance_diagram_plot(g, **kwargs):
    x,y = nr.TMD_persistance_diagram(g, split = True)
    fig, ax = plt.subplots(**kwargs)
    ax.scatter(x,y,c = 'k')
    return fig, ax

def TMD_persistance_im_plot(im, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    ax.imshow(im, origin = 'lower')
    return fig, ax

