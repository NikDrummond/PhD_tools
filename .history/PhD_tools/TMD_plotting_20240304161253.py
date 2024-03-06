import Neurosetta as nr
import matplotlib.pyplot as plt

def TMD_barcode_plot(g, **kwargs):
    x,y = nr.TMD_barcode(g)
    fig, ax = plt.subplots()
    