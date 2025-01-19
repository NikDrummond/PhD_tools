from ast import Raise
import matplotlib.pyplot as plt

# Define A4 size in cm
A4_width_cm = 21.0
A4_height_cm = 29.7

# Margins
margin_top = 2
margin_bottom = 2
margin_left = 2.5
margin_right = 2.5


# Convert cm to inches for Matplotlib
cm_to_inch = 1 / 2.54

# account for margins
# width subtract margin
fig_width = A4_width_cm - margin_left - margin_right
# convert to inches
fig_width_in = fig_width * cm_to_inch

# height subtract margin
fig_height = A4_width_cm - margin_top - margin_bottom
# convert to inches
fig_height_in = fig_height * cm_to_inch

def A4_figure():

    return plt.figure(figsize=(fig_width_in, fig_height_in))

def fig_add_axes(fig,x,y,height,width):

    # make sure we are in bounds
    assert x>=0,'x possition must be >= 0'
    assert y>=0, ' y posstion'
    # bottom left is 0,0
    if (y + height >= fig_height) | (x + width >= fig_width):
        TypeError ('proposed size out of bounds')

    # Convert scatter plot dimensions to relative figure coordinates
    scatter_x_rel = x / width
    scatter_y_rel = y / height
    scatter_width_rel = width / width
    scatter_height_rel = height / height

    ax = fig.add_axes([scatter_x_rel, scatter_y_rel, scatter_width_rel, scatter_height_rel])

    return ax