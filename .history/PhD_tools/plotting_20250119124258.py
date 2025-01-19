import matplotlib.pyplot as plt

# Define A4 size in cm
A4_width_cm = 21.0
A4_height_cm = 29.7

# Margins (top and side)
margin_top = 2
margin_bottom =  


# Convert cm to inches for Matplotlib
cm_to_inch = 1 / 2.54
A4_width_in = A4_width_cm * cm_to_inch
A4_height_in = A4_height_cm * cm_to_inch

def A4_figure():
    return plt.figure(figsize=(A4_width_in, A4_height_in))

def fig_add_axes(fig,x,y,height,width):
    # Convert scatter plot dimensions to relative figure coordinates
    scatter_x_rel = x / A4_width_cm
    scatter_y_rel = y / A4_height_cm
    scatter_width_rel = width / A4_width_cm
    scatter_height_rel = height / A4_height_cm

    ax = fig.add_axes([scatter_x_rel, scatter_y_rel, scatter_width_rel, scatter_height_rel])

    return ax