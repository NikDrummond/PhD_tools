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


def A4_figure(
    A4_width_cm=A4_width_cm,
    A4_height_cm=A4_height_cm,
    margin_top=margin_top,
    margin_bottom=margin_bottom,
    margin_left=margin_left,
    margin_right=margin_right,
):

    # get figure size
    # figure width subtract margin
    fig_width = A4_width_cm - margin_left - margin_right
    # height subtract margin
    fig_height = A4_height_cm - margin_top - margin_bottom
    # convert to inches
    fig_width_in = fig_width * cm_to_inch
    fig_height_in = fig_height * cm_to_inch

    # make figure height and width global
    global fig_width
    


    return plt.figure(figsize=(fig_width_in, fig_height_in))


def fig_add_axes(fig, x, y, height, width):

    # make sure we are in bounds
    assert x >= 0, "x position must be >= 0"
    assert y >= 0, "y position must be >= 0"
    assert y + height <= fig_height, "Top of ax out of bounds"
    assert x + width <= fig_width, "Right of ax out of bounds"

    # Convert scatter plot dimensions to relative figure coordinates
    scatter_x_rel = x / fig_width
    scatter_y_rel = y / fig_height
    scatter_width_rel = width / fig_width
    scatter_height_rel = height / fig_height

    ax = fig.add_axes(
        [scatter_x_rel, scatter_y_rel, scatter_width_rel, scatter_height_rel]
    )

    return ax
