import numpy as np
from tqdm import tqdm
import GeoJax
import Neurosetta as nr
from scipy.stats import chi2
from matplotlib.collections import LineCollection

def Semi_axis_length(N_all, confidence = 0.9973, return_evecs = True, return_2D = True):
    # initialise empty array to put data in 
    axis_len_data = np.zeros((N_all.graph.num_vertices(), 3))
    evec_data = np.zeros_like(axis_len_data)
    # iterate over neurons
    for v in tqdm(N_all.graph.iter_vertices()):
        coords = nr.g_vert_coords(N_all.graph.vp['Neurons'][v])
        coords = coords * 1e-3
        evals, evecs = GeoJax.coord_eig_decomp(coords, robust = True, center = True, PCA = False, transpose = True)
        axis_len_data[v] = evals
        # keep only the first eigenvector
        evec_data[v] = evecs[0]

    # Chi_square value for 3 Dof (99.73% confidence/3-sigma)
    chi2_val = chi2.ppf(confidence, df = 3) 
    # scaled eigenvalues
    scaled_data = np.sqrt(axis_len_data * chi2_val)

    if return_2D:
        scaled_data = scaled_data[:,[0,1]]
    if return_evecs:
        return scaled_data, evec_data
    else:
        return scaled_data
    

def asymmetric_mad(data):
    med = np.median(data)
    above = data[data >= med]
    below = data[data <= med]
    mad_above = np.median(above - med)
    mad_below = np.median(med - below)
    return mad_below, mad_above


def average_ellipse(semi_axes_array, method="asym_mad", resolution=360):
    """
    Plot the average ellipse with optional variability band.

    Args:
        semi_axes_array: (n_samples, 2) np.ndarray of [major, minor] semi-axis lengths
        method: 'std', 'iqr', or 'asym_mad'
        resolution: number of points to draw ellipses (higher = smoother)
    """
    major_lengths = semi_axes_array[:, 0]
    minor_lengths = semi_axes_array[:, 1]

    center = np.median(semi_axes_array, axis=0)

    if method == "std":
        spread = np.std(semi_axes_array, axis=0)
        spread_major = spread[0]
        spread_minor = spread[1]
        spread_below_major = spread_above_major = spread_major
        spread_below_minor = spread_above_minor = spread_minor
    elif method == "iqr":
        q75 = np.percentile(semi_axes_array, 75, axis=0)
        q25 = np.percentile(semi_axes_array, 25, axis=0)
        iqr = (q75 - q25) / 2
        spread_major = iqr[0]
        spread_minor = iqr[1]
        spread_below_major = spread_above_major = spread_major
        spread_below_minor = spread_above_minor = spread_minor
    elif method == "asym_mad":
        spread_below_major, spread_above_major = asymmetric_mad(major_lengths)
        spread_below_minor, spread_above_minor = asymmetric_mad(minor_lengths)
    else:
        raise ValueError("Invalid method. Choose 'std', 'iqr', or 'asym_mad'.")

    # Generate angles
    theta = np.linspace(0, 2 * np.pi, resolution)

    # Outer ellipse (median + spread_above)
    outer_x = (center[0] + spread_above_major) * np.cos(theta)
    outer_y = (center[1] + spread_above_minor) * np.sin(theta)

    # Inner ellipse (median - spread_below)
    inner_x = (center[0] - spread_below_major) * np.cos(theta)
    inner_y = (center[1] - spread_below_minor) * np.sin(theta)

    return center, theta, outer_x, outer_y, inner_x, inner_y


def single_ellipse_parameters(coords_2d, confidence=0.9973, resolution=360):
    """
    Calculate ellipse parameters for a 2D point cloud.
    
    Args:
        coords_2d: (n_samples, 2) np.ndarray
        confidence: chi-squared confidence level (default 99.73% ~ 3 sigma)
        resolution: number of points for drawing ellipse
        
    Returns:
        center (x,y), major semi-axis, minor semi-axis, rotation angle (degrees), theta array
    """
    center = np.mean(coords_2d, axis=0)
    cov = np.cov(coords_2d, rowvar=False)
    chi2_val = chi2.ppf(confidence, df=2)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Calculate axis lengths scaled by chi2
    major = np.sqrt(eigvals[0] * chi2_val)
    minor = np.sqrt(eigvals[1] * chi2_val)

    # Calculate rotation angle
    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))

    # Theta for plotting
    theta = np.linspace(0, 2*np.pi, resolution)

    return center, major, minor, angle, theta

def plot_single_ellipse(ax, center, major, minor, angle, theta, center_point=(0,0), plot_center = False, lw = 2, flip_axes=False, color='blue'):
    """
    Plot a single ellipse using calculated parameters.
    
    Args:
        ax: matplotlib axis
        center: (x, y) center of the ellipse
        major: major semi-axis length
        minor: minor semi-axis length
        angle: rotation angle (degrees)
        theta: angles for drawing the ellipse
        center_point: (offset_x, offset_y) optional shift
        flip_axes: if True, swap x and y
        color: ellipse line color
    """
    # Parametric ellipse
    ellipse_x = major * np.cos(theta)
    ellipse_y = minor * np.sin(theta)

    # Rotate ellipse
    phi = np.deg2rad(angle)
    x_rot = ellipse_x * np.cos(phi) - ellipse_y * np.sin(phi)
    y_rot = ellipse_x * np.sin(phi) + ellipse_y * np.cos(phi)

    # Translate to center + offset
    cx, cy = center_point
    x_final = x_rot + center[0] + cx
    y_final = y_rot + center[1] + cy

    if flip_axes:
        x_final, y_final = y_final, x_final

    # Plot
    ax.plot(x_final, y_final, color=color, lw=lw)
    if plot_center:
        ax.scatter(center[0] + cx, center[1] + cy, color=color)

    ax.set_aspect('equal')


def plot_Ellipse_population(
    ax,
    center,
    theta,
    outer_x,
    outer_y,
    inner_x,
    inner_y,
    colour,
    label,
    plot_center=False,
    fill_alpha=0.5,
    lw=2,
    flip_axes=False,
    center_point=(0, 0),
):
    if flip_axes:
        # Swap x and y
        outer_x, outer_y = outer_y, outer_x
        inner_x, inner_y = inner_y, inner_x

    # Apply center_point shift to outer and inner ellipses
    cx, cy = center_point
    outer_x_shifted = outer_x + cx
    outer_y_shifted = outer_y + cy
    inner_x_shifted = inner_x + cx
    inner_y_shifted = inner_y + cy

    # plot outer elipse
    ax.fill(outer_x_shifted, outer_y_shifted, color=colour, alpha=fill_alpha, zorder=1)
    # plot inner elipse
    ax.fill(inner_x_shifted, inner_y_shifted, color="white", alpha=1.0, zorder=2)

    # Plot average ellipse (sharp line)
    avg_x = center[0] * np.cos(theta)
    avg_y = center[1] * np.sin(theta)

    if flip_axes:
        avg_x, avg_y = avg_y, avg_x

    avg_x += cx
    avg_y += cy

    ax.plot(avg_x, avg_y, color=colour, label=label, lw=lw, zorder=3)

    if plot_center:
        ax.scatter(cx, cy, color=colour, zorder=4)

    ax.set_aspect("equal")


def diamond_layout_positions(n, offset=1.0):
    """
    Generate (x, y) center positions arranged in a diamond shape for n ellipses.

    Args:
        n: number of ellipses (up to 5 for basic diamond: top, left, right, bottom, center)
        offset: spacing from center
    Returns:
        List of (x, y) tuples
    """
    if n > 5:
        raise ValueError(
            "Diamond layout currently supports up to 5 ellipses (top, left, right, bottom, center)."
        )

    positions = []

    order = [
        (0, offset),
        (offset, 0),
        (0, -offset),
        (-offset, 0),
        (0, 0),
    ]  # top, right, bottom, left, center

    for i in range(n):
        positions.append(order[i])

    return positions


def add_scale_bar(
    ax, length=5, label="5 μm", location="lower right", offset=0.1, linewidth=2, fontsize = 10
):
    """
    Add a scale bar to the plot.

    Args:
        ax: matplotlib axis
        length: length of the scale bar in plot units
        label: label text
        location: 'lower right', 'lower left', etc.
        offset: fraction of axis width/height to offset from edges
        linewidth: thickness of the scale bar
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    if location == "lower right":
        start_x = xlim[1] - offset * x_span - length
        start_y = ylim[0] + offset * y_span
    elif location == "lower left":
        start_x = xlim[0] + offset * x_span
        start_y = ylim[0] + offset * y_span
    else:
        raise ValueError("Unsupported location")

    # Plot the scale bar
    ax.plot(
        [start_x, start_x + length], [start_y, start_y], color="black", lw=linewidth
    )

    # Add the label
    ax.text(
        start_x + length / 2,
        start_y - 0.02 * y_span,
        label,
        ha="center",
        va="top",
        fontsize=fontsize,
    )

def plot_2d_N(ax, N, microns = True, ax_pad = 1, colour = 'k', lw = 2, alpha = 0.9):
    # get starts and stops
    starts, stops = nr.plotting._vd_tree_st_end(N)
    # reduce to 2D (remove z axis)
    starts = starts[:,[0,1]]
    stops = stops[:,[0,1]]

    if microns:
        starts = starts * 0.001
        stops = stops * 0.001

    # Stack start and end into a segments array
    segments = np.stack([starts, stops], axis=1)

    # Create LineCollection
    lc = LineCollection(segments, colors=colour, linewidths=lw, alpha=alpha)
    # add to axis
    ax.add_collection(lc)
    # adjust limits
    all_pts = np.vstack((starts, stops))
    ax.set_xlim(all_pts[:,0].min() - ax_pad, all_pts[:,0].max() + ax_pad)
    ax.set_ylim(all_pts[:,1].min() - ax_pad, all_pts[:,1].max() + ax_pad)

    ax.set_aspect('equal')
