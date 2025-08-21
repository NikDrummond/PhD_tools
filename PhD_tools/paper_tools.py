import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import linear_sum_assignment
from numba import jit
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.lines import Line2D


Types = np.array(["T4", "T5"])
Type_colours = np.array(["#17becf", "#ff7f0e"])
Subtypes = np.array(["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])
Subtype_colours = np.array(
    [
        "#1f77b4",
        "#9edae5",
        "#98df8a",
        "#bcbd22",
        "#d62728",
        "#ffbb78",
        "#ff9896",
        "#9467bd",
    ]
)


def fit_mixed_regression(x, y, group, ID, ci=True):
    """
    Fit a mixed-effects regression (y ~ x * group, random intercepts for ID)
    and return parameters for plotting per group.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    group : array-like of str
        Group labels for each observation.
    ID : array-like of int
        Subject identifiers (repeated measures).
    ci : bool
        Whether to compute confidence intervals (approximate via normal errors).

    Returns
    -------
    dict with keys:
        'group_fits': dict mapping group -> {x_fit, y_fit, slope, intercept, ci_low, ci_high}
        'model': fitted statsmodels MixedLMResults object
    """

    # Build dataframe for statsmodels
    df = pd.DataFrame(
        {
            "x": np.asarray(x),
            "y": np.asarray(y),
            "group": np.asarray(group),
            "ID": np.asarray(ID),
        }
    )

    # Fit mixed-effects model: random intercepts for ID, fixed effects of x and group
    model = smf.ols("y ~ x * group", df) 
    result = model.fit(reml=True, cov_type="HC3")

    # Generate fitted lines per group
    group_fits = {}
    x_grid = np.linspace(df["x"].min(), df["x"].max(), 200)

    for g in df["group"].unique():
        newdata = pd.DataFrame(
            {
                "x": x_grid,
                "group": g,
                "ID": df["ID"].iloc[
                    0
                ],  # dummy ID (not used in fixed effects prediction)
            }
        )

        # Manually compute fitted values from fixed effects
        coef = result.params
        intercept = coef["Intercept"] + coef.get(f"group[T.{g}]", 0.0)
        slope = coef["x"] + coef.get(f"x:group[T.{g}]", 0.0)
        y_fit = intercept + slope * x_grid

        fit_dict = {
            "x_fit": x_grid,
            "y_fit": y_fit,
            "slope": slope,
            "intercept": intercept,
            "label": f"{g}: slope={slope:.3f}",
        }

        if ci:
            # Approximate CI for mean prediction using standard errors of fixed effects
            cov = result.cov_params()
            se_pred = []
            for xv in x_grid:
                # design row for fixed effects
                row = [1.0, xv]
                if f"group[T.{g}]" in coef.index:
                    row.append(1.0)
                if f"x:group[T.{g}]" in coef.index:
                    row.append(xv)
                row = np.array(row)
                se = np.sqrt(row @ cov.values[: len(row), : len(row)] @ row)
                se_pred.append(se)
            se_pred = np.array(se_pred)
            ci_low = y_fit - 1.96 * se_pred
            ci_high = y_fit + 1.96 * se_pred
            fit_dict.update({"ci_low": ci_low, "ci_high": ci_high})

        group_fits[g] = fit_dict

    return {"group_fits": group_fits, "model": result}


def depth_regression(ax, df, group, pkwargs=dict(), lkwargs=dict()):

    if group == "T4":
        c_inds = np.arange(4)
    else:
        c_inds = np.arange(4, 8)

    # subgroups = [group + i for i in ["a", "b", "c", "d"]]
    subgroup_colours = Subtype_colours[c_inds]
    sub_df = df.loc[
        (df.Type == group)
        & (df.Normalized_root_Distance != 0)
        & (df.Normalized_root_Distance != 1),
        ["ID", "Subtype", "Layer_Depth", "Normalized_root_Distance"],
    ]

    x = sub_df.Normalized_root_Distance.values
    y = sub_df.Layer_Depth
    group = sub_df.Subtype.values
    ID = sub_df.ID.values
    results = fit_mixed_regression(x, y, group, ID)

    # scatter of raw data
    ax.scatter(x, y, **pkwargs)

    # fitted lines + CIs
    i = 0
    for g, fit in results["group_fits"].items():
        c = subgroup_colours[i]
        i += 1
        ax.plot(fit["x_fit"], fit["y_fit"], label=fit["label"], color=c, **lkwargs)
        if "ci_low" in fit:
            ax.fill_between(
                fit["x_fit"], fit["ci_low"], fit["ci_high"], alpha=0.2, color=c
            )

    ax.legend(loc="upper right", frameon=False)


@jit(nopython=True)
def squared_distance_matrix_with_threshold(coords, threshold_sq):
    """
    Compute squared distance matrix with threshold using Numba for speed.

    Parameters:
    -----------
    coords : np.array
        Array of shape (n, 3) containing 3D coordinates
    threshold_sq : float
        Squared distance threshold - pairs beyond this are set to infinity

    Returns:
    --------
    distances : np.array
        Squared distance matrix with threshold applied
    """
    n = coords.shape[0]
    distances = np.full((n, n), np.inf, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(3):
                diff = coords[i, k] - coords[j, k]
                dist_sq += diff * diff

            if dist_sq <= threshold_sq:
                distances[i, j] = dist_sq
                distances[j, i] = dist_sq

    return distances


def find_optimal_assignment(coords, distance_threshold=None):
    """
    Find optimal pairwise assignment minimizing total Euclidean distance.
    Uses squared distances and optional distance threshold for speed.

    Parameters:
    -----------
    coords : np.array
        Array of shape (n, 3) containing 3D coordinates
    distance_threshold : float, optional
        Maximum distance for pairing. Pairs beyond this distance are not considered.
        If None, all pairs are considered.

    Returns:
    --------
    pairs : list of tuples
        List of (index1, index2) pairs representing optimal assignment
    distances : list of floats
        Euclidean distances corresponding to each pair
    assignments : np.array
        Array of length n where assignments[i] gives the index of point i's partner,
        or -1 if point i is unassigned

    Example:
    --------
    >>> coords = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 6, 6]])
    >>> pairs, distances, assignments = find_optimal_assignment(coords, distance_threshold=3.0)
    >>> print(f"Pairs: {pairs}")
    >>> print(f"Distances: {distances}")
    >>> print(f"Assignments: {assignments}")
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)

    # Handle edge cases
    if n <= 1:
        return [], [], np.full(n, -1, dtype=int)

    # Compute squared distance matrix with threshold
    if distance_threshold is not None:
        threshold_sq = distance_threshold**2
        cost_matrix = squared_distance_matrix_with_threshold(coords, threshold_sq)
    else:
        # No threshold - compute all squared distances
        threshold_sq = np.inf
        cost_matrix = squared_distance_matrix_with_threshold(coords, threshold_sq)

    # Prevent self-assignment
    np.fill_diagonal(cost_matrix, np.inf)

    # Solve assignment problem using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract valid pairs and compute actual Euclidean distances
    pairs = []
    distances = []
    assignments = np.full(n, -1, dtype=int)
    used_points = set()

    for i, j in zip(row_indices, col_indices):
        # Check if this is a valid assignment
        if (
            i != j
            and i not in used_points
            and j not in used_points
            and not np.isinf(cost_matrix[i, j])
        ):

            # Add the pair
            pairs.append((i, j))

            # Compute actual Euclidean distance
            euclidean_dist = np.sqrt(cost_matrix[i, j])
            distances.append(euclidean_dist)

            # Update assignments
            assignments[i] = j
            assignments[j] = i

            # Mark points as used
            used_points.add(i)
            used_points.add(j)

    return pairs, distances, assignments


def classify_pairs(pairs, groups):
    """
    Classify each pair into 'Opposite', 'Same', or 'Orthogonal' categories.

    Parameters:
    -----------
    pairs : list of tuples
        List of (index1, index2) pairs from optimal assignment
    groups : list or np.array
        Group labels for each point (e.g., ['a', 'b', 'c', 'd'])

    Returns:
    --------
    pair_classifications : list
        List of classification labels for each pair
    classification_counts : dict
        Count of each classification type
    pair_details : list of dicts
        Detailed information for each pair
    """
    groups = np.array(groups)

    # Define the classification rules
    opposite_pairs = {("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")}

    same_pairs = {("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")}

    orthogonal_pairs = {
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "a"),
        ("c", "b"),
        ("d", "a"),
        ("d", "b"),
    }

    pair_classifications = []
    pair_details = []

    for i, j in pairs:
        group_i = groups[i]
        group_j = groups[j]
        pair_tuple = (group_i, group_j)

        # Classify the pair
        if pair_tuple in opposite_pairs:
            classification = "Opposite"
        elif pair_tuple in same_pairs:
            classification = "Same"
        elif pair_tuple in orthogonal_pairs:
            classification = "Orthogonal"
        else:
            # This shouldn't happen with a,b,c,d groups, but handle gracefully
            classification = "Unknown"

        pair_classifications.append(classification)
        pair_details.append(
            {
                "pair_indices": (i, j),
                "groups": pair_tuple,
                "classification": classification,
            }
        )

    # Count classifications
    classification_counts = Counter(pair_classifications)

    return pair_classifications, classification_counts, pair_details


def calculate_probabilities(classification_counts, total_pairs=None):
    """
    Calculate probabilities for each classification type.

    Parameters:
    -----------
    classification_counts : dict
        Count of each classification type
    total_pairs : int, optional
        Total number of pairs (if different from sum of counts)

    Returns:
    --------
    probabilities : dict
        Probability of each classification type
    """
    if total_pairs is None:
        total_pairs = sum(classification_counts.values())

    if total_pairs == 0:
        return {"Opposite": 0, "Same": 0, "Orthogonal": 0}

    probabilities = {}
    for classification in ["Opposite", "Same", "Orthogonal"]:
        count = classification_counts.get(classification, 0)
        probabilities[classification] = count / total_pairs

    return probabilities


def plot_group_pair_probability_matrix(
    ax, pairs, groups, x0=0, x1=1, title="Group Pair Probability Matrix", cmap="PuRd"
):
    """
    Plot a probability matrix showing the likelihood of group pairings.

    Parameters:
    -----------
    pairs : list of tuples
        List of (index1, index2) pairs from optimal assignment
    groups : list or np.array
        Group labels for each point (e.g., ['a', 'b', 'c', 'd'])
    title : str, optional
        Title for the plot
    """

    groups = np.array(groups)
    group_labels = ["a", "b", "c", "d"]

    # Create frequency matrix
    pair_matrix = np.zeros((4, 4))

    for i, j in pairs:
        group_i = groups[i]
        group_j = groups[j]
        i_idx = group_labels.index(group_i)
        j_idx = group_labels.index(group_j)
        pair_matrix[i_idx, j_idx] += 1

    # Convert to probabilities
    total_pairs = len(pairs)
    if total_pairs > 0:
        prob_matrix = pair_matrix / total_pairs
    else:
        prob_matrix = pair_matrix

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(prob_matrix, cmap=cmap, alpha=0.8, vmin=x0, vmax=x1)

    # Add text annotations with probabilities
    for i in range(4):
        for j in range(4):
            prob = prob_matrix[i, j]
            # Show probability with appropriate precision
            if prob > 0:
                text = f"{prob:.3f}"
            else:
                text = "0.000"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black" if prob < 0.5 else "white",
                fontweight="bold",
                fontsize=12,
            )

    # Set up axes
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_yticklabels(group_labels, fontsize=12)
    ax.set_xlabel("Group J", fontsize=12, fontweight="bold")
    ax.set_ylabel("Group I", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Probability", fontsize=12, fontweight="bold")

    # Add grid for better readability
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1, alpha=0.3)

    return ax, prob_matrix


def plot_classification_probabilities(
    ax, pairs, groups, title="Pair Classification Probabilities"
):
    """
    Plot a bar chart showing probabilities for Opposite, Same, and Orthogonal classifications.

    Parameters:
    -----------
    pairs : list of tuples
        List of (index1, index2) pairs from optimal assignment
    groups : list or np.array
        Group labels for each point (e.g., ['a', 'b', 'c', 'd'])
    title : str, optional
        Title for the plot
    """

    groups = np.array(groups)

    # Define classification rules
    opposite_pairs = {("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")}
    same_pairs = {("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")}
    orthogonal_pairs = {
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "a"),
        ("c", "b"),
        ("d", "a"),
        ("d", "b"),
    }

    # Classify pairs
    classifications = []
    for i, j in pairs:
        group_i = groups[i]
        group_j = groups[j]
        pair_tuple = (group_i, group_j)

        if pair_tuple in opposite_pairs:
            classifications.append("Opposite")
        elif pair_tuple in same_pairs:
            classifications.append("Same")
        elif pair_tuple in orthogonal_pairs:
            classifications.append("Orthogonal")

    # Count and calculate probabilities
    classification_counts = Counter(classifications)
    total_pairs = len(pairs)

    classification_labels = ["Opposite", "Same", "Orthogonal"]
    probabilities = [
        classification_counts.get(c, 0) / total_pairs if total_pairs > 0 else 0
        for c in classification_labels
    ]

    # Colors: Same as PMFs
    colors = ["#FFB000", "#00C2A0", "#D40078"]

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        classification_labels,
        probabilities,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add probability labels on bars
    for bar, prob in zip(bars, probabilities):
        if prob > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

    # Add chance probability reference lines
    ax.hlines(
        y=0.25,
        xmin=-0.5,
        xmax=1.4,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Chance probability (Same/Opposite)",
    )
    ax.hlines(
        y=0.50,
        xmin=1.5,
        xmax=2.5,
        color="orange",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Chance probability (Orthogonal)",
    )

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)

    # Add text box with summary
    summary_text = f"Total pairs: {total_pairs}\n"
    for i, label in enumerate(classification_labels):
        count = classification_counts.get(label, 0)
        summary_text += f"{label}: {count} ({probabilities[i]:.1%})\n"

    # plt.tight_layout()
    # plt.show()

    return ax, probabilities


def nn_PMFs(ax, group, df, x0, x1, n_bins):
    """The worlds crappiest written function"""

    # get data arrays
    a = df.loc[df.Subtype == group + "a", ["Root_x", "Root_y", "Root_z"]]
    b = df.loc[df.Subtype == group + "b", ["Root_x", "Root_y", "Root_z"]]
    c = df.loc[df.Subtype == group + "c", ["Root_x", "Root_y", "Root_z"]]
    d = df.loc[df.Subtype == group + "d", ["Root_x", "Root_y", "Root_z"]]

    # KD Trees
    a_tree = KDTree(a.values)
    b_tree = KDTree(b.values)
    c_tree = KDTree(c.values)
    d_tree = KDTree(d.values)

    # Opposite Subtype
    # a vs b
    dists, inds = a_tree.query(b.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#FFB000", label="a vs b")
    # b vs a
    dists, inds = b_tree.query(a.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#FFB000", label="b vs a")
    # c vs d
    dists, inds = c_tree.query(d.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#FFB000", label="c vs d", ls="--")
    # d vs c
    dists, inds = d_tree.query(c.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#FFB000", label="d vs c", ls="--")

    # Same Subtype

    # a vs a
    dists, inds = a_tree.query(a.values, k=[2])
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#00C2A0", label="a vs a")
    # b vs b
    dists, inds = b_tree.query(b.values, k=[2])
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#00C2A0", label="b vs b")
    # c vs c
    dists, inds = c_tree.query(c.values, k=[2])
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#00C2A0", label="c vs c", ls="--")
    # d vs d
    dists, inds = d_tree.query(d.values, k=[2])
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#00C2A0", label="d vs d", ls="--")

    # Orthogonal Subtypes - a and b

    # a vs c
    dists, inds = a_tree.query(c.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="a vs c")
    # a vs d
    dists, inds = a_tree.query(d.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="a vs d")
    # b vs c
    dists, inds = b_tree.query(c.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="b vs c")
    # b vs d
    dists, inds = b_tree.query(d.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="b vs d")

    # Orthogonal Subtypes - c and d
    # c vs a
    dists, inds = c_tree.query(a.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="c vs a", ls="--")
    # c vs b
    dists, inds = c_tree.query(b.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="c vs b", ls="--")
    # d vs a
    dists, inds = d_tree.query(a.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="d vs a", ls="--")
    # d vs b
    dists, inds = d_tree.query(b.values, k=1)
    counts, bins = np.histogram(dists, range=(x0, x1), bins=n_bins)
    ax.plot(bins[1:], counts / counts.sum(), c="#D40078", label="d vs b", ls="--")


def depth_PMF(ax, df, lw=3):

    sub_df = df.loc[
        df.Normalized_root_Distance == 0,
        ["Subtype", "Layer_Depth", "Normalized_root_Distance"],
    ]

    for i in range(8):
        t = Subtypes[i]
        c = Subtype_colours[i]
        d = sub_df.loc[sub_df.Subtype == t, "Layer_Depth"].values
        counts, bins = np.histogram(d, bins=26, range=(0, 1))
        counts = counts / counts.sum()
        ax.plot(bins[1:], counts, c=c, label=t, lw=lw)

    sub_df = df.loc[
        df.Normalized_root_Distance != 0,
        ["Subtype", "Layer_Depth", "Normalized_root_Distance"],
    ]

    for i in range(8):
        t = Subtypes[i]
        c = Subtype_colours[i]
        d = sub_df.loc[sub_df.Subtype == t, "Layer_Depth"].values
        counts, bins = np.histogram(d, bins=26, range=(0, 1))
        counts = counts / counts.sum()
        ax.plot(bins[1:], counts, c=c, ls="--", lw=lw)


def add_scale_bar(
    ax, length=5, label="5 μm", location="lower right", offset=0.1, linewidth=2
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
        fontsize=10,
    )


def draw_cross(
    ax, center, length, labels=["A", "B", "C", "D"], arrow_kwargs=dict(), font_size=10
):

    if len(labels) != 4:
        raise ValueError("'labels' must have exactly 4 elements.")

    x0, y0 = center

    # Fixed order: left, right, down, up
    directions = [(-length, 0), (length, 0), (0, -length), (0, length)]

    for (dx, dy), label in zip(directions, labels):
        # Draw arrow
        ax.arrow(
            x0,
            y0,
            dx,
            dy,
            head_width=0.05 * length,
            head_length=0.1 * length,
            fc="k",
            ec="k",
            length_includes_head=True,
            **arrow_kwargs,
        )

        # Place bold label slightly beyond arrow tip
        ax.text(
            x0 + dx * 1.5,
            y0 + dy * 1.5,
            label,
            color="k",
            ha="center",
            va="center",
            fontsize=font_size,
            weight="bold",
        )


def subtype_contour_plot(ax, dictionary):
    for i in range(len(Subtypes)):

        s = Subtypes[i]

        x = dictionary[s]["x"]
        y = dictionary[s]["y"]
        xx = dictionary[s]["xx"]
        yy = dictionary[s]["yy"]
        zz = dictionary[s]["zz"]

        # Plot the contour
        ax.contour(xx, yy, zz, levels=5, cmap="viridis")
        ax.scatter(
            x, y, s=0.3, color="gray", alpha=0.01, rasterized=True
        )  # Optional: plot original points
        ax.scatter(
            dictionary["offsets"][i][0], dictionary["offsets"][i][1], c="r", zorder=500
        )

    ax.set_aspect("equal")

    ax.grid(True)
    ax.set_axis_off()
    add_scale_bar(ax, length=1, label=r"$1 \sigma$")


def create_bootstrap_pmf(d, x0=0, x1=1500, num_bins=25, n_bootstrap=1000):

    # Building observed histogram
    bins = np.linspace(x0, x1, num_bins + 1)
    if x0 == 0:
        half_width = bins[1] / 2
    else:
        half_width = (bins[1] - bins[0]) / 2
    x_values = np.linspace(half_width, x1 - half_width, num_bins)

    # Observed data histogram
    counts, _ = np.histogram(d, range=(x0, x1), bins=num_bins)
    counts = counts / counts.sum()

    # Method 1: Fastest - Single vectorized operation
    n_samples = len(d)
    bootstrap_indices = np.random.choice(
        n_samples, size=(n_bootstrap, n_samples), replace=True
    )
    bootstrap_samples = d[bootstrap_indices]

    # histogram computation
    bootstrap_pmfs = np.array(
        [np.histogram(bootstrap_samples[i], bins=bins)[0] for i in range(n_bootstrap)]
    )
    bootstrap_pmfs = bootstrap_pmfs / bootstrap_pmfs.sum(axis=1, keepdims=True)

    # Calculate 95% confidence intervals
    ci_lower = np.percentile(bootstrap_pmfs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_pmfs, 97.5, axis=0)

    return x_values, counts, ci_lower, ci_upper, bootstrap_pmfs


def create_bootstrap_pmf(d, x0=0, x1=1500, num_bins=25, n_bootstrap=1000):

    # Building observed histogram
    bins = np.linspace(x0, x1, num_bins + 1)

    # Corrected method for finding bin centers
    x_values = (bins[:-1] + bins[1:]) / 2

    # Observed data histogram
    counts, _ = np.histogram(d, range=(x0, x1), bins=num_bins)
    counts = counts / counts.sum()

    # Method 1: Fastest - Single vectorized operation
    n_samples = len(d)
    bootstrap_indices = np.random.choice(
        n_samples, size=(n_bootstrap, n_samples), replace=True
    )
    bootstrap_samples = d[bootstrap_indices]

    # histogram computation
    bootstrap_pmfs = np.array(
        [np.histogram(bootstrap_samples[i], bins=bins)[0] for i in range(n_bootstrap)]
    )
    bootstrap_pmfs = bootstrap_pmfs / bootstrap_pmfs.sum(axis=1, keepdims=True)

    # Calculate 95% confidence intervals
    ci_lower = np.percentile(bootstrap_pmfs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_pmfs, 97.5, axis=0)

    return x_values, counts, ci_lower, ci_upper, bootstrap_pmfs


def point_value_PMF_df(
    ax,
    df,
    DV,
    groups,
    colours,
    x0,
    x1,
    num_bins,
    num_bootstraps,
    line_kwargs=dict(),
    fill_kwargs=dict(),
):

    for i in range(len(groups)):
        t = groups[i]
        c = colours[i]
        # get data
        d = df.loc[df.Type == t, DV].values
        # bootstrap pmf
        x, counts, l, u, pmfs = create_bootstrap_pmf(
            d, x0=x0, x1=x1, num_bins=num_bins, n_bootstrap=num_bootstraps
        )
        # plot
        ax.plot(x, counts, c=c, label=t, **line_kwargs)
        ax.fill_between(x, l, u, color=c, **fill_kwargs)
        ax.set_xlim([x0, x1])

    return ax


def point_value_PMF_1darray(
    ax,
    array,
    colour,
    label,
    x0,
    x1,
    num_bins,
    num_bootstraps,
    line_kwargs=dict(),
    fill_kwargs=dict(),
):

    x, counts, l, u, pmfs = create_bootstrap_pmf(
        array, x0=x0, x1=x1, num_bins=num_bins, n_bootstrap=num_bootstraps
    )
    # plot
    ax.plot(x, counts, c=colour, label=label, **line_kwargs)
    ax.fill_between(x, l, u, color=colour, **fill_kwargs)
    ax.set_xlim([x0, x1])

    return ax


def radial_proportion_positive_negative_PMF(
    ax,
    df,
    given_Subtypes,
    x0,
    x1,
    n_bins,
    n_boots,
    line_kwargs_pos=dict(),
    line_kwargs_neg={"ls": "--"},
    fill_kwargs_pos={"alpha": 0.3},
    fill_kwargs_neg={"alpha": 0.3},
):

    for i in range(len(given_Subtypes)):

        s = Subtypes[np.where(Subtypes == given_Subtypes[i])][0]
        c = Subtype_colours[np.where(Subtypes == given_Subtypes[i])][0]

        subtype_df = df.loc[(df.Subtype == s) & (df.Radial_angle_signed != 0)]

        # proportion o negative and positive values
        result = subtype_df.groupby("ID")["Radial_angle_signed"].agg(
            positive=lambda x: (x > 0).sum() / len(x),
            negative=lambda x: (x < 0).sum() / len(x),
        )

        point_value_PMF_1darray(
            ax,
            result.positive.values,
            colour=c,
            label=s + " Positive",
            x0=x0,
            x1=x1,
            num_bins=n_bins,
            num_bootstraps=n_boots,
            line_kwargs=line_kwargs_pos,
            fill_kwargs=fill_kwargs_pos,
        )
        point_value_PMF_1darray(
            ax,
            result.negative.values,
            colour=c,
            label=s + " Negative",
            x0=x0,
            x1=x1,
            num_bins=n_bins,
            num_bootstraps=100,
            line_kwargs=line_kwargs_neg,
            fill_kwargs=fill_kwargs_neg,
        )

    return ax

def radial_PMF(ax, df, given_Subtypes, n_bins, tick_fontsize = 10):

    bins = np.linspace(-np.pi, np.pi, n_bins)
    x_values = (bins[:-1] + bins[1:])/ 2

    for s in given_Subtypes:

        # get line colour
        c = Subtype_colours[np.where(Subtypes == s)][0]
        
        # Internal
        d_internal = df.loc[(df.Subtype == s) & (df.isExternal == False) & (df.Radial_angle_signed != 0), 'Radial_angle_signed'].values
        counts_internal, _ = np.histogram(d_internal, bins=bins)
        counts_internal = counts_internal / counts_internal.sum()

        # External
        d_external = df.loc[(df.Subtype == s) & (df.isExternal == True) & (df.Radial_angle_signed != 0), 'Radial_angle_signed'].values
        counts_external, _ = np.histogram(d_external, bins=bins)
        counts_external = counts_external / counts_external.sum()

        # Append the first point to the end to close the loop
        closed_x_values = np.append(x_values, x_values[0])
        closed_counts_internal = np.append(counts_internal, counts_internal[0])
        closed_counts_external = np.append(counts_external, counts_external[0])

        # Plotting
        ax.plot(closed_x_values, closed_counts_internal, label=s, c = c)
        ax.plot(closed_x_values, closed_counts_external, c = c, ls='--')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)\

    # Define the positions in radians and the corresponding labels
    tick_locations = [0, np.pi/2, np.pi, -np.pi/2]
    tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pm\pi$', r'$-\frac{\pi}{2}$']

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, fontsize = tick_fontsize)

    # Explicitly set the theta limits to cover the full circle
    ax.set_thetamin(-180)
    ax.set_thetamax(180)

def group_radar_plot(ax, df, dv_col, group_col, groups, colours, plot_type = 'bar',bin_range = (-np.pi, np.pi), bins = 60, alpha = 0.7):
    # will iterate over range(len(curr_types))
    for i in range(len(groups)):

        s = groups[i]
        c = colours[i]
        d = df.loc[df[group_col] == s,dv_col].values
        r, theta = np.histogram(d, range = bin_range, bins = bins)
        r = r / r.sum()
        
        if plot_type == 'bar':
            width = theta[1] - theta[0]
            ax.bar(theta[:-1], r, width=width, color=c, alpha=alpha, label = s)
        elif plot_type == 'radar':
            bin_centers = (theta[:-1] + theta[1:]) / 2
            # make sure we have a closed loop
            bin_centers = np.append(bin_centers, bin_centers[0])
            r = np.append(r, r[0])
            ax.plot(bin_centers, 
                r, 
                color = c, 
                alpha = alpha,
                label = s)

    return ax

# asymmetric MAD from array:
def asymmetric_mad(data):
    """
    Computes the asymmetric median absolute deviation in a vectorized manner.
    """
    # Calculate the median for each column (axis=0)
    median = np.nanmedian(data, axis=0)

    # Calculate absolute deviations from the median
    # Broadcasting makes this efficient: `data` is (M, N), `median` is (1, N)
    diffs = np.abs(data - median)

    # Create boolean masks for values below and above the median
    below = data < median
    above = data > median

    # Use np.where to replace non-relevant deviations with NaN
    # Where `below` is False, place NaN; otherwise, keep the deviation
    diffs_low = np.where(below, diffs, np.nan)
    # Where `above` is False, place NaN; otherwise, keep the deviation
    diffs_high = np.where(above, diffs, np.nan)

    # Calculate the median of the deviations, ignoring the NaNs we just created (supresing nan wearning)
    with np.errstate(invalid = "ignore"):
        mad_low = np.nanmedian(diffs_low, axis=0)
        mad_high = np.nanmedian(diffs_high, axis=0)

    # convert those NaN values to 0
    mad_low[np.isnan(mad_low)] = 0
    mad_high[np.isnan(mad_high)] = 0

    return median, mad_low, mad_high

def repeated_measures_PMF_df(ax, df, DV, groups, colours, x0, x1, num_bins, line_kwargs = dict(), fill_kwargs = dict()):

    # Building observed histogram
    bins = np.linspace(x0, x1, num_bins + 1)
    # Bin centers
    x_values = (bins[:-1] + bins[1:]) / 2

    for i in range(len(groups)):
        s = groups[i]
        c = colours[i]

        sub_df = df.loc[df.Subtype == s]

        # subtypes = df.Subtype.values
        all_ids = sub_df.ID.values
        unique_ids = np.unique(all_ids)
        values = sub_df[DV].values

        # build an array of histograms for some given subtype.
        data = np.zeros((len(unique_ids), num_bins))
        for i in range(len(unique_ids)):

            curr_id = unique_ids[i]
            d = values[np.where(all_ids == curr_id)]

            counts, _ = np.histogram(d, bins = num_bins, range = (x0,x1))
            counts = counts / counts.sum()
            data[i] = counts

        median, mad_low, mad_high = asymmetric_mad(data)

        ax.plot(x_values, median, c = c, label = s, **line_kwargs)
        ax.fill_between(x_values, median - mad_low, median + mad_high, color = c, **fill_kwargs)

def point_ratio_scatter(ax, df, legend_pointsize = 10, legend_fontsize = 10):
    # Upper bound of tree space
    x_line = np.linspace(0, 350, 500)
    y_line = (x_line - 1) / 2

    # Plot the line
    ax.plot(x_line, y_line, color="black", linestyle="--", label = "Upper Bound")

    # Shade the region between y=0 and the line
    ax.fill_between(
        x_line,
        0,
        y_line,
        where=y_line >= 0,        # only shade where line is above x-axis
        color="lightgray",
        alpha=0.7,
        label = "Tree Space"
    )

    # for s in pt.Subtypes:
    #     sub_df = df_point.loc[df_point.Subtype == s]

    x = df.Vertices_numbers.values
    y = df.Branch_number.values

    scatter = ax.scatter(x,y, s = 2, c = '#D97C7C', alpha = 0.3, label = 'Dendites')

    # Create a proxy artist for the legend with custom size
    legend_scatter = Line2D([0], [0], marker='o', color='w', label='Dendrites',
                            markerfacecolor='#D97C7C', markersize=legend_pointsize, alpha=0.7)

    ax.legend(handles=[legend_scatter] + ax.get_legend_handles_labels()[0][:-1],
            frameon=False, loc='upper left', fontsize = legend_fontsize)

    ax.set_xlabel("Total Number of Nodes")
    ax.set_ylabel("Total Number of Branching Nodes")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_aspect('equal')

def error_plot(ax, df, colours, DV, x0, x1, num_bins = None, given_subtypes = ['T4','T5'],offsets = [-0.1,0.1]):

    if num_bins is None:
        num_bins = x1 + 1

    # Building observed histogram
    bins = np.linspace(x0, x1, num_bins + 1)
    # Corrected method for finding bin centers
    x_values = (bins[:-1] + bins[1:]) / 2

    for i in range(len(given_subtypes)):
        s = given_subtypes[i]
        c = colours[i]
        x = x_values + offsets[i]
        sub_df = df.loc[df.Type == s]
        # use numpy for indexing, not pandas
        ids = sub_df.ID.values
        unique_ids = np.unique(ids)
        values = sub_df[DV].values

        data = np.zeros((len(unique_ids),num_bins))

        for j in range(len(unique_ids)):
            curr_id = unique_ids[j]
            d = values[np.where(ids == curr_id)]
            counts, _ = np.histogram(d, range = (x0,x1), bins = num_bins)
            counts = counts / counts.sum()
            data[j] = counts
        
        median, l, u = asymmetric_mad(data)
        ax.errorbar(x, median, yerr = [l,u], fmt = 'o', label = s, color = c)