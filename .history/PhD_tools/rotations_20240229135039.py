import numpy as np


def rotation_matrix_3D(theta1, theta2, theta3, order="xyz"):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        order = rotation order of x,y,z: e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == "xzx":
        matrix = np.array(
            [
                [c2, -c3 * s2, s2 * s3],
                [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "xyx":
        matrix = np.array(
            [
                [c2, s2 * s3, c3 * s2],
                [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yxy":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                [s2 * s3, c2, -c3 * s2],
                [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yzy":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                [c3 * s2, c2, s2 * s3],
                [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "zyz":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                [-c3 * s2, s2 * s3, c2],
            ]
        )
    elif order == "zxz":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                [s2 * s3, c3 * s2, c2],
            ]
        )
    elif order == "xyz":
        matrix = np.array(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
            ]
        )
    elif order == "xzy":
        matrix = np.array(
            [
                [c2 * c3, -s2, c2 * s3],
                [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
            ]
        )
    elif order == "yxz":
        matrix = np.array(
            [
                [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                [c2 * s3, c2 * c3, -s2],
                [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
            ]
        )
    elif order == "yzx":
        matrix = np.array(
            [
                [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                [s2, c2 * c3, -c2 * s3],
                [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
            ]
        )
    elif order == "zyx":
        matrix = np.array(
            [
                [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                [-s2, c2 * s3, c2 * c3],
            ]
        )
    elif order == "zxy":
        matrix = np.array(
            [
                [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                [-c2 * s3, s2, c2 * c3],
            ]
        )

    return matrix


def coords_Eig(
    Coords,
    center=False,
    PCA=False,
):
    """
    Performs Eigen Decomposition on a given array of coordinates. This is done by calculating the covariance matrix of the coordinates array, then the eigenvectors and values of this coordinate matrix.
    Parameters
    ----------

    Coords:         np.array
        Coordinate array from which to calculate eigenvectors and values. This should be in the form n x d, where each row is an observation, and column a dimension.
    center:         Bool
        Whether or not to mean center the data before computing
    PCA:            Bool
        Whether or not to return eigenvalues as a fraction of the sum of all eigenvalues, as in PCA, showing variance explained.
    Returns
    -------
    evals:          np.array
        eigenvalues, ordered from largest to smallest
    evects:         list
        List of np.arrays, each of witch is the eigenvector corresponding to the descending order of eigenvalues

    """

    # check dimensions of input
    if 3 not in Coords.shape:
        raise AttributeError("Input coordinates are not 3 dimensional")
    else:
        if Coords.shape[0] != 3:
            Coords = Coords.T

    # mean center the data, if we want
    if center == True:
        for i in range(Coords.shape[0]):
            Coords[i] -= np.mean(Coords[i])

    cov_mat = np.cov(Coords)
    evals, evects = np.linalg.eig(cov_mat)
    # sort largest to smallest
    sort_inds = np.argsort(evals)[::-1]

    if PCA == True:
        evals /= sum(evals)

    evects = [evects[:, i] for i in sort_inds]

    return evals[sort_inds], evects


def eig_axis_eulers(evects):
    """
    Given a list of eigenvector as returned by coords_Eig, return Euler angles needed to align The first eigenvector with the y-axis, second with the x-axis, and third with the z-axis
    """
    # Yaw
    theta1 = np.rad2deg(np.arctan(evects[0][0] / evects[0][1]))
    # pitch
    theta2 = np.rad2deg(np.arctan(evects[1][2] / evects[1][0]))
    # roll
    theta3 = np.rad2deg(np.arctan(evects[2][1] / evects[2][2]))

    return theta1, theta2, theta3


def snap_to_axis(coords, error_tol=0.0000001, return_theta=False):
    """
    Given a set of 3D coordinates, rotates the coordinates so the Eigenvectors align with the original coordinate system axis. This is done so the first Eigenvector (corresponding to the highest eigenvalue) aligns with the y-axis, the second to the x-axis, and the third to the z-axis. Rotation is done in 'zyx' order, Rotating first around the z-axis to align the first eigenvector to the y-axis (this is Yaw), Second around the y-axis to align the second eigenvector to the x-axis (Pitch), and finally around the x-axis to align the final eigenvector to the z-axis (roll).

    Parameters
    ----------
    coords:         np.array
        dimensions by observations np.array with coordinates. Function can only accept 3D coordinates currently
    error_tol:      float
        Some error around how closely the eigenvectors can align to the image axis seems to be introduced (at this stage, it is unclear why this is the case...). The error_tol parameter sets a threshold where by the rotation will be interatively re-rotated, new eigenvectors calculated, and euler angles calculated, until the euler angles are less than this threshold.
    return_theta:   Bool
        If True, final euler angles are returned, which will be less than error_tol.
    Returns
    -------
    r_coords:       np.array
        Rotated coordinate array, the same shape as the input coordinate array
    thetas:         list
        list of euler angles of the final rotation, which will be less than error_tol. Angles are ordered in the order of rotations.



    """
    # make sure coords are the right shape:
    if 3 in coords.shape:
        if coords.shape[0] != 3:
            coords = coords.T
    else:
        raise AttributeError("Input coordinates are not 3 dimensional")
    ### Rotation - Yaw, Pitch, and Roll
    evals, evects = coords_Eig(coords)

    theta1, theta2, theta3 = eig_axis_eulers(evects)

    R = rotation_matrix_3D(theta1, theta2, theta3, order="zyx")
    r_coords = R @ coords

    # Check and correct for error
    # get "final" angles
    evals, evects = coords_Eig(r_coords)

    theta1, theta2, theta3 = eig_axis_eulers(evects)

    while (
        (abs(theta1) > error_tol)
        and (abs(theta2) > error_tol)
        and (abs(theta3) > error_tol)
    ):

        evals, evects = coords_Eig(r_coords)
        # pitch
        theta1, theta2, theta3 = eig_axis_eulers(evects)

        R = rotation_matrix_3D(theta1, theta2, theta3, order="zyx")
        r_coords = R @ r_coords

    if return_theta == False:
        return r_coords
    else:
        return r_coords, [theta1, theta2, theta3]
