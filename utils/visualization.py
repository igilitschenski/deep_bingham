import manstats as ms
import numpy as np
import quaternion

import matplotlib.pylab as plab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


def plot_pose_bingham(bd_param_m, bd_param_z):
    """
    Plots an uncertain orientation given as a 4d bingham.

    Arguments:
          bd_param_m, bd_param_z: The parameters of the Bingham distribution.
    """
    # PART 0: Configuration
    # Number of samples per sampled angular axis.
    num_samples_axis = 100

    # Number of sampled orientations.
    num_orientation_samples = 200

    # Bandwidth for the von Mises kernels used.
    bandwidth = 50.

    # PART 1: Sample poses.
    bd = ms.BinghamDistribution(bd_param_m, bd_param_z)
    orientations = bd.random_samples(num_orientation_samples)

    mean_idx = 0
    means = np.zeros([num_orientation_samples * 3, 3])
    orientations = quaternion.from_float_array(orientations)
    for orientation in orientations:
        r_mat = quaternion.as_rotation_matrix(orientation)
        means[mean_idx, :] = r_mat[:, 0]
        means[mean_idx+1, :] = r_mat[:, 1]
        means[mean_idx+2, :] = r_mat[:, 2]
        mean_idx += 3

    # PART 2: Generate values on the heatmap.
    # Create a meshgrid sampling over the sphere.
    phi = np.linspace(0, np.pi, num_samples_axis)
    theta = np.linspace(0, 2 * np.pi, num_samples_axis)
    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    x = np.reshape(np.sin(phi) * np.cos(theta), num_samples_axis**2)
    y = np.reshape(np.sin(phi) * np.sin(theta), num_samples_axis**2)
    z = np.reshape(np.cos(phi), num_samples_axis**2)
    sphere_points = np.stack([x, y, z]).transpose()
    intensities = _vmf_kernel(sphere_points, means, bandwidth)

    intensities = np.reshape(intensities, [num_samples_axis, num_samples_axis])
    x = np.reshape(x, [num_samples_axis, num_samples_axis])
    y = np.reshape(y, [num_samples_axis, num_samples_axis])
    z = np.reshape(z, [num_samples_axis, num_samples_axis])

    fmax, fmin = intensities.max(), intensities.min()
    intensities = (intensities - fmin) / (fmax - fmin)

    # PART 3: Plot the heatmap.
    fig = plt.figure(figsize=plt.figaspect(1.))
    base_coordinates = quaternion.as_rotation_matrix(
        quaternion.from_float_array(bd.mode)).transpose()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    _plot_coordinate_axes(base_coordinates)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Choose colormap and make it transparent.
    cmap = plab.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    facecolors=my_cmap(intensities))

    # Turn off the axis planes
    ax.view_init()
    ax.azim = 0
    ax.elev = 0
    ax.set_axis_off()
    plt.show()


def _plot_coordinate_axes(coordinates):
    zeros = np.zeros(3)

    x, y, z = zip(zeros, coordinates[0])
    plt.plot(x, y, z, '-k', linewidth=3)

    x, y, z = zip(zeros, coordinates[1])
    plt.plot(x, y, z, '-k', linewidth=3)

    x, y, z = zip(zeros, coordinates[2])
    plt.plot(x, y, z, '-k', linewidth=3)


def _vmf_kernel(points, means, bandwidth):
    # Evaluates a von Mises-Fisher mixture type kernel on given inputs.
    num_points = points.shape[0]
    result = np.zeros(num_points)
    for cur_mean in means:
        # Use of np.einsum for optimizing dot product computation
        # performance. Based on approach presented in:
        # https://stackoverflow.com/a/15622926/812127
        result += \
            np.exp(bandwidth * np.einsum(
                'ij,ij->i',
                np.repeat(np.expand_dims(cur_mean, 0), num_points, axis=0),
                points))

    return result