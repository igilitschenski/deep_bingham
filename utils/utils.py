""" Utilities for learning pipeline."""
from __future__ import print_function
import copy
import dill
import hashlib
import itertools
import bingham_distribution as ms
import math
import numpy as np
import os
import scipy
import scipy.integrate as integrate
import scipy.special
import sys
import torch

from pathos.multiprocessing import ProcessingPool as Pool

def convert_euler_to_quaternion(roll, yaw, pitch):
    """Converts roll, yaw, pitch to a quaternion.
    """

    # roll (z), yaw (y), pitch (x)

    cy = math.cos(math.radians(roll) * 0.5)
    sy = math.sin(math.radians(roll) * 0.5)

    cp = math.cos(math.radians(yaw) * 0.5)
    sp = math.sin(math.radians(yaw) * 0.5)

    cr = math.cos(math.radians(pitch) * 0.5)
    sr = math.sin(math.radians(pitch) * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    return quat


def radians(degree_tensor):
    """
    Method to convert a torch tensor of angles in degree format to radians.

    Arguments:
        degree_tensor (torch.Tensor): Tensor consisting of angles in degree format.
    
    Returns:
        radian_tensor (torch.Tensor): Tensor consisting of angles in radian format.
    """
    radian_tensor = degree_tensor/180 * math.pi
    return radian_tensor


def generate_coordinates(coords):
    """
    A function that returns all  possible triples of coords

    Parameters:
    coords: a numpy array of coordinates

    Returns:
    x: the first  coordinate of possible triples
    y: the second coordinate of possible triples
    z  the third coordinate of possible triples
    """
    x = coords.reshape(-1, 1).repeat(1, len(coords) * len(coords)).flatten()
    y = coords.reshape(-1, 1).repeat(1, len(coords)).flatten().repeat(len(coords))
    z = coords.reshape(-1, 1).flatten().repeat(len(coords)*len(coords))

    return x, y, z


def ensure_dir_exists(path):
    """ Checks if a directory exists and creates it otherwise. """
    if not os.path.exists(path):
        os.makedirs(path)


def load_lookup_table(path):
    """
    Loads lookup table from dill serialized file.

    Returns a table specific tuple. For the Bingham case, the tuple containins:
        table_type (str):
        options (dict): The options used to generate the lookup table.
        res_tensor (numpy.ndarray): The actual lookup table data.
        coords (numpy.ndarray): Coordinates at which lookup table was evaluated.

    For the von Mises case, it contains:
        options (dict): The options used to generate the lookup table.
        res_tensor (numpy.ndarray): The actual lookup table data.
    """
    assert os.path.exists(path), "Lookup table file not found."
    with open(path, "rb") as dillfile:
        return dill.load(dillfile)


def eaad_von_mises(kappas, integral_options=None):
    """ Expected Absolute Angular Deviation of Bingham Random Vector

    Arguments:
        kappas: Von Mises kappa parameters for roll, pitch, yaw.
        integral_options: Options to pass on to the scipy integrator for
            computing the eaad and the bingham normalization constant.
    """
    def aad(quat_a, quat_b):
        acos_val = np.arccos(np.abs(np.dot(quat_a, quat_b)))
        diff_ang = 2.0 * acos_val
        return diff_ang

    if integral_options is None:
        integral_options = {"epsrel": 1e-2, "epsabs": 1e-2}

    param_mu = np.array([0., 0., 0.])  # radians
    quat_mu = convert_euler_to_quaternion(
        math.degrees(param_mu[0]), math.degrees(param_mu[1]),
        math.degrees(param_mu[2])
    )
    param_kappa = kappas

    direct_norm_const = 8.0 * (np.pi ** 3) \
        * scipy.special.iv(0, param_kappa[0]) \
        * scipy.special.iv(0, param_kappa[1]) \
        * scipy.special.iv(0, param_kappa[2])

    def integrand_aad(phi1, phi2, phi3):
        return np.exp(param_kappa[0] * np.cos(phi1)) \
               * np.exp(param_kappa[1] * np.cos(phi2)) \
               * np.exp(param_kappa[2] * np.cos(phi3)) \
               * aad(quat_mu,
                     convert_euler_to_quaternion(
                         math.degrees(phi1), math.degrees(phi2),
                         math.degrees(phi3)
                     ))

    eaad_int = integrate.tplquad(
        integrand_aad,
        0.0, 2.0 * np.pi,  # phi3
        lambda x: 0.0, lambda x:  2. * np.pi,  # phi2
        lambda x, y: 0.0, lambda x, y: 2. * np.pi,  # phi1
        **integral_options
    )

    return eaad_int[0]/direct_norm_const


def eaad_bingham(bingham_z, integral_options=None):
    """ Expected Absolute Angular Deviation of Bingham Random Vector

    Arguments:
        bingham_z: Bingham dispersion parameter in the format expected by the
            manstats BinghamDistribution class.
        integral_options: Options to pass on to the scipy integrator for
            computing the eaad and the bingham normalization constant.
    """

    def aad(quat_a, quat_b):
        # acos_val = np.arccos(np.dot(quat_a, quat_b))
        # diff_ang = 2 * np.min([acos_val, np.pi - acos_val])
        acos_val = np.arccos(np.abs(np.dot(quat_a, quat_b)))
        diff_ang = 2 * acos_val
        return diff_ang

    if integral_options is None:
        integral_options = {"epsrel": 1e-4, "epsabs": 1e-4}

    bd = ms.BinghamDistribution(
        np.eye(4), bingham_z,
        {"norm_const_mode": "numerical",
         "norm_const_options": integral_options}
    )

    def integrand_transformed(x):
        # To avoid unnecessary divisions, this term does not contain the
        # normalization constant. At the end, the result of the integration is
        # divided by it.
        return aad(x, bd.mode) \
               * np.exp(np.dot(x, np.dot(np.diag(bingham_z), x)))

    def integrand(phi1, phi2, phi3):
        sp1 = np.sin(phi1)
        sp2 = np.sin(phi2)
        return integrand_transformed(np.array([
            sp1 * sp2 * np.sin(phi3),
            sp1 * sp2 * np.cos(phi3),
            sp1 * np.cos(phi2),
            np.cos(phi1)
        ])) * (sp1 ** 2.) * sp2

    eaad_int = integrate.tplquad(
        integrand,
        0.0, 2.0 * np.pi,  # phi3
        lambda x: 0.0, lambda x: np.pi,  # phi2
        lambda x, y: 0.0, lambda x, y: np.pi,  # phi1
        **integral_options
    )

    return eaad_int[0] / bd.norm_const


def build_bd_lookup_table(table_type, options, path=None):
    """
    Builds a lookup table for interpolating the bingham normalization
    constant.  If a lookup table with the given options already exists, it is
    loaded and returned instead of building a new one.

    Arguments:
        table_type: Type of lookup table used. May be 'uniform' or 'nonuniform'
        options: Dict cotaining type specific options.
            If type is "uniform" this dict must contain:
                "bounds" = Tuple (lower_bound, upper_bound) representing bounds.
                "num_points" = Number of points per dimension.
            If type is "nonuniform" this dict must contain a key "coords" which
            is a numpy arrays representing the coordinates at which the
            interpolation is evaluated.
        path: absolute path for the lookup table (optional). The default is to
            create a hash based on the options and to use this for constructing
            a file name and placing the file in the precomputed folder.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(table_type)
    hash_obj.update(dill.dumps(options))
    config_hash = hash_obj.hexdigest()

    if not path:
        path = os.path.dirname(__file__) \
               + "/../precomputed/lookup_{}.dill".format(config_hash)

    # Load existing table or create new one.
    if os.path.exists(path):
        with open(path, "rb") as dillfile:
            (serialized_type, serialized_options, res_table, coords) \
                = dill.load(dillfile)
            hash_obj = hashlib.sha256()
            hash_obj.update(serialized_type)
            hash_obj.update(dill.dumps(serialized_options))
            file_config_hash = hash_obj.hexdigest()
            assert file_config_hash == config_hash, \
                "Serialized lookup table does not match given type & options."

    elif table_type == "uniform":
        # Number of points per axis.
        (lbound, rbound) = options["bounds"]
        num_points = options["num_points"]

        assert num_points > 1, \
            "Grid must have more than one point per dimension."

        nc_options = {"epsrel": 1e-3, "epsabs": 1e-7}

        coords = np.linspace(lbound, rbound, num_points)

        res_table = _compute_bd_lookup_table(coords, nc_options)

        with open(path, "wb") as dillfile:
            dill.dump((table_type, options, res_table, coords), dillfile)

    elif table_type == "nonuniform":
        nc_options = {"epsrel": 1e-3, "epsabs": 1e-7}

        coords = options["coords"]

        res_table = _compute_bd_lookup_table(coords, nc_options)

        with open(path, "wb") as dillfile:
            dill.dump((table_type, options, res_table, coords), dillfile)

    else:
        sys.exit("Unknown lookup table type")

    return res_table


def build_vm_lookup_table(options, path=None):
    """
    Builds a lookup table for interpolating the bingham normalization
    constant.  If a lookup table with the given options already exists, it is
    loaded and returned instead of building a new one.

    Arguments:
        options: Dict cotaining table options. It must contain a key "coords"
            which is a numpy arrays representing the coordinates at which the
            interpolation is evaluated.
        path: absolute path for the lookup table (optional). The default is to
            create a hash based on the options and to use this for constructing
            a file name and placing the file in the precomputed folder.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(dill.dumps(options))
    config_hash = hash_obj.hexdigest()

    if not path:
        path = os.path.dirname(__file__) \
            + "/../precomputed/lookup_{}.dill".format(config_hash)

    # Load existing table or create new one.
    if os.path.exists(path):
        with open(path, "rb") as dillfile:
            (serialized_options, res_table) \
                = dill.load(dillfile)
            hash_obj = hashlib.sha256()
            hash_obj.update(dill.dumps(serialized_options))
            file_config_hash = hash_obj.hexdigest()
            assert file_config_hash == config_hash, \
                "Serialized lookup table does not match given type & options."

    else:
        coords = options["coords"]
        res_table = _compute_vm_lookup_table(coords)

        with open(path, "wb") as dillfile:
            dill.dump((options, res_table), dillfile)

    return res_table


def _compute_bd_lookup_table(coords, nc_options):
    num_points = len(coords)

    pool = Pool()

    def nc_wrapper(idx):
        pt_idx = point_indices[idx]

        # Indexing pt_idx in the order 2,1,0 vs. 0,1,2 has no impact
        # on the result as the Bingham normalization constant is agnostic to it.
        # However, the numpy integration that is used to compute it, combines
        # numerical 2d and 1d integration which is why the order matters for the
        # actual computation time.
        #
        # TODO: Make pymanstats choose best order automatically.
        norm_const = ms.BinghamDistribution.normalization_constant(
            np.array(
                [coords[pt_idx[2]], coords[pt_idx[1]], coords[pt_idx[0]], 0.]),
            "numerical", nc_options)
        print("Computing NC for Z=[{}, {}, {}, 0.0]: {}".format(
            coords[pt_idx[2]], coords[pt_idx[1]], coords[pt_idx[0]],
            norm_const))
        return norm_const

    point_indices = list(itertools.combinations_with_replacement(
        range(0, num_points), 3))
    results = pool.map(nc_wrapper, range(len(point_indices)))

    res_tensor = -np.ones((num_points, num_points, num_points))
    for idx_pos, pt_idx in enumerate(point_indices):
        res_tensor[pt_idx[0], pt_idx[1], pt_idx[2]] = results[idx_pos]
        res_tensor[pt_idx[0], pt_idx[2], pt_idx[1]] = results[idx_pos]
        res_tensor[pt_idx[1], pt_idx[0], pt_idx[2]] = results[idx_pos]
        res_tensor[pt_idx[1], pt_idx[2], pt_idx[0]] = results[idx_pos]
        res_tensor[pt_idx[2], pt_idx[0], pt_idx[1]] = results[idx_pos]
        res_tensor[pt_idx[2], pt_idx[1], pt_idx[0]] = results[idx_pos]

    return res_tensor


class AverageMeter(object):
    """Computes and stores the averages over a numbers or dicts of numbers.

    For the dict, this class assumes that no new keys are added during
    the computation.
    """

    def __init__(self):
        self.last_val = 0
        self.avg = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.last_val = val
        n = float(n)
        if type(val) == dict:
            if self.count == 0:
                self.avg = copy.deepcopy(val)
            else:
                for key in val:
                    self.avg[key] *= self.count / (self.count + n)
                    self.avg[key] += val[key] * n / (self.count + n)
        else:
            self.avg *= self.count / (self.count + n)
            self.avg += val * n / (self.count + n)

        self.count += n
        self.last_val = val


def _compute_vm_lookup_table(coords):
    num_points = len(coords)

    pool = Pool()

    def nc_wrapper(idx):
        cur_pt_idx = point_indices[idx]

        log_norm_const = np.log(8.0) + (3. * np.log(np.pi)) \
            + np.log(scipy.special.iv(0, coords[cur_pt_idx[0]])) \
            + np.log(scipy.special.iv(0, coords[cur_pt_idx[1]])) \
            + np.log(scipy.special.iv(0, coords[cur_pt_idx[2]]))

        print("Computing NC for kappas=[{}, {}, {}]: {}".format(
            coords[cur_pt_idx[2]], coords[cur_pt_idx[1]], coords[cur_pt_idx[0]],
            log_norm_const))
        return log_norm_const

    point_indices = list(itertools.combinations_with_replacement(
        range(0, num_points), 3))
    results = pool.map(nc_wrapper, range(len(point_indices)))

    res_tensor = -np.ones((num_points, num_points, num_points))
    for idx_pos, pt_idx in enumerate(point_indices):
        res_tensor[pt_idx[0], pt_idx[1], pt_idx[2]] = results[idx_pos]
        res_tensor[pt_idx[0], pt_idx[2], pt_idx[1]] = results[idx_pos]
        res_tensor[pt_idx[1], pt_idx[0], pt_idx[2]] = results[idx_pos]
        res_tensor[pt_idx[1], pt_idx[2], pt_idx[0]] = results[idx_pos]
        res_tensor[pt_idx[2], pt_idx[0], pt_idx[1]] = results[idx_pos]
        res_tensor[pt_idx[2], pt_idx[1], pt_idx[0]] = results[idx_pos]

    return res_tensor


def vec_to_bingham_z_many(y):
    z = -torch.exp(y).cumsum(1)[:, [2, 1, 0]].unsqueeze(0)
    return z 


def vec_to_bingham_z(y):
    z = -torch.exp(y).cumsum(0)[[2, 1, 0]].unsqueeze(0)
    if not all(z[0][:-1] <= z[0][1:]):
        print(z)
    return z
