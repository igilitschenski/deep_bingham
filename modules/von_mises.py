"""Implementation of von Mises loss function

Code based on:
https://github.com/sergeyprokudin/deep_direct_stat/blob/master/utils/losses.py
"""

import numpy as np
import torch
import math
import sys
from scipy.interpolate import Rbf
import utils
from utils import generate_coordinates
from modules.maad import maad_biternion
from modules.vm_operations import *


class VonMisesLoss(object):
    """
    Computes the von Mises log likelihood loss on a batch of target-output
    values.

    """

    def __init__(self):

        self._bessel_taylor_coefs = torch.tensor(
                [1.00000000e+00, 2.50000000e-01, 1.56250000e-02,
                 4.34027778e-04, 6.78168403e-06])

    def __call__(self, target, output):
        """
        Calculates the von mises likelihood loss on a batch of target-output
        values.

        parameters:
            target: target values at which the likelihood is evaluated
                    of shape (n, 1, 6)
            output: output values from which kappa and biterion representation
             of the angles  are extracted, shape (n, 1, 9)

        returns:
            neg_log_likelihood_loss: the negative sum of the log-likelihood of
                each sample.
            log_likelihood: the average log likelihood.
        """
        log_likelihood = 0
        data_type = output.type()
        for i in range(output.shape[0]):
            angles, kappas = output_to_angles_and_kappas(output[i])
            x = math.radians(target[i][0])
            y = math.radians(target[i][1])
            z = math.radians(target[i][2])
            pan_target = torch.tensor([math.cos(x), math.sin(x)])
            tilt_target = torch.tensor([math.cos(y), math.sin(y)])
            roll_target = torch.tensor([math.cos(z), math.sin(z)])

            if output.is_cuda:
                device = output.get_device()
                pan_target = pan_target.to(device)
                tilt_target = tilt_target.to(device)
                roll_target = roll_target.to(device)

            log1 = self._von_mises_log_likelihood_single_angle(
                pan_target, angles[:2], kappas[0:1])
            log2 = self._von_mises_log_likelihood_single_angle(
                tilt_target, angles[2:4], kappas[1:2])
            log3 = self._von_mises_log_likelihood_single_angle(
                roll_target, angles[4:], kappas[2:])

            log_likelihood += log1 + log2 + log3 

        loss = -log_likelihood
        return loss, log_likelihood / output.shape[0]

    def _von_mises_log_likelihood_single_angle(self, y_true, mu_pred,
                                               kappa_pred):
        r"""
        Compute log-likelihood given data samples and predicted von-mises model
        parameters

        Parameters:
            y_true: true values of an angle in biternion (cos, sin)
                representation.
            mu_pred: predicted mean values of an angle in biternion (cos, sin)
                representation.
            kappa_pred: predicted kappa (inverse variance) values of an angle

        Returns:
            log_likelihood: the von Mises log likelihood.
        """
        cosine_dist = torch.sum(torch.mul(y_true, mu_pred)).reshape([-1, 1])

        if kappa_pred.is_cuda:
            device = kappa_pred.get_device()
            cosine_dist = cosine_dist.to(device)

        norm_const = self._log_bessel_approx_dds(kappa_pred) \
                         + torch.log(torch.tensor(2. * 3.14159))
        log_likelihood = (kappa_pred * cosine_dist) - norm_const

        return log_likelihood.reshape([-1, 1])

    def _log_bessel_approx_dds(self, kappa):
        kappa.reshape([-1, 1])

        def _log_bessel_approx_taylor(cls, x):
            num_coef = cls._bessel_taylor_coefs.shape[0]
            arg = torch.arange(0, num_coef, 1) * 2
            deg = arg.reshape([1, -1])
            n_rows = x.shape[0]
            x_tiled = x.repeat([1, num_coef])
            deg_tiled = deg.repeat([n_rows, 1]).float()
            coef_tiled = cls._bessel_taylor_coefs[0:num_coef].reshape(
                1, num_coef).repeat([n_rows, 1])

            if x.is_cuda:
                device = x.get_device()
                x_tiled = x_tiled.to(device)
                deg_tiled = deg_tiled.to(device)
                coef_tiled = coef_tiled.to(device)

            val = torch.log(
                torch.sum(torch.pow(x_tiled, deg_tiled) * coef_tiled, 1))

            return val.reshape([-1, 1])

        def _log_bessel_approx_large(x):
            return x - 0.5 * torch.log(2 * np.pi * x)

        if kappa[0] > 5:
            return _log_bessel_approx_large(kappa)
        else:
            return _log_bessel_approx_taylor(self, kappa)

    def statistics(self, target, output, epoch=None):

        param_kappas = output_to_kappas(output)
        stats = {"maad" : float(maad_biternion(target, output)), 
                 "kappa_0": float(param_kappas[:, 0].mean()),
                 "kappa_1": float(param_kappas[:, 1].mean()),
                 "kappa_2": float(param_kappas[:, 2].mean())}

        return stats 



