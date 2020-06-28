"""Implementation of the Bingham Mixture Loss"""
import torch

from .maad import angular_loss_single_sample
from .bingham_fixed_dispersion import BinghamFixedDispersionLoss
from .bingham_loss import BinghamLoss
from .gram_schmidt import gram_schmidt_batched
from utils import vec_to_bingham_z_many


class BinghamMixtureLoss(object):
    """ Bingham Mixture Loss

    Computes the log likelihood bingham mixture loss on a batch. Can be
    configured such that for a predefined number of epochs

    Arguments:
        lookup_table_file (str): Path to the location of the lookup table.
        mixture_component_count (int): Number of Bingham mixture components.
        interpolation_kernel (str): The kernel to use for rbf interpolaition
            (can be "multiquadric" or "gaussian").
        fixed_dispersion_stage (int): Number of epochs in which the network is
            trained using a fixed dispersion parameter z.
        fixed_param_z (list): The fixed dispersion parameter Z used for all
            mixture components during the fixed dispersion stage.

    Inputs:
        target (torch.Tensor): Target values at which the likelihood is
            evaluated of shape (N, 4)
        output (torch.Tensor): Output values from which M and Z are extracted of
            shape (N, MIXTURE_COMPONENT_COUNT * 20). The first of the 20 values
            per mixture component is for computing the weight of that component.
            The remaining 19 are passed on to the BinghamLoss class.
    """
    def __init__(self, lookup_table_file, mixture_component_count,
                 interpolation_kernel="multiquadric", fixed_dispersion_stage=25,
                 fixed_param_z=[-1, -1, -1, 0]):

        self._num_components = mixture_component_count
        self._fixed_dispersion_stage = fixed_dispersion_stage
        self._softmax = torch.nn.Softmax(dim=1)
        self._bingham_fixed_dispersion_loss = BinghamFixedDispersionLoss(
            fixed_param_z, orthogonalization="gram_schmidt")
        self._bingham_loss = BinghamLoss(
            lookup_table_file, interpolation_kernel,
            orthogonalization="gram_schmidt")

    def __call__(self, target, output, epoch):
        batch_size = output.shape[0]
        weights = self._softmax(output[:, 0:-1:20])

        log_likelihood = torch.tensor(0., device=output.device, dtype=output.dtype)
        for i in range(batch_size):
            current_likelihood = torch.tensor(
                0., device=output.device, dtype=output.dtype)
            for j in range(self._num_components):
                if epoch < self._fixed_dispersion_stage:
                    bd_log_likelihood = self._bingham_fixed_dispersion_loss(
                        target[i].unsqueeze(0),
                        output[i, (j*20+4):((j+1)*20)].unsqueeze(0))[1]
                else:
                    bd_log_likelihood = self._bingham_loss(
                        target[i].unsqueeze(0),
                        output[i, (j*20+1):((j+1)*20)].unsqueeze(0))[1]

                current_likelihood += weights[i, j] * \
                    torch.exp(bd_log_likelihood).squeeze()

            log_likelihood += torch.log(current_likelihood)

        loss = -log_likelihood
        log_likelihood /= batch_size

        return loss, log_likelihood

    def statistics(self, target, output, epoch):
        """ Reports some additional loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): NN output shaped as loss output parameter.
            epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): Bingham parameters and angular deviation.
        """
        batch_size = output.shape[0]
        weights = self._softmax(output[:, 0:-1:20])

        maad = torch.zeros(
            batch_size, device=output.device, dtype=output.dtype)
        mode_stats = dict()

        for j in range(self._num_components):
            bd_z = torch.mean(vec_to_bingham_z_many(
                output[:, (j*20+1):(j*20+4)]
            ).squeeze(0), 0)
            mode_stats["mode_" + str(j) + "_weight"] \
                = float(torch.mean(weights[:, j]))
            if epoch >= self._fixed_dispersion_stage:
                mode_stats["mode_" + str(j) + "_z_0"] = float(bd_z[0])
                mode_stats["mode_" + str(j) + "_z_1"] = float(bd_z[1])
                mode_stats["mode_" + str(j) + "_z_2"] = float(bd_z[2])

        param_m = torch.zeros((batch_size, self._num_components, 4, 4),
                              device=output.device, dtype=output.dtype)
        for j in range(self._num_components):
            param_m[:, j, :, :] = gram_schmidt_batched(
                output[:, (j * 20 + 4):((j + 1) * 20)].reshape(batch_size, 4, 4)
            )

        # Setting mmaad to 10 such that the minimum succeeds in the first run.
        mmaad = 10. * torch.ones(
            batch_size, device=output.device, dtype=output.dtype)
        for i in range(batch_size):
            for j in range(self._num_components):
                cur_angular_deviation = angular_loss_single_sample(
                    target[i], param_m[i, j, :, 3])

                maad[i] += cur_angular_deviation * weights[i, j]
                mmaad[i] = torch.min(mmaad[i], cur_angular_deviation)

        maad = torch.mean(maad)
        mmaad = torch.mean(mmaad)
        stats = {
            "maad": float(maad),
            "mmaad": float(mmaad)
        }
        stats.update(mode_stats)

        return stats
