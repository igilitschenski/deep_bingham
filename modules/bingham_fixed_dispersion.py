import torch

from modules.gram_schmidt import gram_schmidt_batched
from modules.bingham_loss import batched_logprob
from modules.quaternion_matrix import quaternion_matrix


class BinghamFixedDispersionLoss(object):
    """
    Class for calculating bingham loss assuming a fixed Z.

    Parameters:
        bd_z (list): Values of parameter matrix Z of size 3 (the bingham is four
            dimensional but the last parameter is assumed to be 0). All must be
            negative and in ascending order.
        orthogonalization (str): Orthogonalization method to use. Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.
    """
    def __init__(self, bd_z, orthogonalization="gram_schmidt"):

        self.name = "bingham_fixed_z"
        self.bd_z = bd_z
        self.orthogonalization = orthogonalization

    def __call__(self, target, output):
        """
        Calculates the bingham fixed dispersion log likelihood loss
        on a batch of target-output values.

        Inputs:
           target: Target values at which the likelihood is evaluated
                of shape (N, 4)
           output: Output values from which M is computed, shape
                (N, 16) if orthogonalization is "gram_schmidt" and (N, 4) if it
                is "quaternion_matrix".
        Result:
           loss: The loss of the current batch.
           log_likelihood: Average log likelihood.

        """
        if type(self.bd_z) != torch.Tensor:
            bd_z = torch.tensor([
                [self.bd_z[0], 0, 0, 0],
                [0, self.bd_z[1], 0, 0],
                [0, 0, self.bd_z[2], 0],
                [0, 0, 0, 0]
            ], device=output.device, dtype=output.dtype)

        log_likelihood = 0.0
        bd_m = self._output_to_m(output)

        for i in range(output.shape[0]):
            log_likelihood \
                += self._bingham_loss_fixed_dispersion_single_sample(
                    target[i], bd_m[i], bd_z)
        loss = -log_likelihood
        return loss, log_likelihood / output.shape[0]

    def statistics(self, target, output, epoch):
        """ Reports some additional loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): Network output.
            epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): Bingham parameters and angular deviation.
        """
        stats = {
            "maad": float(maad_quaternion(
                            target, output, self.orthogonalization))
        }

        return stats

    @staticmethod
    def _bingham_loss_fixed_dispersion_single_sample(target, bd_m, bd_z):
        """
        Calculates the bingham likelihood loss on
        a single sample.

        Parameters:
           target: Target value at which the likelihood is
                evaluated
           bd_m: Bingham distribution location and axes parameter of shape
           (1, 4, 4)
           bd_z: Z parameter matrix of shape (1, 4, 4)
        """
        target = target.reshape(1, 4)
        loss = torch.mm(torch.mm(torch.mm(torch.mm(
            target, bd_m), bd_z), torch.t(bd_m)), torch.t(target))
        return loss

    def _output_to_m(self, output):
        """ Creates orthogonal matrix from output.

        Parameters:
            output (torch.Tensor): Output values from which M is extracted,
                shape (batch_size, 16) for gram-schmidt orthogonalization
                and (batch_size, 4) for quaternion_matrix orthogonalization.
        """
        batch_size = output.shape[0]
        if self.orthogonalization == "gram_schmidt":
            reshaped_output = output.reshape(batch_size, 4, 4)
            bd_m = gram_schmidt_batched(reshaped_output)
        elif self.orthogonalization == "modified_gram_schmidt":
            reshaped_output = output.reshape(batch_size, 4, 4)
            bd_m = gram_schmidt_batched(reshaped_output, modified=True)
        elif self.orthogonalization == "quaternion_matrix":
            #bd_m = quaternion_matrix(output)
            raise NotImplementedError
        else:
            raise ValueError("Invalid orthogonalization type.")
        return bd_m

