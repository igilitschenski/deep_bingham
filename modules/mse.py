import torch
import torch.nn as nn
from modules.maad import maad_mse

class MSELoss(object):
    """
    Class for the MSE loss function
    """
    def __init__(self):
        self.loss = nn.MSELoss(reduction='sum')
    def __call__(self, target, output):
        """
        Calculates the MSE loss on a batch of target-output values.
        Target value is the true unit quaternion pose. Output is the predicted
        quaternion after normalization.
 
        Arguments:
            target (torch.Tensor): Target values at which the loss is evaluated of shape (N, 4)
            output (torch.Tensor): Output values of shape (N, 4)

        Returns:
            loss: The loss of the current batch
            log_likelihood: 0. This loss function does not calculate a log likelihood so 0
            is returned.
        """
        return self.loss(target, output), torch.Tensor([0])

    def statistics(self, target, output, cur_epoch=None):
        """ Reports loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): Network output.
            cur_epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): angular deviation.
        """
        return {"maad": maad_mse(target, output.detach())}
